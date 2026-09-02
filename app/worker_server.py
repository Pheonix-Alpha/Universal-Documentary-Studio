"""
Worker Server - Complete implementation with auto-download and fallback
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64
import json
import threading
import time
import traceback
import torch
import os

from app import model_manager, video_models, clip_ranker

# Store active download jobs
_download_jobs = {}


class RankRequest(BaseModel):
    text: str
    candidates: List[Dict[str, Any]]
    model_id: str
    top_k: int


class VideoGenRequest(BaseModel):
    prompt: str
    model_id: str
    context: Dict[str, Any] = {}
    duration_seconds: int = 4
    fps: int = 24
    width: int = 576
    height: int = 320
    seed: int = 42
    reference_image: Optional[str] = None


class VideoGenResponse(BaseModel):
    video_data: str  # base64 encoded
    duration: int
    fps: int
    width: int
    height: int
    metadata: Dict[str, Any]


def build_fastapi_app():
    app = FastAPI(title="Universal Documentary Studio Worker")

    @app.get("/health")
    async def health():
        try:
            import torch

            return {
                "status": "ok",
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "vram": (
                    model_manager.get_vram_status()
                    if hasattr(model_manager, "get_vram_status")
                    else {}
                ),
                "storage": (
                    model_manager.get_available_storage()
                    if hasattr(model_manager, "get_available_storage")
                    else {}
                ),
                "installed_models": (
                    model_manager.get_installed_models()
                    if hasattr(model_manager, "get_installed_models")
                    else []
                ),
                "capabilities": {"clip_ranking": True, "video_generation": True},
            }
        except Exception as e:
            return {"status": "error", "error": str(e)}

    @app.get("/models")
    async def get_models():
        """List all models with their status"""
        try:
            if hasattr(model_manager, "list_models"):
                models = model_manager.list_models()
            else:
                models = []

            # Add video models
            if hasattr(video_models, "list_video_models"):
                video_models_list = video_models.list_video_models()
            else:
                video_models_list = []

            return {
                "models": models,
                "video_models": video_models_list,
                "installed_models": (
                    model_manager.get_installed_models()
                    if hasattr(model_manager, "get_installed_models")
                    else []
                ),
                "storage": (
                    model_manager.get_available_storage()
                    if hasattr(model_manager, "get_available_storage")
                    else {}
                ),
                "vram": (
                    model_manager.get_vram_status()
                    if hasattr(model_manager, "get_vram_status")
                    else {}
                ),
            }
        except Exception as e:
            return {"error": str(e), "models": [], "video_models": []}

    @app.post("/models/{model_id}/download")
    async def download_model(model_id: str, background_tasks: BackgroundTasks):
        """Smart download with automatic cleanup"""
        try:
            if model_id not in model_manager.MODEL_REGISTRY:
                raise HTTPException(404, f"Model {model_id} not found")

            if model_manager.is_installed(model_id):
                return {"status": "already_installed", "model_id": model_id}

            # Start download in background
            def download_task():
                try:
                    job_data = {
                        "model_id": model_id,
                        "progress": 0,
                        "message": "Starting...",
                        "status": "downloading",
                    }
                    _download_jobs[model_id] = job_data

                    for pct, msg in model_manager.smart_download_model(model_id):
                        job_data["progress"] = pct
                        job_data["message"] = msg
                        job_data["status"] = "downloading" if pct < 100 else "completed"

                        if "messages" not in job_data:
                            job_data["messages"] = []
                        job_data["messages"].append(msg)
                        if len(job_data["messages"]) > 100:
                            job_data["messages"] = job_data["messages"][-100:]

                    _download_jobs[model_id]["status"] = "completed"

                except Exception as e:
                    _download_jobs[model_id] = {
                        "model_id": model_id,
                        "status": "failed",
                        "error": str(e),
                        "traceback": traceback.format_exc(),
                    }

            background_tasks.add_task(download_task)

            return {"status": "started", "model_id": model_id}

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "traceback": traceback.format_exc()},
            )

    @app.get("/models/{model_id}/progress")
    async def get_download_progress(model_id: str):
        """Get download progress"""
        try:
            job = _download_jobs.get(model_id)
            if not job:
                if model_manager.is_installed(model_id):
                    return {
                        "status": "installed",
                        "progress": 100,
                        "model_id": model_id,
                    }
                return {"status": "not_started", "progress": 0, "model_id": model_id}

            return job

        except Exception as e:
            return {"status": "error", "error": str(e)}

    @app.delete("/models/{model_id}")
    async def delete_model_endpoint(model_id: str):
        """Delete a model"""
        try:
            if model_manager.delete_model(model_id):
                if model_id in _download_jobs:
                    del _download_jobs[model_id]
                return {"status": "deleted", "model_id": model_id}
            return {"status": "not_found", "model_id": model_id}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

    @app.get("/models/storage")
    async def get_storage_info():
        """Get storage information"""
        try:
            if hasattr(model_manager, "get_available_storage"):
                return model_manager.get_available_storage()
            return {"error": "Storage info not available"}
        except Exception as e:
            return {"error": str(e)}

    @app.post("/models/cleanup")
    async def cleanup_models(needed_gb: float = 1.0):
        """Manually trigger cleanup"""
        try:
            if hasattr(model_manager, "_smart_cleanup"):
                freed = model_manager._smart_cleanup(needed_gb)
                storage = (
                    model_manager.get_available_storage()
                    if hasattr(model_manager, "get_available_storage")
                    else {}
                )
                return {"freed_gb": freed, "storage": storage}
            return {"error": "Cleanup not available"}
        except Exception as e:
            return {"error": str(e)}

    # ---- CLIP Ranking Endpoint ----
    @app.post("/rank")
    async def rank(request: RankRequest):
        """Rank candidates using CLIP"""
        try:
            if not hasattr(clip_ranker, "rank_candidates"):
                raise HTTPException(500, "CLIP ranking not available")

            result = clip_ranker.rank_candidates(
                text=request.text,
                candidates=request.candidates,
                model_id=request.model_id,
                top_k=request.top_k,
            )
            return result

        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "traceback": traceback.format_exc()},
            )

    # ---- VIDEO GENERATION ENDPOINT - COMPLETE FIX ----
    @app.post("/video/generate")
    async def generate_video(request: VideoGenRequest):
        """Generate video with auto-download and fallback"""
        try:
            print(f"🎬 Video Generation Request")
            print(f"   Model: {request.model_id}")
            print(f"   Prompt: {request.prompt[:100]}...")
            print(f"   Duration: {request.duration_seconds}s")
            print(f"   Seed: {request.seed}")

            # ---- STEP 1: AUTO-DOWNLOAD if not installed ----
            if not model_manager.is_installed(request.model_id):
                print(f"📥 Model not installed. Auto-downloading {request.model_id}...")
                
                # Download with progress
                for pct, msg in model_manager.smart_download_model(request.model_id):
                    print(f"  [{pct}%] {msg}")
                
                print(f"✅ Model downloaded successfully!")

            # ---- STEP 2: Load into VRAM (auto-unloads old models) ----
            print(f"🔄 Loading model into VRAM...")
            model_manager.auto_download_and_load_model(request.model_id)
            print(f"✅ Model loaded into VRAM")

            # ---- STEP 3: Generate video ----
            print(f"🎬 Generating video...")
            
            # Check if video_models has the function
            if hasattr(video_models, 'generate_video'):
                result = video_models.generate_video(
                    prompt=request.prompt,
                    model_id=request.model_id,
                    context=request.context,
                    duration_seconds=request.duration_seconds,
                    fps=request.fps,
                    width=request.width,
                    height=request.height,
                    seed=request.seed,
                    reference_image=request.reference_image
                )
            else:
                # Fallback to using the model directly
                result = _generate_with_model(
                    request.model_id,
                    request.prompt,
                    request.duration_seconds,
                    request.fps,
                    request.width,
                    request.height,
                    request.seed
                )

            # ---- STEP 4: Return result ----
            if result and result.get('video_data'):
                print(f"✅ Video generated: {len(result['video_data'])} bytes")
                return result
            else:
                raise RuntimeError("Video generation returned empty result")

        except Exception as e:
            error_msg = f"❌ Video generation failed: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())

            # ---- FALLBACK: Use fallback video ----
            try:
                print("🔄 Using fallback video...")
                fallback = video_models._generate_fallback_video(
                    request.prompt,
                    request.duration_seconds,
                    request.fps,
                    request.width,
                    request.height,
                    request.seed
                )
                
                if fallback and fallback.get('video_data'):
                    fallback['metadata']['error'] = str(e)
                    fallback['metadata']['note'] = "Fallback video generated"
                    print(f"✅ Fallback video: {len(fallback['video_data'])} bytes")
                    return fallback
            except Exception as fallback_error:
                print(f"❌ Fallback failed: {fallback_error}")

            # Return error response
            return JSONResponse(
                status_code=500,
                content={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "model_id": request.model_id,
                    "prompt": request.prompt[:100]
                }
            )

    return app


def _generate_with_model(model_id: str, prompt: str, duration: int, fps: int, width: int, height: int, seed: int) -> Dict[str, Any]:
    """
    Generate video using the loaded model.
    This is a fallback if video_models.generate_video is not available.
    """
    try:
        import torch
        from diffusers import DiffusionPipeline
        
        # Get the loaded model
        model_obj = model_manager._loaded_models.get(model_id)
        if not model_obj:
            raise ValueError(f"Model {model_id} not loaded")
        
        pipeline = model_obj.get('pipeline')
        device = model_obj.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        if not pipeline:
            raise ValueError("Pipeline not available")
        
        # Generate
        generator = torch.Generator(device=device).manual_seed(seed)
        
        # Handle different model types
        if hasattr(pipeline, 'generate'):
            # Some pipelines use generate()
            frames = pipeline.generate(
                prompt=prompt,
                num_frames=min(duration * fps, 24),
                height=height,
                width=width,
                generator=generator
            )
        else:
            # Standard diffusion pipeline
            frames = pipeline(
                prompt,
                num_frames=min(duration * fps, 24),
                num_inference_steps=20,
                height=height,
                width=width,
                generator=generator
            ).frames
        
        # Convert to video bytes
        video_bytes = video_models._frames_to_video_bytes(frames, fps)
        
        return {
            'video_data': base64.b64encode(video_bytes).decode('utf-8'),
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {'model': model_id, 'seed': seed}
        }
        
    except Exception as e:
        raise RuntimeError(f"Model generation failed: {e}")


# ---- Additional helper for video_models ----
def _generate_fallback_video(prompt: str, duration: int, fps: int, width: int, height: int, seed: int) -> Dict[str, Any]:
    """
    Simple fallback video generation - always works
    """
    try:
        import cv2
        import numpy as np
        
        num_frames = min(duration * fps, 30)
        frames = []
        
        # Use seed
        np.random.seed(seed)
        
        for i in range(num_frames):
            # Create frame with gradient
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            
            # Animated gradient
            phase = i / max(1, num_frames)
            r = int(80 + 155 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi + seed)))
            g = int(80 + 155 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi + 2.094 + seed)))
            b = int(80 + 155 * (0.5 + 0.5 * np.sin(phase * 2 * np.pi + 4.188 + seed)))
            
            for y in range(height):
                ratio = y / max(1, height)
                color = (
                    int(r * (0.5 + 0.5 * (1 - ratio))),
                    int(g * (0.5 + 0.5 * (1 - ratio))),
                    int(b * (0.5 + 0.5 * (1 - ratio)))
                )
                frame[y, :] = color
            
            # Add text
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = prompt[:50]
            text_size = cv2.getTextSize(text, font, 0.6, 2)[0]
            text_x = (width - text_size[0]) // 2
            text_y = (height + text_size[1]) // 2
            
            # Text shadow + main
            cv2.putText(frame, text, (text_x + 1, text_y + 1), font, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, text, (text_x, text_y), font, 0.6, (255, 255, 255), 2)
            
            frames.append(frame)
        
        # Convert to video bytes
        import tempfile
        
        with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(tmp.name, fourcc, fps, (width, height))
            for frame in frames:
                out.write(frame)
            out.release()
            
            with open(tmp.name, 'rb') as f:
                video_bytes = f.read()
            
            os.unlink(tmp.name)
        
        return {
            'video_data': base64.b64encode(video_bytes).decode('utf-8'),
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {
                'model': 'fallback',
                'seed': seed,
                'note': 'Fallback video generated'
            }
        }
        
    except Exception as e:
        print(f"Fallback generation failed: {e}")
        # Return empty
        return {
            'video_data': '',
            'duration': duration,
            'fps': fps,
            'width': width,
            'height': height,
            'metadata': {'error': str(e)}
        }


# Patch video_models if needed
if not hasattr(video_models, '_generate_fallback_video'):
    video_models._generate_fallback_video = _generate_fallback_video