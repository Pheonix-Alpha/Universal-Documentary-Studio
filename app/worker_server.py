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
    """Build the FastAPI app with all endpoints"""
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

    # ---- VIDEO GENERATION ENDPOINT ----
    @app.post("/video/generate")
    async def generate_video(request: VideoGenRequest):
        """Generate video with auto-download on worker"""
        try:
            print(f"🎬 Worker received video generation request")
            print(f"   Model: {request.model_id}")
            print(f"   Prompt: {request.prompt[:100]}...")
            print(f"   Duration: {request.duration_seconds}s")
            print(f"   Seed: {request.seed}")

            # ---- STEP 1: Check if model is installed on worker ----
            if not model_manager.is_installed(request.model_id):
                print(f"📥 Model not installed on worker. Downloading...")
                
                # Download on worker
                for pct, msg in model_manager.smart_download_model(request.model_id):
                    print(f"  [{pct}%] {msg}")
                
                print(f"✅ Model downloaded on worker!")

            # ---- STEP 2: Load into VRAM on worker ----
            print(f"🔄 Loading model into VRAM on worker...")
            model_manager.auto_download_and_load_model(request.model_id)
            print(f"✅ Model loaded on worker")

            # ---- STEP 3: Generate video on worker ----
            print(f"🎬 Generating video on worker...")
            
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
                # Fallback using direct model
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
                print(f"✅ Video generated on worker: {len(result['video_data'])} bytes")
                return result
            else:
                raise RuntimeError("Video generation returned empty result")

        except Exception as e:
            print(f"❌ Worker generation failed: {e}")
            import traceback
            traceback.print_exc()

            # ---- FALLBACK: Use fallback video on worker ----
            try:
                print("🔄 Using fallback video on worker...")
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
                    fallback['metadata']['note'] = "Fallback video generated on worker"
                    print(f"✅ Fallback video: {len(fallback['video_data'])} bytes")
                    return fallback
            except Exception as fallback_error:
                print(f"❌ Fallback failed: {fallback_error}")

            return JSONResponse(
                status_code=500,
                content={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                    "model_id": request.model_id
                }
            )

    return app


def _generate_with_model(model_id: str, prompt: str, duration: int, fps: int, width: int, height: int, seed: int) -> Dict[str, Any]:
    """Generate video using loaded model (fallback)"""
    try:
        import torch
        
        model_obj = model_manager._loaded_models.get(model_id)
        if not model_obj:
            raise ValueError(f"Model {model_id} not loaded")
        
        pipeline = model_obj.get('pipeline')
        device = model_obj.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        
        if not pipeline:
            raise ValueError("Pipeline not available")
        
        generator = torch.Generator(device=device).manual_seed(seed)
        
        if hasattr(pipeline, 'generate'):
            frames = pipeline.generate(
                prompt=prompt,
                num_frames=min(duration * fps, 24),
                height=height,
                width=width,
                generator=generator
            )
        else:
            frames = pipeline(
                prompt,
                num_frames=min(duration * fps, 24),
                num_inference_steps=20,
                height=height,
                width=width,
                generator=generator
            ).frames
        
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