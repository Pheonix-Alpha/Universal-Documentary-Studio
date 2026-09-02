"""
Worker Server - Complete implementation with proper error handling
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
                "vram": model_manager.get_vram_status() if hasattr(model_manager, 'get_vram_status') else {},
                "storage": model_manager.get_available_storage() if hasattr(model_manager, 'get_available_storage') else {},
                "capabilities": {
                    "clip_ranking": True,
                    "video_generation": True
                }
            }
        except Exception as e:
            return {
                "status": "error",
                "error": str(e)
            }
    
    @app.get("/models")
    async def get_models():
        """List all models with their status"""
        try:
            if hasattr(model_manager, 'list_models'):
                models = model_manager.list_models()
            else:
                models = []
            
            # Add video models
            if hasattr(video_models, 'list_video_models'):
                video_models_list = video_models.list_video_models()
            else:
                video_models_list = []
            
            return {
                'models': models,
                'video_models': video_models_list,
                'storage': model_manager.get_available_storage() if hasattr(model_manager, 'get_available_storage') else {},
                'vram': model_manager.get_vram_status() if hasattr(model_manager, 'get_vram_status') else {}
            }
        except Exception as e:
            return {
                'error': str(e),
                'models': [],
                'video_models': []
            }
    
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
                    job_data = {'model_id': model_id, 'progress': 0, 'message': 'Starting...', 'status': 'downloading'}
                    _download_jobs[model_id] = job_data
                    
                    for pct, msg in model_manager.smart_download_model(model_id):
                        job_data['progress'] = pct
                        job_data['message'] = msg
                        job_data['status'] = 'downloading' if pct < 100 else 'completed'
                        
                        if 'messages' not in job_data:
                            job_data['messages'] = []
                        job_data['messages'].append(msg)
                        if len(job_data['messages']) > 100:
                            job_data['messages'] = job_data['messages'][-100:]
                    
                    _download_jobs[model_id]['status'] = 'completed'
                    
                except Exception as e:
                    _download_jobs[model_id] = {
                        'model_id': model_id,
                        'status': 'failed',
                        'error': str(e),
                        'traceback': traceback.format_exc()
                    }
            
            background_tasks.add_task(download_task)
            
            return {"status": "started", "model_id": model_id}
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    @app.get("/models/{model_id}/progress")
    async def get_download_progress(model_id: str):
        """Get download progress"""
        try:
            job = _download_jobs.get(model_id)
            if not job:
                if model_manager.is_installed(model_id):
                    return {"status": "installed", "progress": 100, "model_id": model_id}
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
            return JSONResponse(
                status_code=500,
                content={"error": str(e)}
            )
    
    @app.get("/models/storage")
    async def get_storage_info():
        """Get storage information"""
        try:
            if hasattr(model_manager, 'get_available_storage'):
                return model_manager.get_available_storage()
            return {"error": "Storage info not available"}
        except Exception as e:
            return {"error": str(e)}
    
    @app.post("/models/cleanup")
    async def cleanup_models(needed_gb: float = 1.0):
        """Manually trigger cleanup"""
        try:
            if hasattr(model_manager, '_smart_cleanup'):
                freed = model_manager._smart_cleanup(needed_gb)
                storage = model_manager.get_available_storage() if hasattr(model_manager, 'get_available_storage') else {}
                return {"freed_gb": freed, "storage": storage}
            return {"error": "Cleanup not available"}
        except Exception as e:
            return {"error": str(e)}
    
    # ---- CLIP Ranking Endpoint ----
    @app.post("/rank")
    async def rank(request: RankRequest):
        """Rank candidates using CLIP"""
        try:
            if not hasattr(clip_ranker, 'rank_candidates'):
                raise HTTPException(500, "CLIP ranking not available")
            
            result = clip_ranker.rank_candidates(
                text=request.text,
                candidates=request.candidates,
                model_id=request.model_id,
                top_k=request.top_k
            )
            return result
            
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"error": str(e), "traceback": traceback.format_exc()}
            )
    
    # ---- VIDEO GENERATION ENDPOINT - FIXED ----
    @app.post("/video/generate")
    async def generate_video(request: VideoGenRequest):
        """Generate video from prompt using the specified model"""
        try:
            print(f"🎬 Received video generation request")
            print(f"   Model: {request.model_id}")
            print(f"   Prompt: {request.prompt[:100]}...")
            print(f"   Duration: {request.duration_seconds}s")
            print(f"   Seed: {request.seed}")
            
            # Check if model is installed
            if hasattr(video_models, 'is_video_model_installed'):
                if not video_models.is_video_model_installed(request.model_id):
                    # Try to check if it's installed in the main model manager
                    if hasattr(model_manager, 'is_installed'):
                        if not model_manager.is_installed(request.model_id):
                            raise HTTPException(
                                400, 
                                f"Model {request.model_id} not installed. Please download it first."
                            )
            
            # Check if we have enough VRAM
            if torch.cuda.is_available():
                vram_used = torch.cuda.memory_allocated(0) / (1024**3)
                vram_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   VRAM: {vram_used:.2f}GB / {vram_total:.2f}GB")
                
                if vram_used > vram_total * 0.85:
                    # Try to free up VRAM
                    torch.cuda.empty_cache()
                    print(f"   Cleared CUDA cache")
            
            # Generate the video
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
            
            # Verify result
            if not result or 'video_data' not in result:
                raise RuntimeError("Video generation returned empty result")
            
            print(f"✅ Video generated successfully: {len(result['video_data'])} bytes")
            
            return result
            
        except HTTPException:
            raise
        except Exception as e:
            error_msg = f"Video generation failed: {str(e)}\n{traceback.format_exc()}"
            print(f"❌ {error_msg}")
            
            # Return a fallback video if possible
            try:
                # Try to generate a fallback video
                fallback = video_models._generate_fallback_video(
                    request.prompt,
                    request.duration_seconds,
                    request.fps,
                    request.width,
                    request.height,
                    request.seed
                )
                if fallback and fallback.get('video_data'):
                    print("⚠️ Returning fallback video")
                    return fallback
            except:
                pass
            
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