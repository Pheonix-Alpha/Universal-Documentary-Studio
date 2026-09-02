"""
Worker Server - Enhanced with smart model management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import base64
import json
import threading
import time

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
    context: Dict[str, Any]
    duration_seconds: int = 4
    fps: int = 24
    width: int = 576
    height: int = 320
    seed: int = 42
    reference_image: Optional[str] = None


def build_fastapi_app():
    app = FastAPI(title="Universal Documentary Studio Worker - Smart")
    
    @app.get("/health")
    async def health():
        import torch
        return {
            "status": "ok",
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "vram": model_manager.get_vram_status(),
            "storage": model_manager.get_available_storage(),
            "loaded_models": list(model_manager._loaded_models.keys()),
            "capabilities": {
                "clip_ranking": True,
                "video_generation": True,
                "video_models": [
                    {'id': mid, 'name': info['name']}
                    for mid, info in model_manager.MODEL_REGISTRY.items()
                    if info['type'] == 'video'
                ]
            }
        }
    
    @app.get("/models")
    async def get_models():
        """List all models with their status and sizes"""
        models = model_manager.list_models()
        storage = model_manager.get_available_storage()
        vram = model_manager.get_vram_status()
        
        return {
            'models': models,
            'storage': storage,
            'vram': vram,
            'loaded_models': list(model_manager._loaded_models.keys())
        }
    
    @app.post("/models/{model_id}/download")
    async def download_model(model_id: str, background_tasks: BackgroundTasks):
        """Smart download with automatic cleanup"""
        if model_id not in model_manager.MODEL_REGISTRY:
            raise HTTPException(404, f"Model {model_id} not found")
        
        # Check if already installed
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
                    
                    # Only keep last 100 messages
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
                    'error': str(e)
                }
        
        background_tasks.add_task(download_task)
        
        return {"status": "started", "model_id": model_id}
    
    @app.get("/models/{model_id}/progress")
    async def get_download_progress(model_id: str):
        """Get download progress"""
        job = _download_jobs.get(model_id)
        if not job:
            # Check if already installed
            if model_manager.is_installed(model_id):
                return {"status": "installed", "progress": 100, "model_id": model_id}
            return {"status": "not_started", "progress": 0, "model_id": model_id}
        
        return job
    
    @app.delete("/models/{model_id}")
    async def delete_model_endpoint(model_id: str):
        """Delete a model"""
        if model_manager.delete_model(model_id):
            # Clean up any download job
            if model_id in _download_jobs:
                del _download_jobs[model_id]
            return {"status": "deleted", "model_id": model_id}
        return {"status": "not_found", "model_id": model_id}
    
    @app.get("/models/storage")
    async def get_storage_info():
        """Get storage information"""
        return model_manager.get_available_storage()
    
    @app.post("/models/cleanup")
    async def cleanup_models(needed_gb: float = 1.0):
        """Manually trigger cleanup"""
        freed = model_manager._smart_cleanup(needed_gb)
        return {"freed_gb": freed, "storage": model_manager.get_available_storage()}
    
    # Keep ranking endpoints
    @app.post("/rank")
    async def rank(request: RankRequest):
        # ... existing code ...
        pass
    
    @app.post("/video/generate")
    async def generate_video(request: VideoGenRequest):
        # Use smart loading to handle VRAM
        try:
            # This will automatically unload other models if needed
            video_model = model_manager.load_model_into_vram(
                request.model_id, 
                'cuda' if torch.cuda.is_available() else 'cpu'
            )
            
            # Generate video using the loaded model
            result = video_models.generate_with_loaded_model(
                video_model,
                request.prompt,
                request.context,
                request.duration_seconds,
                request.fps,
                request.width,
                request.height,
                request.seed,
                request.reference_image
            )
            return result
        except Exception as e:
            raise HTTPException(500, detail=str(e))
    
    return app