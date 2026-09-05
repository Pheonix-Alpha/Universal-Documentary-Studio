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
from app import runtime_manager
from app import model_manager, video_models, clip_ranker

# Store active download jobs
_download_jobs = {}

# ------------------------------------------------------------
# VIDEO JOB MANAGEMENT
# ------------------------------------------------------------

_video_jobs = {}

# Only one GPU video generation at a time.
# This protects a 14-16 GB GPU from multiple SVD jobs.
_video_gpu_lock = threading.Lock()


def _create_video_job() -> str:
    """Create a unique asynchronous video job."""
    import uuid

    job_id = f"video_{uuid.uuid4().hex}"

    _video_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "progress": 0,
        "message": "Job queued",
        "result": None,
        "error": None,
        "created_at": time.time(),
        "started_at": None,
        "completed_at": None,
    }

    return job_id


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

            runtime_info = runtime_manager.get_runtime_info()

            return {
            "status": "ok",

            # Worker identity
            "runtime_id": runtime_info["runtime_id"],
            "platform": runtime_info["platform"],

            # Runtime hardware
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "gpu_count": runtime_info["gpu_count"],
            "gpus": runtime_info["gpus"],

            # Existing worker information
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

            # Runtime capabilities
            "capabilities": {
                "clip_ranking": True,
                "video_generation": True,
                "cuda": runtime_info["gpu_count"] > 0,
                "multi_gpu": runtime_info["gpu_count"] > 1,
            },
        }

        except Exception as e:
          return {
            "status": "error",
            "error": str(e),
        }

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
    # ---- VIDEO GENERATION ENDPOINT ----
    @app.post("/video/generate")
    async def generate_video(request: VideoGenRequest):
        """
        Start asynchronous video generation.

        IMPORTANT:
        This endpoint returns immediately with a job_id.
        GPU generation happens in a background thread so
        Cloudflare does not have to keep a long HTTP request alive.
        """

        try:
            job_id = _create_video_job()

            print("")
            print("=" * 70)
            print("🎬 NEW VIDEO JOB")
            print(f"   Job ID: {job_id}")
            print(f"   Model: {request.model_id}")
            print(f"   Prompt: {request.prompt[:100]}...")
            print(f"   Duration: {request.duration_seconds}s")
            print(f"   Seed: {request.seed}")
            print("=" * 70)

            thread = threading.Thread(
                target=_run_video_job,
                args=(job_id, request),
                daemon=True,
            )

            thread.start()

            return {
                "job_id": job_id,
                "status": "queued",
                "message": "Video generation started",
            }

        except Exception as e:
            print(f"❌ Failed to start video job: {e}")
            traceback.print_exc()

            return JSONResponse(
                status_code=500,
                content={
                    "error": str(e),
                    "traceback": traceback.format_exc(),
                },
            )
    @app.get("/video/status/{job_id}")
    async def video_status(job_id: str):
        """Return current status of an asynchronous video job."""

        job = _video_jobs.get(job_id)

        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Video job {job_id} not found",
            )

        return {
            "job_id": job["job_id"],
            "status": job["status"],
            "progress": job.get("progress", 0),
            "message": job.get("message", ""),
            "error": job.get("error"),
            "created_at": job.get("created_at"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }
    
    @app.get("/video/result/{job_id}")
    async def video_result(job_id: str):
        """Return completed video result."""

        job = _video_jobs.get(job_id)

        if not job:
            raise HTTPException(
                status_code=404,
                detail=f"Video job {job_id} not found",
            )

        if job["status"] != "completed":
            return JSONResponse(
                status_code=409,
                content={
                    "job_id": job_id,
                    "status": job["status"],
                    "progress": job.get("progress", 0),
                    "message": job.get("message", ""),
                    "error": job.get("error"),
                },
            )

        if not job.get("result"):
            raise HTTPException(
                status_code=500,
                detail="Job completed but result is missing",
            )

        return job["result"]

    return app


def _run_video_job(job_id: str, request: VideoGenRequest):
    """
    Execute a video generation job in the background.

    The HTTP request is already finished by the time this runs.
    """

    job = _video_jobs.get(job_id)

    if not job:
        print(f"❌ Video job not found: {job_id}")
        return

    try:
        # --------------------------------------------------------
        # WAIT FOR GPU
        # --------------------------------------------------------

        job["status"] = "queued"
        job["progress"] = 0
        job["message"] = "Waiting for GPU..."

        print(f"⏳ [{job_id}] Waiting for GPU...")

        with _video_gpu_lock:

            job["started_at"] = time.time()

            # ----------------------------------------------------
            # STEP 1: MODEL DOWNLOAD
            # ----------------------------------------------------

            job["status"] = "loading"
            job["progress"] = 5
            job["message"] = "Checking worker model..."

            print(f"🔍 [{job_id}] Checking model: {request.model_id}")

            if not model_manager.is_installed(request.model_id):

                print(f"📥 [{job_id}] Model not installed. " f"Downloading...")

                job["status"] = "downloading"
                job["progress"] = 5
                job["message"] = "Downloading model..."

                for pct, msg in model_manager.smart_download_model(request.model_id):
                    job["progress"] = min(25, 5 + pct * 0.20)
                    job["message"] = msg

                    print(f"  [{pct}%] {msg}")

                print(f"✅ [{job_id}] Model downloaded")

            # ----------------------------------------------------
            # STEP 2: LOAD MODEL
            # ----------------------------------------------------

            job["status"] = "loading"
            job["progress"] = 25
            job["message"] = "Loading model into VRAM..."

            print(f"🔄 [{job_id}] Loading model into VRAM...")

            model_manager.auto_download_and_load_model(request.model_id)

            print(f"✅ [{job_id}] Model loaded on worker")

            # ----------------------------------------------------
            # STEP 3: GENERATE
            # ----------------------------------------------------

            job["status"] = "generating"
            job["progress"] = 30
            job["message"] = "Generating video..."

            print(f"🎬 [{job_id}] Generating video...")

            if hasattr(video_models, "generate_video"):

                result = video_models.generate_video(
                    prompt=request.prompt,
                    model_id=request.model_id,
                    context=request.context,
                    duration_seconds=request.duration_seconds,
                    fps=request.fps,
                    width=request.width,
                    height=request.height,
                    seed=request.seed,
                    reference_image=request.reference_image,
                )

            else:

                result = _generate_with_model(
                    request.model_id,
                    request.prompt,
                    request.duration_seconds,
                    request.fps,
                    request.width,
                    request.height,
                    request.seed,
                )

            # ----------------------------------------------------
            # STEP 4: VERIFY RESULT
            # ----------------------------------------------------

            if not result or not result.get("video_data"):
                raise RuntimeError("Video generation returned empty result")

            job["status"] = "encoding"
            job["progress"] = 90
            job["message"] = "Video generated, finalizing..."

            print(
                f"✅ [{job_id}] Video generated: " f"{len(result['video_data'])} bytes"
            )

            # ----------------------------------------------------
            # STEP 5: STORE RESULT
            # ----------------------------------------------------

            job["result"] = result
            job["status"] = "completed"
            job["progress"] = 100
            job["message"] = "Video generation completed"
            job["completed_at"] = time.time()

            elapsed = job["completed_at"] - job["started_at"]

            print("")
            print("=" * 70)
            print(f"✅ [{job_id}] VIDEO JOB COMPLETED")
            print(f"   Time: {elapsed:.1f}s")
            print("=" * 70)
            print("")

    except Exception as e:

        print(f"❌ [{job_id}] Worker generation failed: {e}")
        traceback.print_exc()

        # --------------------------------------------------------
        # FALLBACK
        # --------------------------------------------------------

        try:

            print(f"🔄 [{job_id}] Trying fallback video...")

            job["status"] = "fallback"
            job["progress"] = 90
            job["message"] = "Primary generation failed, using fallback..."

            fallback = video_models._generate_fallback_video(
                request.prompt,
                request.duration_seconds,
                request.fps,
                request.width,
                request.height,
                request.seed,
            )

            if fallback and fallback.get("video_data"):

                fallback.setdefault("metadata", {})
                fallback["metadata"]["error"] = str(e)
                fallback["metadata"]["note"] = "Fallback video generated on worker"

                job["result"] = fallback
                job["status"] = "completed"
                job["progress"] = 100
                job["message"] = "Fallback video completed"
                job["completed_at"] = time.time()

                print(f"✅ [{job_id}] Fallback video completed")

                return

        except Exception as fallback_error:

            print(f"❌ [{job_id}] Fallback failed: " f"{fallback_error}")

        # --------------------------------------------------------
        # FINAL FAILURE
        # --------------------------------------------------------

        job["status"] = "failed"
        job["progress"] = 0
        job["message"] = "Video generation failed"
        job["error"] = str(e)
        job["traceback"] = traceback.format_exc()
        job["completed_at"] = time.time()


def _generate_with_model(
    model_id: str,
    prompt: str,
    duration: int,
    fps: int,
    width: int,
    height: int,
    seed: int,
) -> Dict[str, Any]:
    """Generate video using loaded model (fallback)"""
    try:
        import torch

        model_obj = model_manager._loaded_models.get(model_id)
        if not model_obj:
            raise ValueError(f"Model {model_id} not loaded")

        pipeline = model_obj.get("pipeline")
        device = model_obj.get("device", "cuda" if torch.cuda.is_available() else "cpu")

        if not pipeline:
            raise ValueError("Pipeline not available")

        generator = torch.Generator(device=device).manual_seed(seed)

        if hasattr(pipeline, "generate"):
            frames = pipeline.generate(
                prompt=prompt,
                num_frames=min(duration * fps, 24),
                height=height,
                width=width,
                generator=generator,
            )
        else:
            frames = pipeline(
                prompt,
                num_frames=min(duration * fps, 24),
                num_inference_steps=20,
                height=height,
                width=width,
                generator=generator,
            ).frames

        video_bytes = video_models._frames_to_video_bytes(frames, fps)

        return {
            "video_data": base64.b64encode(video_bytes).decode("utf-8"),
            "duration": duration,
            "fps": fps,
            "width": width,
            "height": height,
            "metadata": {"model": model_id, "seed": seed},
        }

    except Exception as e:
        raise RuntimeError(f"Model generation failed: {e}")
