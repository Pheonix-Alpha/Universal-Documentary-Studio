"""
Worker Client - Handles communication with GPU workers
"""

from typing import List, Dict, Any, Optional
import requests
import time
import json
import base64
from io import BytesIO
from PIL import Image
import itertools
from app import worker_registry

# Worker registry
_workers = {}
_rr_counter = itertools.count()

def add_worker(url: str, label: str = None) -> str:
    """Add a worker and register its runtime information."""

    # Normalize URL
    url = url.strip()

    if not url.startswith(("http://", "https://")):
        url = f"https://{url}"

    url = url.rstrip("/")

    # Check if already exists
    for worker_id, w in _workers.items():
        if w["url"] == url:
            return worker_id

    # Create temporary local worker entry
    worker_id = f"worker_{int(time.time())}"

    _workers[worker_id] = {
        "id": worker_id,
        "url": url,
        "label": label or url,
        "added_at": time.time(),
        "last_health_check": 0,
        "connected": False,
        "status": "unknown",
        "device": "unknown",
        "capabilities": {},
        "load": 0,
    }

    # Initial health check
    connected, _ = check_worker(worker_id)

    if connected:
        worker = _workers[worker_id]

        # Register the worker in the Main registry
        worker_registry.register_worker({
            "runtime_id": worker["runtime_id"],
            "platform": worker.get("platform", "unknown"),
            "gpu_count": worker.get("gpu_count", 0),
            "gpus": worker.get("gpus", []),
            "capabilities": worker.get("capabilities", {}),
            "endpoint": worker["url"],
            "label": worker["label"],
        })

    return worker_id

def remove_worker(worker_id: str) -> bool:
    """Remove a worker from the registry"""
    if worker_id in _workers:
        del _workers[worker_id]
        return True
    return False


def check_worker(worker_id: str) -> tuple:
    """Check a worker's health and update its status"""
    worker = _workers.get(worker_id)
    if not worker:
        return False, "Worker not found"

    try:
        response = requests.get(f"{worker['url']}/health", timeout=5)
        if response.status_code == 200:
            data = response.json()
            worker["runtime_id"] = data.get("runtime_id")
            worker["platform"] = data.get("platform", "unknown")
            worker["gpu_count"] = data.get("gpu_count", 0)
            worker["gpus"] = data.get("gpus", [])
            worker["connected"] = True
            worker["status"] = data.get("status", "ok")
            worker["device"] = data.get("device", "unknown")
            worker["capabilities"] = data.get("capabilities", {})
            worker["last_health_check"] = time.time()
            return True, "Connected"
    except Exception as e:
        print(f"Worker health check failed: {e}")

    worker["connected"] = False
    worker["status"] = "disconnected"
    return False, "Disconnected"


def list_workers() -> List[Dict[str, Any]]:
    """List all workers with current status"""
    workers = []
    for worker_id, worker in _workers.items():
        connected, status = check_worker(worker_id)
        workers.append(
            {
                "id": worker_id,
                "url": worker["url"],
                "label": worker["label"],
                "connected": connected,
                "status": status,
                "device": worker.get("device", "unknown"),
                "capabilities": worker.get("capabilities", {}),
                "load": worker.get("load", 0),
            }
        )
    return workers


def is_any_connected() -> bool:
    """Check if any workers are connected"""
    for worker_id in list(_workers.keys()):
        connected, _ = check_worker(worker_id)
        if connected:
            return True
    return False


def connected_worker_ids() -> List[str]:
    """Get all connected worker IDs"""
    connected = []
    for worker_id in list(_workers.keys()):
        connected_worker, _ = check_worker(worker_id)
        if connected_worker:
            connected.append(worker_id)
    return connected


def get_worker(worker_id: str) -> Optional[Dict[str, Any]]:
    """Get worker by ID"""
    return _workers.get(worker_id)


# ---- Video Generation Functions ----
def generate_video_on_worker(
    worker_id: str,
    prompt: str,
    model_id: str,
    context: Dict[str, Any] = None,
    duration_seconds: int = 4,
    fps: int = 24,
    width: int = 576,
    height: int = 320,
    seed: int = 42,
    reference_image: Optional[str] = None,
    timeout: int = 1800,
) -> Dict[str, Any]:
    """
    Generate video asynchronously on a worker.

    The worker immediately returns a job_id.
    This function then polls the worker until the video is ready.

    timeout is the TOTAL maximum time allowed for the job.
    Default: 30 minutes.
    """

    worker = _workers.get(worker_id)

    if not worker:
        raise ValueError(f"Worker {worker_id} not found")

    if not worker.get("connected", False):
        raise RuntimeError(f"Worker {worker_id} is not connected")

    worker["load"] = worker.get("load", 0) + 1

    try:

        payload = {
            "prompt": prompt,
            "model_id": model_id,
            "context": context or {},
            "duration_seconds": duration_seconds,
            "fps": fps,
            "width": width,
            "height": height,
            "seed": seed,
            "reference_image": reference_image,
        }

        # ========================================================
        # STEP 1: SUBMIT JOB
        # ========================================================

        print("")
        print("🖥️ Submitting video job to worker...")

        response = requests.post(
            f"{worker['url']}/video/generate",
            json=payload,
            timeout=30,
        )

        response.raise_for_status()

        job_info = response.json()

        job_id = job_info.get("job_id")

        if not job_id:
            raise RuntimeError(f"Worker did not return job_id: {job_info}")

        print(f"✅ Worker accepted job: {job_id}")

        # ========================================================
        # STEP 2: POLL STATUS
        # ========================================================

        started = time.time()
        last_status = None
        last_progress = None

        poll_interval = 5

        while True:

            elapsed = time.time() - started

            if elapsed > timeout:
                raise TimeoutError(
                    f"Worker video job timed out after " f"{timeout} seconds: {job_id}"
                )

            try:

                status_response = requests.get(
                    f"{worker['url']}/video/status/{job_id}",
                    timeout=15,
                )

                status_response.raise_for_status()

                status = status_response.json()

            except requests.RequestException as poll_error:

                # A temporary tunnel hiccup should NOT immediately
                # destroy the long-running video job.
                print(f"⚠️ Status poll failed: {poll_error}")
                print("   Retrying...")

                time.sleep(poll_interval)
                continue

            current_status = status.get("status", "unknown")

            progress = status.get("progress", 0)

            message = status.get("message", "")

            if current_status != last_status or progress != last_progress:

                print(f"🖥️ Worker job {job_id}: " f"{current_status} " f"{progress}%")

                if message:
                    print(f"   └─ {message}")

                last_status = current_status
                last_progress = progress

            # ====================================================
            # FAILED
            # ====================================================

            if current_status == "failed":

                error = status.get("error", "Unknown worker error")

                raise RuntimeError(f"Worker video job failed: {error}")

            # ====================================================
            # COMPLETED
            # ====================================================

            if current_status == "completed":

                print(f"✅ Worker completed job {job_id}")

                break

            time.sleep(poll_interval)

        # ========================================================
        # STEP 3: DOWNLOAD RESULT
        # ========================================================

        print(f"📥 Retrieving video result: {job_id}")

        result_response = requests.get(
            f"{worker['url']}/video/result/{job_id}",
            timeout=60,
        )

        result_response.raise_for_status()

        result = result_response.json()

        if not result.get("video_data"):
            raise RuntimeError("Worker completed the job but returned " "no video_data")

        print(f"✅ Video received from worker: " f"{len(result['video_data'])} bytes")

        return result

    finally:

        worker["load"] = max(0, worker.get("load", 0) - 1)


def generate_video_round_robin(
    prompt: str,
    model_id: str,
    context: Dict[str, Any] = None,
    duration_seconds: int = 4,
    fps: int = 24,
    width: int = 576,
    height: int = 320,
    seed: int = 42,
    reference_image: Optional[str] = None,
    max_retries: int = 3,
) -> Dict[str, Any]:
    """Generate video using workers in round-robin fashion"""
    connected = connected_worker_ids()
    if not connected:
        raise RuntimeError("No connected workers available")

    # Try each worker
    for attempt in range(max_retries):
        # Pick next worker
        worker_id = connected[next(_rr_counter) % len(connected)]

        try:
            return generate_video_on_worker(
                worker_id,
                prompt,
                model_id,
                context,
                duration_seconds,
                fps,
                width,
                height,
                seed,
                reference_image,
            )
        except Exception as e:
            print(f"Worker {worker_id} failed: {e}")
            # Remove from connected and try next
            if worker_id in connected:
                connected.remove(worker_id)

            if not connected:
                break

            continue

    raise RuntimeError("All workers failed to generate video")


def list_video_models_on_worker(worker_id: str) -> List[Dict[str, Any]]:
    """List video models available on a specific worker"""
    worker = _workers.get(worker_id)
    if not worker:
        return []

    try:
        response = requests.get(f"{worker['url']}/models", timeout=5)
        if response.status_code == 200:
            data = response.json()
            return data.get("video_models", [])
    except Exception:
        pass
    return []


def list_models(worker_id: str) -> List[Dict[str, Any]]:
    """List ALL models (video + clip, per model_manager's registry) known
    to a specific worker, each with its installed status. This is what
    compute.py's list_models()/is_installed() call -- it was previously
    missing entirely, which made every call to compute.py crash with an
    AttributeError as soon as a worker was connected."""
    worker = _workers.get(worker_id)
    if not worker:
        return []

    try:
        response = requests.get(f"{worker['url']}/models", timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get("models", [])
    except Exception as e:
        print(f"[worker_client] list_models failed for {worker_id}: {e}")
        return []


def rank_candidates_on_worker(
    worker_id: str,
    text: str,
    candidates: List[Dict[str, Any]],
    model_id: str,
    top_k: int = 5,
    timeout: int = 60,
) -> List[Dict[str, Any]]:
    """Call a single worker's /rank endpoint."""
    worker = _workers.get(worker_id)
    if not worker:
        raise ValueError(f"Worker {worker_id} not found")

    response = requests.post(
        f"{worker['url']}/rank",
        json={
            "text": text,
            "candidates": candidates,
            "model_id": model_id,
            "top_k": top_k,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return response.json()


def rank_candidates_round_robin(
    text: str,
    candidates: List[Dict[str, Any]],
    model_id: str,
    top_k: int = 5,
    max_retries: int = 3,
) -> List[Dict[str, Any]]:
    """Rank candidates using workers in round-robin fashion (mirrors
    generate_video_round_robin). This was referenced by compute.py but
    never defined, so any ranking call crashed with an AttributeError as
    soon as a worker was connected."""
    connected = connected_worker_ids()
    if not connected:
        raise RuntimeError("No connected workers available")

    for attempt in range(max_retries):
        worker_id = connected[next(_rr_counter) % len(connected)]
        try:
            return rank_candidates_on_worker(
                worker_id, text, candidates, model_id, top_k
            )
        except Exception as e:
            print(f"Worker {worker_id} failed to rank: {e}")
            if worker_id in connected:
                connected.remove(worker_id)
            if not connected:
                break
            continue

    raise RuntimeError("All workers failed to rank candidates")


def switch_video_model(worker_id: str, new_model_id: str) -> None:
    """Delete every OTHER installed video model on this worker before we
    ask it to install `new_model_id`. Colab disks are small and shared with
    the rest of the runtime, so we keep at most one heavy video model
    resident at a time instead of relying only on the reactive
    "clean up when we run out of space" path in smart_download_model.
    CLIP/text models are left alone -- they're small and may still be
    needed for reference-image ranking."""
    worker = _workers.get(worker_id)
    if not worker:
        return

    try:
        models = list_models(worker_id)
    except Exception as e:
        print(
            f"[worker_client] switch_video_model: could not list models on {worker_id}: {e}"
        )
        return

    for m in models:
        if (
            m.get("type") == "video"
            and m.get("installed")
            and m.get("id") != new_model_id
        ):
            try:
                print(
                    f"🧹 Deleting {m['id']} on {worker_id} to make room for {new_model_id}"
                )
                requests.delete(f"{worker['url']}/models/{m['id']}", timeout=60)
            except Exception as e:
                print(
                    f"[worker_client] switch_video_model: failed to delete {m['id']} on {worker_id}: {e}"
                )
