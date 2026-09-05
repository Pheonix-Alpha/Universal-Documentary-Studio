"""
Smart Model Selector - "the brain" decides which video model to use.

This is the piece that was missing: previously the UI's model dropdown just
passed whatever the user manually picked straight through to the worker for
every single scene, with no automatic reasoning about the job's
requirements or the worker's actual resources.

Design, matching the requested flow:
    script -> bible -> auto-pick best model for the job -> worker installs
    it (deleting any other resident video model first) -> generate.

The main Colab ("the director"/"the brain") NEVER downloads a heavy video
model itself -- it only inspects a connected worker's reported VRAM/storage
(via GET /health) and its installed models (via GET /models), decides which
model_id best fits, and tells the worker to use it. The only things that
ever run locally on the main Colab are small supervisory jobs: the local
"brain" LLM (model_manager.generate_text) for the bible/script, and
optionally a local CLIP model for reference-image ranking when no worker is
connected (see compute.py).
"""
from typing import Any, List, Optional, Dict

from app import model_manager, worker_client

# Best -> worst quality. Must match the video-type entries in
# model_manager.MODEL_REGISTRY.
VIDEO_MODEL_PRIORITY = [
    "stabilityai/stable-video-diffusion-img2vid",
    "cerspense/zeroscope_v2_576w",
]


def _worker_health(worker_id: Optional[str]) -> Dict[str, Any]:
    """Ask the worker what it has available right now. Best-effort: if the
    worker doesn't answer (or none is connected), the caller treats
    resources as unknown and falls back to the safest/lightest model."""
    if not worker_id:
        return {}
    worker = worker_client.get_worker(worker_id)
    if not worker:
        return {}
    try:
        import requests
        r = requests.get(f"{worker['url']}/health", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        print(f"[model_selector] Could not read worker health for {worker_id}: {e}")
        return {}


def select_video_model(
    production_units: List[Any],
    worker_id: Optional[str] = None,
    quality_preference: str = "auto",  # "auto" | "quality" | "speed"
) -> str:
    """Pick the best video model_id for this job.

    Requirement signals taken from the job itself:
      - total number of scenes / total runtime (a big job biases toward the
        lighter, faster model even in "auto" mode, since Colab sessions have
        a time budget)
      - whether characters are involved doesn't currently change the pick,
        but is threaded through so future logic (e.g. preferring img2vid
        with a reference image when we have named characters) has it.

    Resource signal taken from the worker (if connected):
      - free VRAM must cover the candidate model's requirement
      - a model whose *total* footprint (not just currently-free space)
        would never fit the worker's storage budget is skipped entirely,
        since no amount of cleanup would make it fit

    If nothing is connected, or nothing fits, we fall back to the lightest
    model -- the worker will still smart-download/clean up for us.
    """
    total_scenes = len(production_units)
    total_seconds = sum(getattr(u, "duration_seconds", 4) for u in production_units)

    if quality_preference == "speed":
        candidates = list(reversed(VIDEO_MODEL_PRIORITY))
    elif quality_preference == "quality":
        candidates = list(VIDEO_MODEL_PRIORITY)
    else:  # auto
        candidates = list(VIDEO_MODEL_PRIORITY)
        # A big job leans toward the faster/lighter model so a single
        # worker doesn't spend the whole session on one documentary.
        if total_scenes > 12 or total_seconds > 180:
            heaviest = candidates[0]
            candidates = [m for m in candidates if m != heaviest] + [heaviest]

    health = _worker_health(worker_id)
    vram = health.get("vram") or {}
    storage = health.get("storage") or {}
    have_resource_info = bool(vram.get("available"))
    free_vram = vram.get("free_gb") if have_resource_info else None
    # free_storage isn't actually used to reject candidates -- the worker's
    # smart_download_model()/​_smart_cleanup() will free space by deleting
    # old models automatically, and switch_video_model() (called by the
    # pipeline right before this) already proactively removes the previous
    # video model. We only reject a model outright if it could never fit at
    # all, regardless of cleanup.
    max_storage = model_manager.MAX_STORAGE_GB

    if not have_resource_info:
        # No worker connected, or it didn't report usable VRAM info -- we
        # have no basis for picking a heavy model, so play it safe with the
        # lightest one rather than assuming the biggest model "fits" just
        # because we couldn't prove otherwise.
        return candidates[-1]

    for model_id in candidates:
        vram_needed = model_manager.get_vram_required(model_id)
        size_needed = model_manager.get_model_size(model_id)

        if free_vram is not None and free_vram < vram_needed:
            continue
        if size_needed > max_storage:
            continue
        return model_id

    # Worker is connected but nothing fit its reported VRAM -> safest
    # default: the cheapest model.
    return candidates[-1]


def describe_selection(model_id: str, worker_id: Optional[str]) -> Dict[str, Any]:
    """Small structured summary for the UI's "Model Status" panel."""
    info = model_manager.MODEL_REGISTRY.get(model_id, {})
    return {
        "selected_model": model_id,
        "selected_model_name": info.get("name", model_id),
        "size_gb": info.get("size_gb"),
        "vram_gb_required": info.get("vram_gb"),
        "worker": worker_id or "none (local fallback)",
    }
