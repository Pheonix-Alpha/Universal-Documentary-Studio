"""
Single entry point the rest of the app (pipeline.py, gradio_app.py) calls
for model management + CLIP ranking. Transparently routes to whichever
worker(s) are currently connected (separate Colab GPUs -- see worker.py /
app/worker_client.py), round-robining ranking calls across all of them, and
falls back to the local model_manager/clip_ranker when none are connected.
This is the "check in the background if a worker is present" logic --
callers never need to know which backend actually did the work.
"""
from app import clip_ranker
from app import model_manager as local_mm
from app import worker_client

DEFAULT_CLIP_MODEL_ID = clip_ranker.DEFAULT_CLIP_MODEL_ID


def backend_name() -> str:
    """'worker' if at least one worker is connected and reachable right
    now, else 'local'."""
    return "worker" if worker_client.is_any_connected() else "local"


def list_models():
    """Model registry for populating dropdowns etc. Every worker runs the
    same codebase, so any connected worker's registry is representative --
    this just needs *a* list of available model ids, not installed-ness
    (see is_installed() for that, which checks every connected worker)."""
    if backend_name() == "worker":
        wid = worker_client.connected_worker_ids()[0]
        models = worker_client.list_models(wid)
        if models:
            return models
    return local_mm.list_models()


def is_installed(model_id: str) -> bool:
    if backend_name() == "worker":
        ids = worker_client.connected_worker_ids()
        if not ids:
            return False
        # Ranking round-robins across every connected worker, so the model
        # needs to be installed on ALL of them or a run could fail partway.
        for wid in ids:
            models = worker_client.list_models(wid)
            if not any(m["id"] == model_id and m["installed"] for m in models):
                return False
        return True
    for m in local_mm.list_models():
        if m["id"] == model_id:
            return m["installed"]
    return False


def rank_candidates(text: str, candidates: list, model_id: str = DEFAULT_CLIP_MODEL_ID, top_k: int = 5):
    if backend_name() == "worker":
        return worker_client.rank_candidates_round_robin(text, candidates, model_id=model_id, top_k=top_k)
    return clip_ranker.rank_candidates(text, candidates, model_id=model_id, top_k=top_k)