"""
Single entry point the rest of the app (pipeline.py, gradio_app.py) calls
for model management + CLIP ranking. Transparently routes to a connected
worker (separate Colab GPU -- see worker.py / app/worker_client.py) when
one is connected, and falls back to the local model_manager/clip_ranker
otherwise. This is the "check in the background if a worker is present"
logic -- callers never need to know which backend actually did the work.
"""
from app import clip_ranker
from app import model_manager as local_mm
from app import worker_client


def backend_name() -> str:
    """'worker' if a worker is connected and reachable right now, else 'local'."""
    connected, _ = worker_client.is_connected()
    return "worker" if connected else "local"


def list_models():
    if backend_name() == "worker":
        models = worker_client.list_models()
        if models:  # only trust the worker's answer if it actually returned one
            return models
    return local_mm.list_models()


def is_installed(model_id: str) -> bool:
    for m in list_models():
        if m["id"] == model_id:
            return m["installed"]
    return False


def download_model_stream(model_id: str):
    if backend_name() == "worker":
        yield from worker_client.download_model_stream(model_id)
    else:
        yield from local_mm.download_model_stream(model_id)


def delete_model(model_id: str):
    if backend_name() == "worker":
        worker_client.delete_model(model_id)
    else:
        local_mm.delete_model(model_id)


def rank_candidates(text: str, candidates: list, model_id: str = "clip-vit-b-32", top_k: int = 5):
    if backend_name() == "worker":
        return worker_client.rank_candidates(text, candidates, model_id=model_id, top_k=top_k)
    return clip_ranker.rank_candidates(text, candidates, model_id=model_id, top_k=top_k)