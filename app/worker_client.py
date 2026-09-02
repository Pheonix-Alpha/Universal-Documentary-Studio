"""
FastAPI app exposing model management + CLIP ranking, meant to run on a
SEPARATE ("worker") Colab notebook so its GPU can be offloaded to from the
main UI notebook. Started by worker.py -- not meant to be imported by the
main app's own process (the main app talks to it over HTTP via
app/worker_client.py).
"""
import base64
import io
import threading

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app import clip_ranker
from app import model_manager as mm

_download_jobs = {}  # model_id -> {"pct": float, "msg": str, "done": bool, "error": str|None}


class RankRequest(BaseModel):
    text: str
    candidates: list
    model_id: str = "clip-vit-b-32"
    top_k: int = 5


def build_fastapi_app() -> FastAPI:
    app = FastAPI(title="Storyboard Worker")

    @app.get("/health")
    def health():
        try:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:  # noqa: BLE001
            device = "unknown"
        return {"status": "ok", "device": device}

    @app.get("/models")
    def list_models():
        return mm.list_models()

    @app.post("/models/{model_id}/download")
    def download(model_id: str):
        if model_id not in mm.MODEL_REGISTRY:
            raise HTTPException(404, f"Unknown model {model_id}")
        if mm.is_installed(model_id):
            return {"already_installed": True}

        _download_jobs[model_id] = {"pct": 0, "msg": "starting...", "done": False, "error": None}

        def _worker():
            try:
                for pct, msg in mm.download_model_stream(model_id):
                    if pct is None:
                        _download_jobs[model_id] = {"pct": 0, "msg": msg, "done": True, "error": msg}
                        return
                    _download_jobs[model_id] = {"pct": pct, "msg": msg, "done": False, "error": None}
                _download_jobs[model_id]["done"] = True
                _download_jobs[model_id]["pct"] = 100
            except Exception as e:  # noqa: BLE001
                _download_jobs[model_id] = {"pct": 0, "msg": str(e), "done": True, "error": str(e)}

        threading.Thread(target=_worker, daemon=True).start()
        return {"started": True}

    @app.get("/models/{model_id}/progress")
    def progress(model_id: str):
        if model_id in _download_jobs:
            return _download_jobs[model_id]
        installed = mm.is_installed(model_id)
        return {"pct": 100 if installed else 0, "msg": "installed" if installed else "not started",
                "done": True, "error": None}

    @app.delete("/models/{model_id}")
    def delete(model_id: str):
        mm.delete_model(model_id)
        _download_jobs.pop(model_id, None)
        return {"deleted": True}

    @app.post("/rank")
    def rank(req: RankRequest):
        if not mm.is_installed(req.model_id):
            raise HTTPException(400, f"Model {req.model_id} not installed on this worker")
        ranked = clip_ranker.rank_candidates(req.text, req.candidates, model_id=req.model_id, top_k=req.top_k)
        out = []
        for r in ranked:
            buf = io.BytesIO()
            thumb = r["image"].copy()
            thumb.thumbnail((480, 480))
            thumb.convert("RGB").save(buf, format="JPEG", quality=80)
            out.append(
                {
                    "url": r.get("url"),
                    "thumbnail_url": r.get("thumbnail_url"),
                    "source": r.get("source"),
                    "title": r.get("title"),
                    "media_type": r.get("media_type", "image"),
                    "score": r["score"],
                    "thumbnail_base64": base64.b64encode(buf.getvalue()).decode("ascii"),
                }
            )
        return {"results": out}

    return app