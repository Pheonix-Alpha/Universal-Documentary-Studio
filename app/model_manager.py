"""
Tracks which models are downloaded locally (under ./models/<model_id>/),
and provides:
  - list_models()            -> for the Gradio "Models" tab
  - is_installed(model_id)
  - download_model_stream()  -> generator yielding (percent, message)
  - delete_model(model_id)   -> frees disk space (important on Colab)
"""
import queue
import shutil
import threading

from app.config import MODEL_DIR

MODEL_REGISTRY = {
    "clip-vit-b-32": {
        "name": "CLIP ViT-B/32",
        "repo_id": "openai/clip-vit-base-patch32",
        "size_mb": 605,
        "description": "Fastest CLIP model. Good default for a free-tier T4 in Colab.",
    },
    "clip-vit-b-16": {
        "name": "CLIP ViT-B/16",
        "repo_id": "openai/clip-vit-base-patch16",
        "size_mb": 605,
        "description": "A bit more accurate than B/32, still fast enough for a T4.",
    },
    "clip-vit-l-14": {
        "name": "CLIP ViT-L/14",
        "repo_id": "openai/clip-vit-large-patch14",
        "size_mb": 1700,
        "description": "Best zero-shot accuracy, heaviest. Try this once B/16 works well.",
    },
}


def _model_dir(model_id):
    return MODEL_DIR / model_id


def is_installed(model_id) -> bool:
    d = _model_dir(model_id)
    return d.exists() and any(d.iterdir())


def get_model_path(model_id) -> str:
    return str(_model_dir(model_id))


def list_models():
    """Returns a list of dicts describing every model in the registry,
    including whether it's currently downloaded."""
    out = []
    for mid, meta in MODEL_REGISTRY.items():
        out.append(
            {
                "id": mid,
                "name": meta["name"],
                "size_mb": meta["size_mb"],
                "description": meta["description"],
                "installed": is_installed(mid),
            }
        )
    return out


def delete_model(model_id: str):
    d = _model_dir(model_id)
    if d.exists():
        shutil.rmtree(d)


def _download_blocking(model_id: str, progress_cb):
    """Blocking download of a full HF repo snapshot, reporting % via progress_cb(pct, msg)."""
    from huggingface_hub import snapshot_download
    from tqdm import tqdm as tqdm_base

    meta = MODEL_REGISTRY[model_id]
    dest = _model_dir(model_id)
    dest.mkdir(parents=True, exist_ok=True)

    class ProgressTqdm(tqdm_base):
        def update(self, n=1):
            super().update(n)
            try:
                if self.total:
                    pct = min(100.0, (self.n / self.total) * 100.0)
                    progress_cb(pct, f"Downloading {model_id}: {self.n}/{self.total} bytes")
            except Exception:
                pass

    snapshot_download(repo_id=meta["repo_id"], local_dir=str(dest), tqdm_class=ProgressTqdm)


def download_model_stream(model_id: str):
    """Generator you can iterate in a Gradio click handler:
    for pct, msg in download_model_stream(model_id): ...
    pct is None on error (msg carries the error text)."""
    if model_id not in MODEL_REGISTRY:
        yield None, f"Unknown model: {model_id}"
        return

    q: "queue.Queue" = queue.Queue()

    def worker():
        try:
            _download_blocking(model_id, progress_cb=lambda pct, msg: q.put((pct, msg)))
            q.put((100, f"{model_id} download complete."))
        except Exception as e:  # noqa: BLE001
            q.put((None, f"Error downloading {model_id}: {e}"))
        finally:
            q.put(None)

    t = threading.Thread(target=worker, daemon=True)
    t.start()

    while True:
        item = q.get()
        if item is None:
            break
        yield item
