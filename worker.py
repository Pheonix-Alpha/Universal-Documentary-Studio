"""
Client for an optional remote worker (a separate Colab notebook running
worker.py) that offloads model downloads and CLIP ranking to its own GPU.

The main app calls set_worker_url() when the person clicks Connect in the
Worker tab, then is_connected()/list_models()/download_model_stream()/
delete_model()/rank_candidates() all transparently no-op or raise if no
worker is set, so the rest of the app doesn't need to special-case it.
"""
import base64
import io
import time

import requests
from PIL import Image

_state = {"url": None}


def set_worker_url(url: str):
    _state["url"] = url.strip().rstrip("/") if url and url.strip() else None


def get_worker_url():
    return _state["url"]


def is_connected(timeout: float = 4):
    url = _state["url"]
    if not url:
        return False, "No worker URL set."
    try:
        r = requests.get(f"{url}/health", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return True, f"Connected ({data.get('device', '?')})"
    except Exception as e:  # noqa: BLE001
        return False, f"Not reachable: {e}"


def list_models(timeout: float = 8):
    url = _state["url"]
    if not url:
        return []
    try:
        r = requests.get(f"{url}/models", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        print(f"[worker_client] Failed to list worker models: {e}")
        return []


def download_model_stream(model_id: str, poll_interval: float = 1.0, timeout: float = 10):
    """Starts a download on the worker and polls its progress, yielding
    (pct, msg) with the same shape as model_manager.download_model_stream()."""
    url = _state["url"]
    if not url:
        yield None, "No worker connected."
        return
    try:
        r = requests.post(f"{url}/models/{model_id}/download", timeout=timeout)
        r.raise_for_status()
        if r.json().get("already_installed"):
            yield 100, f"{model_id} already installed on worker."
            return
    except Exception as e:  # noqa: BLE001
        yield None, f"Could not start worker download: {e}"
        return

    while True:
        try:
            r = requests.get(f"{url}/models/{model_id}/progress", timeout=timeout)
            r.raise_for_status()
            data = r.json()
        except Exception as e:  # noqa: BLE001
            yield None, f"Lost connection to worker: {e}"
            return
        if data.get("error"):
            yield None, data["error"]
            return
        yield data.get("pct", 0), data.get("msg", "")
        if data.get("done"):
            return
        time.sleep(poll_interval)


def delete_model(model_id: str, timeout: float = 15):
    url = _state["url"]
    if not url:
        return
    try:
        requests.delete(f"{url}/models/{model_id}", timeout=timeout)
    except Exception as e:  # noqa: BLE001
        print(f"[worker_client] Failed to delete worker model {model_id}: {e}")


def rank_candidates(text: str, candidates: list, model_id: str = "clip-vit-b-32", top_k: int = 5, timeout: float = 60):
    """Same contract as clip_ranker.rank_candidates(), executed remotely."""
    url = _state["url"]
    if not url:
        raise RuntimeError("No worker connected.")
    payload = {"text": text, "candidates": candidates, "model_id": model_id, "top_k": top_k}
    r = requests.post(f"{url}/rank", json=payload, timeout=timeout)
    r.raise_for_status()
    out = []
    for item in r.json()["results"]:
        img = Image.open(io.BytesIO(base64.b64decode(item["thumbnail_base64"]))).convert("RGB")
        c = dict(item)
        c["image"] = img
        out.append(c)
    return out