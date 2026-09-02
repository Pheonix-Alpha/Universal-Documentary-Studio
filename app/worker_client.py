"""
Client for zero or more remote workers (each a separate Colab notebook
running worker.py) that offload model downloads and CLIP ranking to their
own GPU.

Multiple workers can be connected at once: add_worker() registers one and
returns an id, remove_worker() drops it, and list_workers() reports live
connection status for every registered worker (used by the "Workers" tab
in the UI). Model management (list_models/download_model_stream/
delete_model) is per worker, keyed by that id.

rank_candidates_round_robin() is what the rest of the app actually calls
during a run (see app/compute.py) -- it spreads ranking calls across every
currently-reachable worker in rotation, retrying on the next worker if one
fails, so a story with many scenes gets processed across several GPUs at
once instead of piling onto a single worker.
"""
import base64
import io
import itertools
import time
import uuid

import requests
from PIL import Image

_workers = {}  # worker_id -> {"url": str, "label": str}
_rr_counter = itertools.count()


def add_worker(url: str, label: str = "") -> str:
    """Registers a worker and returns its id. Re-adding the same URL just
    returns the existing id rather than creating a duplicate entry."""
    url = (url or "").strip().rstrip("/")
    if not url:
        raise ValueError("Worker URL can't be empty.")
    for wid, w in _workers.items():
        if w["url"] == url:
            return wid
    wid = uuid.uuid4().hex[:8]
    _workers[wid] = {"url": url, "label": (label or "").strip() or url}
    return wid


def remove_worker(worker_id: str):
    _workers.pop(worker_id, None)


def get_worker(worker_id: str):
    return _workers.get(worker_id)


def check_worker(worker_id: str, timeout: float = 4):
    w = _workers.get(worker_id)
    if not w:
        return False, "Unknown worker."
    try:
        r = requests.get(f"{w['url']}/health", timeout=timeout)
        r.raise_for_status()
        data = r.json()
        return True, f"Connected ({data.get('device', '?')})"
    except Exception as e:  # noqa: BLE001
        return False, f"Not reachable: {e}"


def list_workers():
    """Returns [{id, url, label, connected, status}, ...] for the UI --
    pings every registered worker so the list is always current."""
    out = []
    for wid, w in _workers.items():
        connected, status = check_worker(wid)
        out.append({"id": wid, "url": w["url"], "label": w["label"], "connected": connected, "status": status})
    return out


def connected_worker_ids():
    return [w["id"] for w in list_workers() if w["connected"]]


def is_any_connected() -> bool:
    return len(connected_worker_ids()) > 0


def list_models(worker_id: str, timeout: float = 8):
    w = _workers.get(worker_id)
    if not w:
        return []
    try:
        r = requests.get(f"{w['url']}/models", timeout=timeout)
        r.raise_for_status()
        return r.json()
    except Exception as e:  # noqa: BLE001
        print(f"[worker_client] Failed to list models on worker {worker_id}: {e}")
        return []


def download_model_stream(worker_id: str, model_id: str, poll_interval: float = 1.0, timeout: float = 10):
    """Starts a download on the given worker and polls its progress,
    yielding (pct, msg) with the same shape as
    model_manager.download_model_stream()."""
    w = _workers.get(worker_id)
    if not w:
        yield None, "Unknown worker."
        return
    url = w["url"]
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


def delete_model(worker_id: str, model_id: str, timeout: float = 15):
    w = _workers.get(worker_id)
    if not w:
        return
    try:
        requests.delete(f"{w['url']}/models/{model_id}", timeout=timeout)
    except Exception as e:  # noqa: BLE001
        print(f"[worker_client] Failed to delete model on worker {worker_id}: {e}")


def rank_candidates(worker_id: str, text: str, candidates: list, model_id: str = "clip-vit-b-32", top_k: int = 5, timeout: float = 60):
    """Same contract as clip_ranker.rank_candidates(), executed on one specific worker."""
    w = _workers.get(worker_id)
    if not w:
        raise RuntimeError(f"Unknown worker: {worker_id}")
    payload = {"text": text, "candidates": candidates, "model_id": model_id, "top_k": top_k}
    r = requests.post(f"{w['url']}/rank", json=payload, timeout=timeout)
    r.raise_for_status()
    out = []
    for item in r.json()["results"]:
        img = Image.open(io.BytesIO(base64.b64decode(item["thumbnail_base64"]))).convert("RGB")
        c = dict(item)
        c["image"] = img
        out.append(c)
    return out


def rank_candidates_round_robin(text: str, candidates: list, model_id: str = "clip-vit-b-32", top_k: int = 5):
    """Picks the next connected worker in rotation and ranks on it, falling
    through to the other connected workers in turn if one is offline or
    errors, so a single flaky worker doesn't fail the whole run."""
    ids = connected_worker_ids()
    if not ids:
        raise RuntimeError("No worker connected.")
    start = next(_rr_counter) % len(ids)
    ordered = ids[start:] + ids[:start]
    last_err = None
    for wid in ordered:
        try:
            return rank_candidates(wid, text, candidates, model_id=model_id, top_k=top_k)
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"[worker_client] Worker {wid} failed, trying next: {e}")
    raise RuntimeError(f"All connected workers failed: {last_err}")