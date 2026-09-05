"""
Worker registry for Universal Documentary Studio.

The registry lives in the Main process and tracks currently
connected GPU workers.
"""

import threading
import time
from typing import Any, Dict, List, Optional


# ============================================================
# REGISTRY
# ============================================================

_workers: Dict[str, Dict[str, Any]] = {}

_registry_lock = threading.Lock()


# ============================================================
# REGISTER
# ============================================================

def register_worker(worker_info: Dict[str, Any]) -> bool:
    """
    Register or refresh a worker in the registry.

    runtime_id is the unique worker identity.
    """

    runtime_id = worker_info.get("runtime_id")

    if not runtime_id:
        print("❌ Cannot register worker without runtime_id")
        return False

    worker = dict(worker_info)

    worker["status"] = "online"
    worker["last_seen"] = time.time()

    with _registry_lock:
        _workers[runtime_id] = worker

    print(f"🟢 Worker registered: {runtime_id}")

    return True


# ============================================================
# UPDATE
# ============================================================

def update_worker(
    runtime_id: str,
    updates: Dict[str, Any],
) -> bool:
    """
    Update information about an existing worker.
    """

    with _registry_lock:

        if runtime_id not in _workers:
            return False

        _workers[runtime_id].update(updates)
        _workers[runtime_id]["last_seen"] = time.time()

    return True


# ============================================================
# HEARTBEAT
# ============================================================

def heartbeat(runtime_id: str) -> bool:
    """
    Mark a worker as alive.
    """

    return update_worker(
        runtime_id,
        {"status": "online"},
    )


# ============================================================
# GET WORKER
# ============================================================

def get_worker(runtime_id: str) -> Optional[Dict[str, Any]]:
    """
    Get one worker by runtime ID.
    """

    with _registry_lock:

        worker = _workers.get(runtime_id)

        if worker is None:
            return None

        return dict(worker)


# ============================================================
# GET ALL WORKERS
# ============================================================

def get_workers() -> List[Dict[str, Any]]:
    """
    Return all registered workers.
    """

    with _registry_lock:
        return [
            dict(worker)
            for worker in _workers.values()
        ]


# ============================================================
# ONLINE WORKERS
# ============================================================

def get_online_workers() -> List[Dict[str, Any]]:
    """
    Return workers currently marked online.
    """

    with _registry_lock:
        return [
            dict(worker)
            for worker in _workers.values()
            if worker.get("status") == "online"
        ]


# ============================================================
# REMOVE WORKER
# ============================================================

def remove_worker(runtime_id: str) -> bool:
    """
    Remove a worker from the registry.
    """

    with _registry_lock:

        if runtime_id not in _workers:
            return False

        del _workers[runtime_id]

    print(f"🔴 Worker removed: {runtime_id}")

    return True


# ============================================================
# CLEAR REGISTRY
# ============================================================

def clear_registry() -> None:
    """
    Remove all workers.

    Useful when Main starts a fresh runtime.
    """

    with _registry_lock:
        _workers.clear()

    print("🧹 Worker registry cleared")