"""
Background monitor for connected GPU workers.

Main periodically checks each registered worker's /health endpoint.
"""

import threading
import time
import requests

from app import worker_registry

CHECK_INTERVAL = 10
WORKER_TIMEOUT = 20

_monitor_thread = None
_monitor_running = False


def check_worker_health(worker):
    """Check whether a worker is reachable."""

    runtime_id = worker.get("runtime_id")
    url = worker.get("endpoint")

    if not runtime_id or not url:
        return False

    try:
        response = requests.get(
            f"{url.rstrip('/')}/health",
            timeout=5,
        )

        if response.status_code == 200:
            worker_registry.update_worker(
                runtime_id,
                {
                    "status": "online",
                },
            )

            return True

    except Exception:
        pass

    return False


def monitor_workers():
    """Continuously monitor registered workers."""

    global _monitor_running

    print("💓 Worker health monitor started")

    while _monitor_running:

        workers = worker_registry.get_workers()

        current_time = time.time()

        for worker in workers:
            runtime_id = worker.get("runtime_id")

            if not runtime_id:
                continue

            healthy = check_worker_health(worker)

            if not healthy:

                last_seen = worker.get(
                    "last_seen",
                    current_time,
                )

                if current_time - last_seen >= WORKER_TIMEOUT:

                    worker_registry.update_worker(
                        runtime_id,
                        {
                            "status": "offline",
                        },
                        update_last_seen=False,
                    )

                    print(f"🔴 Worker offline: {runtime_id}")

        time.sleep(CHECK_INTERVAL)


def start_worker_monitor():
    """Start the worker monitor in a background thread."""

    global _monitor_thread
    global _monitor_running

    if _monitor_running:
        return

    _monitor_running = True

    _monitor_thread = threading.Thread(
        target=monitor_workers,
        daemon=True,
    )

    _monitor_thread.start()
