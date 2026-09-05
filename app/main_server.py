"""
Main API Server
Handles communication between Main and GPU workers.
"""

from fastapi import FastAPI, HTTPException
from typing import Dict, Any

from app import worker_registry


def build_main_app():
    app = FastAPI(title="Universal Documentary Studio Main")

    @app.get("/health")
    async def health():
        return {
            "status": "ok",
            "service": "main",
        }

    @app.post("/workers/register")
    async def register_worker(worker_info: Dict[str, Any]):
        runtime_id = worker_info.get("runtime_id")

        if not runtime_id:
            raise HTTPException(
                status_code=400,
                detail="Worker registration requires runtime_id",
            )

        registered = worker_registry.register_worker(worker_info)

        if not registered:
            raise HTTPException(
                status_code=400,
                detail="Worker registration failed",
            )

        return {
            "status": "registered",
            "runtime_id": runtime_id,
        }

    @app.post("/workers/{runtime_id}/heartbeat")
    async def worker_heartbeat(runtime_id: str):
        if not worker_registry.heartbeat(runtime_id):
            raise HTTPException(
                status_code=404,
                detail="Worker not registered",
            )

        return {
            "status": "ok",
            "runtime_id": runtime_id,
        }

    @app.get("/workers")
    async def list_registered_workers():
        return {
            "workers": worker_registry.get_workers()
        }

    return app