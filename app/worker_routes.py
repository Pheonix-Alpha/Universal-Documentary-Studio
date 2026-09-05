"""
Worker registration routes for Universal Documentary Studio.
"""

from fastapi import APIRouter, HTTPException

from app import worker_registry

router = APIRouter(prefix="/workers", tags=["workers"])


@router.post("/register")
def register_worker(worker_info: dict):
    """
    Register a worker with Main.
    """

    success = worker_registry.register_worker(worker_info)

    if not success:
        raise HTTPException(
            status_code=400,
            detail="Invalid worker registration data",
        )

    return {
        "success": True,
        "runtime_id": worker_info.get("runtime_id"),
        "message": "Worker registered successfully",
    }


@router.get("")
def list_workers():
    """
    Return currently registered workers.
    """

    return {
        "workers": worker_registry.get_workers()
    }