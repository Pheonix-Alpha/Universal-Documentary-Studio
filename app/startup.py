"""Startup routine (spec section 59/60).

Run once at process start (locally or on a Colab worker) to detect
hardware and report what the system *will* do, without blindly
installing or loading any heavyweight model. Actual model loading is
always deferred to the first job that needs it (see ModelRegistry +
Scheduler), never performed eagerly here.
"""
from __future__ import annotations

import json

from app.config import load_app_config
from core.logging import get_logger
from core.resource_manager import ResourceManager

logger = get_logger(__name__)


def run_startup(lightweight: bool = True) -> dict:
    config = load_app_config()
    rm = ResourceManager(vram_safety_margin_gb=config["resource"]["vram_safety_margin_gb"])
    report = rm.detect(refresh=True)
    profile = rm.classify()

    summary = {
        "runtime_report": report.to_dict(),
        "worker_profile": profile.value,
        "mock_mode": config["mock_mode"],
        "local_gpu_enabled": config["local_gpu_enabled"],
        "lightweight_mode": lightweight,
    }

    logger.info("Startup summary: %s", json.dumps(summary, indent=2))

    if not lightweight and not config["mock_mode"]:
        logger.info(
            "Non-lightweight startup requested: real model downloads would be "
            "triggered lazily on first use via ModelRegistry, not eagerly here."
        )

    return summary


if __name__ == "__main__":
    run_startup()
