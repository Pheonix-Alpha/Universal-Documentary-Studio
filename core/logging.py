"""Centralized logging configuration for UDS.

Every module should call `get_logger(__name__)` rather than configuring
logging itself, so behavior stays consistent across local, Colab, and
test environments.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

_CONFIGURED = False


def _configure_root(log_dir: str | None = None) -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return

    level_name = os.environ.get("UDS_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)

    root = logging.getLogger("uds")
    root.setLevel(level)
    root.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    root.addHandler(stream_handler)

    if log_dir:
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(Path(log_dir) / "uds.log")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)

    _CONFIGURED = True


def get_logger(name: str, log_dir: str | None = None) -> logging.Logger:
    """Return a namespaced logger under the shared `uds` root logger."""
    _configure_root(log_dir=log_dir or os.environ.get("UDS_LOG_DIR"))
    if not name.startswith("uds"):
        name = f"uds.{name}"
    return logging.getLogger(name)
