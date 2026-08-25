"""Application configuration loading (config/config.yaml)."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


def load_app_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        return _default_config()
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    merged = _default_config()
    _deep_merge(merged, data)
    return merged


def _deep_merge(base: dict, override: dict) -> None:
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def _default_config() -> dict[str, Any]:
    return {
        "mock_mode": True,
        "local_gpu_enabled": False,
        "projects_root": "projects",

        "resource": {
            "vram_safety_margin_gb": 1.5,
        },

        "tts": {
            "provider": "mock",
            "model_path": "",
            "use_cuda": False,
        },

        "video": {
            "long_form": {"width": 1920, "height": 1080, "fps": 24},
            "short": {"width": 1080, "height": 1920, "fps": 30},
        },

        "pipeline": {
            "target_duration_minutes": 10.0,
            "short_count": 4,
            "research_depth": "standard",
        },

        "qa": {
            "ready_threshold": 90,
            "review_threshold": 80,
            "regenerate_threshold": 70,
        },
    }