"""
Runtime Manager
---------------
Detects the current runtime/platform and provides startup diagnostics.

Supported environments:
- Google Colab
- Kaggle
- Local / other Linux environments
"""

import os
import sys
import uuid
import shutil
import platform
from typing import Dict, Any, List


# ---------------------------------------------------------------------------
# Runtime identity
# ---------------------------------------------------------------------------

_RUNTIME_ID = None


def get_runtime_id() -> str:
    """
    Return a stable ID for the current Python runtime/session.

    The ID is generated once per process and reused for the lifetime
    of this runtime.
    """
    global _RUNTIME_ID

    if _RUNTIME_ID is None:
        _RUNTIME_ID = f"runtime_{uuid.uuid4().hex[:12]}"

    return _RUNTIME_ID


# ---------------------------------------------------------------------------
# Platform detection
# ---------------------------------------------------------------------------

def detect_platform() -> str:
    """
    Detect the execution platform.

    Returns:
        'colab'
        'kaggle'
        'local'
    """

    # Google Colab
    try:
        import google.colab  # noqa: F401
        return "colab"
    except ImportError:
        pass

    # Kaggle
    if os.path.exists("/kaggle") or os.environ.get("KAGGLE_KERNEL_RUN_TYPE"):
        return "kaggle"

    return "local"


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------

def get_gpu_info() -> List[Dict[str, Any]]:
    """
    Return information about every CUDA GPU visible to PyTorch.
    """

    try:
        import torch
    except ImportError:
        return []

    if not torch.cuda.is_available():
        return []

    gpus = []

    for index in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(index)

            total_vram = props.total_memory / (1024 ** 3)

            gpus.append({
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "vram_gb": round(total_vram, 2),
                "cuda_capability": f"{props.major}.{props.minor}",
            })

        except Exception as exc:
            gpus.append({
                "index": index,
                "name": "Unknown",
                "vram_gb": 0,
                "cuda_capability": "unknown",
                "error": str(exc),
            })

    return gpus


def get_gpu_count() -> int:
    """Return the number of visible CUDA GPUs."""
    return len(get_gpu_info())


# ---------------------------------------------------------------------------
# Storage detection
# ---------------------------------------------------------------------------

def get_storage_info() -> Dict[str, float]:
    """
    Return disk information for the current environment.
    """

    try:
        total, used, free = shutil.disk_usage("/")

        return {
            "total_gb": round(total / (1024 ** 3), 2),
            "used_gb": round(used / (1024 ** 3), 2),
            "free_gb": round(free / (1024 ** 3), 2),
        }

    except Exception:
        return {
            "total_gb": 0,
            "used_gb": 0,
            "free_gb": 0,
        }


# ---------------------------------------------------------------------------
# Runtime summary
# ---------------------------------------------------------------------------

def get_runtime_info() -> Dict[str, Any]:
    """
    Return complete runtime information.
    """

    gpus = get_gpu_info()
    storage = get_storage_info()

    return {
        "runtime_id": get_runtime_id(),
        "platform": detect_platform(),
        "python_version": platform.python_version(),
        "os": platform.system(),
        "machine": platform.machine(),
        "gpu_count": len(gpus),
        "gpus": gpus,
        "storage": storage,
    }


# ---------------------------------------------------------------------------
# Pretty startup report
# ---------------------------------------------------------------------------

def print_runtime_report(role: str) -> Dict[str, Any]:
    """
    Print a human-readable startup report.

    role:
        'main'
        'worker'
    """

    info = get_runtime_info()

    print()
    print("=" * 70)
    print("🔎 UNIVERSAL DOCUMENTARY STUDIO — STARTUP CHECK")
    print("=" * 70)

    print(f"Role:       {role}")
    print(f"Platform:   {info['platform']}")
    print(f"Runtime ID: {info['runtime_id']}")
    print(f"Python:     {info['python_version']}")
    print(f"OS:         {info['os']} / {info['machine']}")

    print()
    print(f"🎮 GPUs detected: {info['gpu_count']}")

    if info["gpus"]:
        for gpu in info["gpus"]:
            print(
                f"   GPU {gpu['index']}: "
                f"{gpu['name']} "
                f"({gpu['vram_gb']:.1f} GB)"
            )
    else:
        print("   ⚠️ No CUDA GPU detected")

    storage = info["storage"]

    print()
    print("💾 Storage:")
    print(f"   Total: {storage['total_gb']:.1f} GB")
    print(f"   Used:  {storage['used_gb']:.1f} GB")
    print(f"   Free:  {storage['free_gb']:.1f} GB")

    print("=" * 70)
    print()

    return info