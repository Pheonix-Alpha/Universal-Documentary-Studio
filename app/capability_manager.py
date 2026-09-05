"""
Runtime capability detection for Universal Documentary Studio.

Converts detected runtime hardware into capabilities that can
be used by the worker registry and scheduler.
"""

from typing import Any, Dict

from app import runtime_manager


# ============================================================
# CAPABILITY DETECTION
# ============================================================

def get_capabilities() -> Dict[str, Any]:
    """
    Detect what the current runtime is capable of.
    """

    runtime = runtime_manager.get_runtime_info()

    gpus = runtime.get("gpus", [])
    gpu_count = runtime.get("gpu_count", 0)

    cuda_available = gpu_count > 0

    # Current conservative capability rules.
    #
    # We are intentionally NOT deciding exact model support here.
    # Model-specific requirements will be handled later.
    video_generation = cuda_available

    capabilities = {
        "cuda": cuda_available,
        "gpu_count": gpu_count,
        "multi_gpu": gpu_count > 1,
        "video_generation": video_generation,
    }

    return capabilities


# ============================================================
# RUNTIME CAPABILITY REPORT
# ============================================================

def get_runtime_capability_info() -> Dict[str, Any]:
    """
    Return runtime information together with capabilities.
    """

    runtime = runtime_manager.get_runtime_info()
    capabilities = get_capabilities()

    return {
        "runtime": runtime,
        "capabilities": capabilities,
    }


# ============================================================
# PRINT REPORT
# ============================================================

def print_capability_report() -> None:
    """
    Print a human-readable capability report.
    """

    info = get_runtime_capability_info()
    capabilities = info["capabilities"]

    print()
    print("=" * 70)
    print("⚙️  RUNTIME CAPABILITY CHECK")
    print("=" * 70)

    print(f"CUDA:             {'✅' if capabilities['cuda'] else '❌'}")
    print(f"GPU Count:        {capabilities['gpu_count']}")
    print(f"Multi-GPU:        {'✅' if capabilities['multi_gpu'] else '❌'}")
    print(
        f"Video Generation: "
        f"{'✅' if capabilities['video_generation'] else '❌'}"
    )

    print("=" * 70)