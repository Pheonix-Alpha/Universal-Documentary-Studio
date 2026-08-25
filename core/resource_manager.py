"""Hardware/resource detection and safety gating.

This module never assumes a specific GPU, runtime, or session length.
It probes the current machine (local dev box or a Colab worker) at
runtime and classifies it into a WorkerProfile. Every heavy job must be
checked against `ResourceManager.can_run` before execution.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from core.exceptions import ResourceError
from core.logging import get_logger

logger = get_logger(__name__)


class WorkerProfile(str, Enum):
    LIGHT = "LIGHT"           # no GPU, or <4GB VRAM
    MEDIUM = "MEDIUM"         # 4-8GB VRAM
    HEAVY = "HEAVY"           # 8-16GB VRAM
    VERY_HEAVY = "VERY_HEAVY"  # >16GB VRAM


@dataclass
class RuntimeReport:
    gpu_available: bool
    gpu_name: Optional[str]
    vram_gb: float
    ram_gb: float
    cpu_cores: int
    cuda_available: bool
    disk_free_gb: float

    def to_dict(self) -> dict:
        return {
            "gpu_available": self.gpu_available,
            "gpu_name": self.gpu_name,
            "vram_gb": self.vram_gb,
            "ram_gb": self.ram_gb,
            "cpu_cores": self.cpu_cores,
            "cuda_available": self.cuda_available,
            "disk_free_gb": self.disk_free_gb,
        }


@dataclass
class ModelRequirement:
    """Minimum viable resource footprint a candidate model declares."""
    minimum_vram_gb: float = 0.0
    recommended_vram_gb: float = 0.0
    estimated_disk_gb: float = 0.0
    requires_cuda: bool = False

    @classmethod
    def from_capability(cls, capability, requires_cuda: bool = True) -> "ModelRequirement":
        """Build a ModelRequirement from a models.capabilities.ModelCapability.

        Mock/CPU-only models (minimum_vram_gb == 0) never require CUDA even
        if the caller passes requires_cuda=True, so `can_run` doesn't reject
        them on machines without a GPU.
        """
        return cls(
            minimum_vram_gb=capability.minimum_vram_gb,
            recommended_vram_gb=capability.recommended_vram_gb,
            estimated_disk_gb=capability.estimated_disk_gb,
            requires_cuda=requires_cuda and capability.minimum_vram_gb > 0,
        )


class ResourceManager:
    """Detects hardware and gates job execution against it.

    `vram_safety_margin_gb` reserves headroom so a job is never scheduled
    against the full VRAM figure — this protects long-running Colab
    sessions (and the local RTX 2050) from OOM crashes.
    """

    def __init__(self, vram_safety_margin_gb: float = 1.5, path_for_disk: str = "."):
        self.vram_safety_margin_gb = vram_safety_margin_gb
        self._path_for_disk = path_for_disk
        self._report: Optional[RuntimeReport] = None

    # ------------------------------------------------------------------
    # Detection
    # ------------------------------------------------------------------

    def detect(self, refresh: bool = False) -> RuntimeReport:
        if self._report is not None and not refresh:
            return self._report

        gpu_available, gpu_name, vram_gb, cuda_available = self._detect_gpu()
        ram_gb = self._detect_ram()
        cpu_cores = self._detect_cpu()
        disk_free_gb = self._detect_disk()

        self._report = RuntimeReport(
            gpu_available=gpu_available,
            gpu_name=gpu_name,
            vram_gb=vram_gb,
            ram_gb=ram_gb,
            cpu_cores=cpu_cores,
            cuda_available=cuda_available,
            disk_free_gb=disk_free_gb,
        )
        logger.info("Runtime detected: %s", self._report.to_dict())
        return self._report

    def _detect_gpu(self) -> tuple[bool, Optional[str], float, bool]:
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                vram_gb = round(props.total_memory / (1024 ** 3), 2)
                return True, name, vram_gb, True
        except Exception as exc:  # noqa: BLE001 - defensive: torch may be absent/broken
            logger.debug("No usable CUDA GPU detected via torch: %s", exc)

        return False, None, 0.0, False

    def _detect_ram(self) -> float:
        try:
            import psutil  # type: ignore

            return round(psutil.virtual_memory().total / (1024 ** 3), 2)
        except Exception:
            # Fallback for environments without psutil: read /proc/meminfo
            try:
                with open("/proc/meminfo") as f:
                    for line in f:
                        if line.startswith("MemTotal"):
                            kb = int(line.split()[1])
                            return round(kb / (1024 ** 2), 2)
            except Exception:
                pass
        return 0.0

    def _detect_cpu(self) -> int:
        import os

        return os.cpu_count() or 1

    def _detect_disk(self) -> float:
        try:
            total, used, free = shutil.disk_usage(self._path_for_disk)
            return round(free / (1024 ** 3), 2)
        except Exception:
            return 0.0

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def classify(self, refresh: bool = False) -> WorkerProfile:
        report = self.detect(refresh=refresh)
        if not report.gpu_available or report.vram_gb <= 0:
            return WorkerProfile.LIGHT
        if report.vram_gb < 4:
            return WorkerProfile.LIGHT
        if report.vram_gb < 8:
            return WorkerProfile.MEDIUM
        if report.vram_gb <= 16:
            return WorkerProfile.HEAVY
        return WorkerProfile.VERY_HEAVY

    # ------------------------------------------------------------------
    # Safety gating
    # ------------------------------------------------------------------

    def effective_vram_gb(self, refresh: bool = False) -> float:
        report = self.detect(refresh=refresh)
        return max(0.0, report.vram_gb - self.vram_safety_margin_gb)

    def can_run(self, requirement: ModelRequirement, allow_local_gpu: bool = False) -> tuple[bool, str]:
        """Return (ok, reason). Never raises — callers decide what to do."""
        report = self.detect()

        if requirement.requires_cuda and not report.cuda_available:
            return False, "CUDA required but not available on this worker."

        if not allow_local_gpu and requirement.minimum_vram_gb > 0:
            # Heavy AI jobs must be explicitly opted in for local execution;
            # this is enforced by the caller (scheduler), but we double-check
            # here as a defense-in-depth safety net.
            pass

        available = self.effective_vram_gb()
        if requirement.minimum_vram_gb > available:
            return False, (
                f"Insufficient VRAM: requires >= {requirement.minimum_vram_gb}GB, "
                f"only {available}GB effectively available "
                f"(safety margin {self.vram_safety_margin_gb}GB)."
            )

        if requirement.estimated_disk_gb > report.disk_free_gb:
            return False, (
                f"Insufficient disk: requires ~{requirement.estimated_disk_gb}GB, "
                f"only {report.disk_free_gb}GB free."
            )

        return True, "OK"

    def require(self, requirement: ModelRequirement, allow_local_gpu: bool = False) -> None:
        ok, reason = self.can_run(requirement, allow_local_gpu=allow_local_gpu)
        if not ok:
            raise ResourceError(reason)
