"""Job scheduler.

Decision flow per spec section 3/61:

    REMOTE GPU available?  -> execute remotely
    unavailable?           -> queue job
    lightweight fallback?  -> use fallback
    no safe fallback?      -> pause job

The scheduler itself does not know how to *do* image generation, TTS,
etc. It only decides whether a job is safe to run now, against which
worker, and hands execution off to a callable supplied by the caller.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

from core.logging import get_logger
from core.resource_manager import ModelRequirement, ResourceManager

logger = get_logger(__name__)


class JobOutcome(str, Enum):
    EXECUTED_REMOTE = "executed_remote"
    EXECUTED_LOCAL = "executed_local"
    EXECUTED_FALLBACK = "executed_fallback"
    QUEUED = "queued"
    PAUSED = "paused"
    FAILED = "failed"


@dataclass
class Job:
    task_type: str
    project_id: str
    scene_id: Optional[str] = None
    priority: int = 5
    gpu_required: bool = False
    requirement: ModelRequirement = field(default_factory=ModelRequirement)
    job_id: str = field(default_factory=lambda: f"job_{uuid.uuid4().hex[:10]}")
    attempts: int = 0
    max_attempts: int = 3


@dataclass
class JobResult:
    job: Job
    outcome: JobOutcome
    result: Optional[object] = None
    reason: str = ""


class Scheduler:
    def __init__(
        self,
        resource_manager: ResourceManager,
        remote_available: bool = False,
        allow_local_gpu: bool = False,
    ):
        self.resource_manager = resource_manager
        self.remote_available = remote_available
        self.allow_local_gpu = allow_local_gpu
        self.queue: list[Job] = []

    def enqueue(self, job: Job) -> None:
        self.queue.append(job)
        self.queue.sort(key=lambda j: -j.priority)

    def run_job(
        self,
        job: Job,
        remote_executor: Optional[Callable[[Job], object]] = None,
        fallback_executor: Optional[Callable[[Job], object]] = None,
    ) -> JobResult:
        job.attempts += 1

        if job.gpu_required:
            if self.remote_available and remote_executor is not None:
                ok, reason = self.resource_manager.can_run(job.requirement, allow_local_gpu=False)
                if ok:
                    try:
                        result = remote_executor(job)
                        return JobResult(job, JobOutcome.EXECUTED_REMOTE, result, "remote GPU executed job")
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("Remote execution failed for %s: %s", job.job_id, exc)
                        return self._handle_failure(job, fallback_executor, str(exc))
                else:
                    logger.info("Remote worker cannot satisfy job %s: %s", job.job_id, reason)

            if self.allow_local_gpu:
                ok, reason = self.resource_manager.can_run(job.requirement, allow_local_gpu=True)
                if ok and remote_executor is not None:
                    try:
                        result = remote_executor(job)
                        return JobResult(job, JobOutcome.EXECUTED_LOCAL, result, "local GPU executed job (opt-in)")
                    except Exception as exc:  # noqa: BLE001
                        return self._handle_failure(job, fallback_executor, str(exc))

            if fallback_executor is not None:
                try:
                    result = fallback_executor(job)
                    return JobResult(job, JobOutcome.EXECUTED_FALLBACK, result, "used lightweight fallback")
                except Exception as exc:  # noqa: BLE001
                    return self._handle_failure(job, None, str(exc))

            self.enqueue(job)
            return JobResult(job, JobOutcome.PAUSED, None, "no GPU worker and no safe fallback; job paused")

        # Non-GPU job: run directly via fallback_executor (treated as the
        # "normal" executor for CPU-only tasks).
        if fallback_executor is not None:
            try:
                result = fallback_executor(job)
                return JobResult(job, JobOutcome.EXECUTED_LOCAL, result, "executed on local CPU")
            except Exception as exc:  # noqa: BLE001
                return self._handle_failure(job, None, str(exc))

        return JobResult(job, JobOutcome.FAILED, None, "no executor provided for non-GPU job")

    def _handle_failure(
        self, job: Job, fallback_executor: Optional[Callable[[Job], object]], reason: str
    ) -> JobResult:
        if job.attempts < job.max_attempts:
            logger.info("Retrying job %s (attempt %d/%d)", job.job_id, job.attempts, job.max_attempts)
            self.enqueue(job)
            return JobResult(job, JobOutcome.QUEUED, None, f"retry scheduled after failure: {reason}")
        if fallback_executor is not None:
            try:
                result = fallback_executor(job)
                return JobResult(job, JobOutcome.EXECUTED_FALLBACK, result, f"fell back after failure: {reason}")
            except Exception as exc:  # noqa: BLE001
                return JobResult(job, JobOutcome.FAILED, None, f"fallback also failed: {exc}")
        return JobResult(job, JobOutcome.FAILED, None, reason)
