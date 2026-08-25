from __future__ import annotations

from core.resource_manager import ModelRequirement, ResourceManager
from core.scheduler import Job, JobOutcome, Scheduler


def _rm_with_vram(vram: float) -> ResourceManager:
    rm = ResourceManager(vram_safety_margin_gb=1.0)
    rm._detect_gpu = lambda: (True, "fake-gpu", vram, True)  # type: ignore
    rm.detect(refresh=True)
    return rm


def test_remote_execution_when_available_and_sufficient():
    rm = _rm_with_vram(16.0)
    scheduler = Scheduler(rm, remote_available=True)
    job = Job(task_type="image_generation", project_id="p1", gpu_required=True,
              requirement=ModelRequirement(minimum_vram_gb=8.0, requires_cuda=True))
    result = scheduler.run_job(job, remote_executor=lambda j: "remote_result")
    assert result.outcome == JobOutcome.EXECUTED_REMOTE
    assert result.result == "remote_result"


def test_falls_back_when_remote_insufficient():
    rm = _rm_with_vram(2.0)
    scheduler = Scheduler(rm, remote_available=True)
    job = Job(task_type="image_generation", project_id="p1", gpu_required=True,
              requirement=ModelRequirement(minimum_vram_gb=8.0, requires_cuda=True))
    result = scheduler.run_job(
        job, remote_executor=lambda j: "remote_result", fallback_executor=lambda j: "fallback_result",
    )
    assert result.outcome == JobOutcome.EXECUTED_FALLBACK
    assert result.result == "fallback_result"


def test_pauses_job_when_no_remote_and_no_fallback():
    rm = _rm_with_vram(0.0)
    scheduler = Scheduler(rm, remote_available=False)
    job = Job(task_type="image_generation", project_id="p1", gpu_required=True,
              requirement=ModelRequirement(minimum_vram_gb=8.0, requires_cuda=True))
    result = scheduler.run_job(job, remote_executor=None, fallback_executor=None)
    assert result.outcome == JobOutcome.PAUSED
    assert job in scheduler.queue


def test_local_gpu_requires_explicit_opt_in():
    rm = _rm_with_vram(16.0)
    scheduler_disallowed = Scheduler(rm, remote_available=False, allow_local_gpu=False)
    job = Job(task_type="image_generation", project_id="p1", gpu_required=True,
              requirement=ModelRequirement(minimum_vram_gb=8.0, requires_cuda=True))
    result = scheduler_disallowed.run_job(job, remote_executor=lambda j: "should_not_run")
    assert result.outcome != JobOutcome.EXECUTED_LOCAL

    scheduler_allowed = Scheduler(rm, remote_available=False, allow_local_gpu=True)
    job2 = Job(task_type="image_generation", project_id="p1", gpu_required=True,
               requirement=ModelRequirement(minimum_vram_gb=8.0, requires_cuda=True))
    result2 = scheduler_allowed.run_job(job2, remote_executor=lambda j: "ran_locally")
    assert result2.outcome == JobOutcome.EXECUTED_LOCAL
    assert result2.result == "ran_locally"


def test_non_gpu_job_runs_via_fallback_executor_directly():
    rm = _rm_with_vram(0.0)
    scheduler = Scheduler(rm, remote_available=False)
    job = Job(task_type="research", project_id="p1", gpu_required=False)
    result = scheduler.run_job(job, fallback_executor=lambda j: "cpu_result")
    assert result.outcome == JobOutcome.EXECUTED_LOCAL
    assert result.result == "cpu_result"


def test_retries_before_falling_back_on_repeated_failure():
    rm = _rm_with_vram(16.0)
    scheduler = Scheduler(rm, remote_available=True)
    job = Job(task_type="image_generation", project_id="p1", gpu_required=True,
              requirement=ModelRequirement(minimum_vram_gb=8.0, requires_cuda=True), max_attempts=2)

    def always_fails(_j):
        raise RuntimeError("boom")

    result1 = scheduler.run_job(job, remote_executor=always_fails, fallback_executor=lambda j: "fallback")
    assert result1.outcome == JobOutcome.QUEUED

    result2 = scheduler.run_job(job, remote_executor=always_fails, fallback_executor=lambda j: "fallback")
    assert result2.outcome == JobOutcome.EXECUTED_FALLBACK
    assert result2.result == "fallback"
