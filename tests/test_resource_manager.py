from __future__ import annotations

from core.resource_manager import ModelRequirement, ResourceManager, WorkerProfile


def test_detect_returns_report_with_expected_fields():
    rm = ResourceManager()
    report = rm.detect()
    assert isinstance(report.gpu_available, bool)
    assert report.ram_gb >= 0
    assert report.cpu_cores >= 1
    assert report.disk_free_gb >= 0


def test_classify_no_gpu_is_light():
    rm = ResourceManager()
    # Force a no-GPU report deterministically regardless of test host.
    rm._report = None
    rm._detect_gpu = lambda: (False, None, 0.0, False)  # type: ignore
    profile = rm.classify(refresh=True)
    assert profile == WorkerProfile.LIGHT


def test_classify_thresholds():
    rm = ResourceManager()
    for vram, expected in [(2.0, WorkerProfile.LIGHT), (6.0, WorkerProfile.MEDIUM),
                            (12.0, WorkerProfile.HEAVY), (24.0, WorkerProfile.VERY_HEAVY)]:
        rm._detect_gpu = lambda vram=vram: (True, "fake-gpu", vram, True)  # type: ignore
        profile = rm.classify(refresh=True)
        assert profile == expected, f"vram={vram} expected {expected} got {profile}"


def test_effective_vram_applies_safety_margin():
    rm = ResourceManager(vram_safety_margin_gb=1.5)
    rm._detect_gpu = lambda: (True, "fake-gpu", 10.0, True)  # type: ignore
    assert rm.effective_vram_gb(refresh=True) == 8.5


def test_can_run_rejects_insufficient_vram():
    rm = ResourceManager(vram_safety_margin_gb=1.5)
    rm._detect_gpu = lambda: (True, "fake-gpu", 4.0, True)  # type: ignore
    rm.detect(refresh=True)
    ok, reason = rm.can_run(ModelRequirement(minimum_vram_gb=8.0, requires_cuda=True))
    assert ok is False
    assert "Insufficient VRAM" in reason


def test_can_run_rejects_missing_cuda():
    rm = ResourceManager()
    rm._detect_gpu = lambda: (False, None, 0.0, False)  # type: ignore
    rm.detect(refresh=True)
    ok, reason = rm.can_run(ModelRequirement(minimum_vram_gb=0.0, requires_cuda=True))
    assert ok is False
    assert "CUDA" in reason


def test_can_run_accepts_sufficient_resources():
    rm = ResourceManager(vram_safety_margin_gb=1.0)
    rm._detect_gpu = lambda: (True, "fake-gpu", 16.0, True)  # type: ignore
    rm.detect(refresh=True)
    ok, reason = rm.can_run(ModelRequirement(minimum_vram_gb=8.0, requires_cuda=True))
    assert ok is True


def test_require_raises_on_failure():
    from core.exceptions import ResourceError
    import pytest as _pytest

    rm = ResourceManager()
    rm._detect_gpu = lambda: (False, None, 0.0, False)  # type: ignore
    rm.detect(refresh=True)
    with _pytest.raises(ResourceError):
        rm.require(ModelRequirement(minimum_vram_gb=4.0))
