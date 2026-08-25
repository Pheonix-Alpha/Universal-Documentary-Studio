from __future__ import annotations

from app.config import load_app_config
from app.startup import run_startup


def test_load_app_config_has_expected_defaults():
    config = load_app_config(path="/nonexistent/path.yaml")
    assert config["mock_mode"] is True
    assert config["local_gpu_enabled"] is False
    assert "vram_safety_margin_gb" in config["resource"]


def test_load_app_config_reads_real_file():
    config = load_app_config()
    assert config["video"]["long_form"]["width"] == 1920
    assert config["video"]["short"]["width"] == 1080


def test_run_startup_returns_summary_without_crashing():
    summary = run_startup(lightweight=True)
    assert "runtime_report" in summary
    assert "worker_profile" in summary
    assert summary["lightweight_mode"] is True


def test_ui_builds_without_launching():
    from app.ui import build_interface
    demo = build_interface()
    assert demo is not None
