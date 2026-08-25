"""
Regression tests for:
- the ProjectConfig.tts wiring fix (previously AttributeError in real mode)
- launcher.py's testable, non-network steps (directory prep, cache env,
  hardware detection passthrough, config writing)

Network-dependent steps (Drive mount, pip install, model downloads,
Gradio launch) are intentionally NOT exercised here — they're covered by
launcher.py's own --skip-* flags for manual/Colab verification.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.models import ProjectConfig, TTSConfig
from core.pipeline import DocumentaryPipeline
from core.project_manager import ProjectManager

launcher = importlib.import_module("launcher")


# ---------------------------------------------------------------------------
# tts config wiring
# ---------------------------------------------------------------------------

def test_project_config_has_tts_field_with_mock_default():
    config = ProjectConfig(topic="x")
    assert config.tts.provider == "mock"
    assert config.tts.model_path == ""


def test_mock_mode_true_always_uses_mock_voice_engine(tmp_path):
    config = ProjectConfig(topic="x", mock_mode=True, tts=TTSConfig(provider="piper", model_path="/nonexistent"))
    pm = ProjectManager(projects_root=str(tmp_path), config=config)
    pipeline = DocumentaryPipeline(pm)
    assert type(pipeline.audio_agent.voice_engine).__name__ == "MockVoiceEngine"


def test_real_mode_with_mock_tts_provider_does_not_raise(tmp_path):
    """Previously: real mode (mock_mode=False) + tts.provider='mock' raised
    ValueError('Unsupported TTS provider: mock') because only 'piper' was
    handled. Now it should resolve to MockVoiceEngine instead of crashing."""
    config = ProjectConfig(topic="x", mock_mode=False, local_gpu_enabled=False, tts=TTSConfig(provider="mock"))
    pm = ProjectManager(projects_root=str(tmp_path), config=config)
    pipeline = DocumentaryPipeline(pm)
    assert type(pipeline.audio_agent.voice_engine).__name__ == "MockVoiceEngine"


def test_real_mode_with_piper_provider_but_missing_model_raises_filenotfound(tmp_path):
    config = ProjectConfig(topic="x", mock_mode=False, tts=TTSConfig(provider="piper", model_path="/nonexistent/x.onnx"))
    pm = ProjectManager(projects_root=str(tmp_path), config=config)
    with pytest.raises(FileNotFoundError):
        DocumentaryPipeline(pm)


def test_real_mode_with_piper_provider_and_present_model_constructs(tmp_path):
    model_path = tmp_path / "voice.onnx"
    model_path.write_bytes(b"fake")
    (tmp_path / "voice.onnx.json").write_text("{}")
    config = ProjectConfig(topic="x", mock_mode=False, tts=TTSConfig(provider="piper", model_path=str(model_path)))
    pm = ProjectManager(projects_root=str(tmp_path), config=config)
    pipeline = DocumentaryPipeline(pm)
    assert type(pipeline.audio_agent.voice_engine).__name__ == "PiperVoiceEngine"


def test_real_mode_with_unknown_provider_still_raises(tmp_path):
    config = ProjectConfig(topic="x", mock_mode=False, tts=TTSConfig(provider="elevenlabs"))
    pm = ProjectManager(projects_root=str(tmp_path), config=config)
    with pytest.raises(ValueError):
        DocumentaryPipeline(pm)


# ---------------------------------------------------------------------------
# launcher.py — non-network steps
# ---------------------------------------------------------------------------

def test_is_colab_false_outside_colab():
    assert launcher.is_colab() is False


def test_mount_drive_force_local_returns_local_path(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)
    root = launcher.mount_drive(force_local=True)
    assert launcher.DRIVE_ROOT_NAME in str(root)
    assert "content/drive" not in str(root)


def test_prepare_directories_creates_full_layout(tmp_path):
    root = tmp_path / "Universal-Documentary-Studio"
    dirs = launcher.prepare_directories(root)
    for key, parts in launcher.DIR_LAYOUT.items():
        expected = root.joinpath(*parts)
        assert dirs[key] == expected
        assert expected.exists() and expected.is_dir()


def test_configure_cache_env_sets_expected_vars(tmp_path, monkeypatch):
    root = tmp_path / "Universal-Documentary-Studio"
    dirs = launcher.prepare_directories(root)
    env = launcher.configure_cache_env(dirs)
    assert set(env) == {"HF_HOME", "TRANSFORMERS_CACHE", "HF_DATASETS_CACHE", "TORCH_HOME"}
    for key, value in env.items():
        assert Path(value).exists() or Path(value).parent.exists()
        import os
        assert os.environ[key] == value


def test_write_runtime_config_mock_when_no_piper(tmp_path, monkeypatch):
    root = tmp_path / "Universal-Documentary-Studio"
    dirs = launcher.prepare_directories(root)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)

    class FakeReport:
        gpu_available = False
        cuda_available = False

    config_path = launcher.write_runtime_config(dirs, {"piper_path": None}, FakeReport())
    data = yaml.safe_load(config_path.read_text())
    assert data["mock_mode"] is True
    assert data["local_gpu_enabled"] is False
    assert data["tts"]["provider"] == "mock"
    assert data["projects_root"] == str(dirs["projects"])


def test_write_runtime_config_piper_when_voice_ready(tmp_path, monkeypatch):
    root = tmp_path / "Universal-Documentary-Studio"
    dirs = launcher.prepare_directories(root)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)

    class FakeReport:
        gpu_available = True
        cuda_available = True

    piper_path = dirs["piper_models"] / "en_US-ryan-high.onnx"
    config_path = launcher.write_runtime_config(dirs, {"piper_path": piper_path}, FakeReport())
    data = yaml.safe_load(config_path.read_text())
    assert data["mock_mode"] is False
    assert data["local_gpu_enabled"] is True
    assert data["tts"]["provider"] == "piper"
    assert data["tts"]["model_path"] == str(piper_path)


def test_written_config_actually_constructs_a_working_pipeline(tmp_path, monkeypatch):
    """End-to-end: whatever launcher.write_runtime_config produces must be
    loadable by app.config and buildable into a real DocumentaryPipeline
    without raising -- this is the exact bug that shipped before the fix."""
    root = tmp_path / "Universal-Documentary-Studio"
    dirs = launcher.prepare_directories(root)
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    monkeypatch.setattr(launcher, "REPO_ROOT", tmp_path)

    class FakeReport:
        gpu_available = False
        cuda_available = False

    launcher.write_runtime_config(dirs, {"piper_path": None}, FakeReport())

    import app.config as app_config_module
    app_config = app_config_module.load_app_config(tmp_path / "config" / "config.yaml")

    config = ProjectConfig(
        topic="Regression test",
        mock_mode=app_config["mock_mode"],
        local_gpu_enabled=app_config["local_gpu_enabled"],
        tts=TTSConfig(**app_config["tts"]),
    )
    pm = ProjectManager(projects_root=str(tmp_path / "projects"), config=config)
    pipeline = DocumentaryPipeline(pm)
    assert type(pipeline.audio_agent.voice_engine).__name__ == "MockVoiceEngine"


def test_detect_hardware_returns_report_and_profile():
    rm, report, profile = launcher.detect_hardware()
    assert report.ram_gb >= 0
    assert report.cpu_cores >= 1
    assert profile.value in {"LIGHT", "MEDIUM", "HEAVY", "VERY_HEAVY"}


def test_prepare_ai_models_skip_returns_empty_selection(tmp_path):
    root = tmp_path / "Universal-Documentary-Studio"
    dirs = launcher.prepare_directories(root)
    from core.resource_manager import ResourceManager
    rm = ResourceManager(vram_safety_margin_gb=1.5, path_for_disk=str(tmp_path))
    rm.detect(refresh=True)
    result = launcher.prepare_ai_models(dirs, rm, skip=True)
    assert result["piper_path"] is None
    assert result["image_model"] is None


def test_validate_runs_without_raising():
    summary = launcher.validate()
    assert "runtime_report" in summary
    assert "worker_profile" in summary
