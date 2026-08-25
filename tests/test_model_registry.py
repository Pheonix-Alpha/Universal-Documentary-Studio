from __future__ import annotations

from models.capabilities import ModelTask
from models.registry import ModelRegistry


def test_returns_none_when_no_model_fits():
    registry = ModelRegistry()
    model = registry.get_best_image_model(available_vram=0.0, commercial_use=True)
    # mock-image-v1 requires 0 VRAM, so it should still be selected.
    assert model is not None
    assert model.model_name == "mock-image-v1"


def test_selects_highest_quality_within_vram_budget():
    registry = ModelRegistry()
    model = registry.get_best_image_model(available_vram=10.0, commercial_use=True)
    assert model is not None
    # sdxl-base (quality_rank=8) requires 8GB and should beat sd-turbo-small (rank 5).
    assert model.model_name == "sdxl-base"


def test_falls_back_to_smaller_model_when_budget_tight():
    registry = ModelRegistry()
    model = registry.get_best_image_model(available_vram=5.0, commercial_use=True)
    assert model is not None
    assert model.model_name == "sd-turbo-small"


def test_commercial_use_filter_excludes_noncommercial_models():
    registry = ModelRegistry()
    model = registry.get_best_model(ModelTask.TTS, available_vram=8.0, commercial_use=True)
    assert model is not None
    assert model.commercial_use is True
    assert model.model_name != "coqui-xtts"  # coqui-xtts is non-commercial by default


def test_video_model_selection_respects_vram():
    registry = ModelRegistry()
    model = registry.get_best_video_model(available_vram=20.0, commercial_use=True)
    assert model is not None
    assert model.model_name == "svd-open"

    model_low_vram = registry.get_best_video_model(available_vram=2.0, commercial_use=True)
    assert model_low_vram is not None
    assert model_low_vram.model_name == "mock-video-v1"
