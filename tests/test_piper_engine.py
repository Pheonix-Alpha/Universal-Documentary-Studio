"""Tests for the Piper TTS adapter."""

from pathlib import Path

import pytest

from adapters.tts.piper_engine import PiperVoiceEngine


def test_missing_piper_model_fails_cleanly(tmp_path: Path):
    missing_model = tmp_path / "missing.onnx"

    with pytest.raises(FileNotFoundError, match="Piper model not found"):
        PiperVoiceEngine(str(missing_model))