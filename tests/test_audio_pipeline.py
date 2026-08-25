from __future__ import annotations

import os
import wave

from adapters.tts.base import TTSRequest
from adapters.tts.mock_engine import MockVoiceEngine, estimate_duration_seconds
from engines.voice.audio_pipeline import get_audio_duration_seconds, normalize_narration


def test_estimate_duration_scales_with_word_count():
    short_text = "Hello there."
    long_text = " ".join(["word"] * 300)
    assert estimate_duration_seconds(long_text) > estimate_duration_seconds(short_text)


def test_mock_voice_engine_produces_valid_wav(tmp_path):
    engine = MockVoiceEngine()
    out_path = str(tmp_path / "voice.wav")
    result = engine.synthesize(TTSRequest(text="This is a test narration line."), out_path)
    assert os.path.exists(out_path)
    assert result.duration_seconds > 0

    with wave.open(out_path, "r") as wav_file:
        assert wav_file.getnchannels() == 1
        assert wav_file.getframerate() == 22050


def test_audio_duration_matches_ffprobe(tmp_path):
    engine = MockVoiceEngine()
    out_path = str(tmp_path / "voice.wav")
    result = engine.synthesize(TTSRequest(text="Another sample narration for duration checking."), out_path)
    probed_duration = get_audio_duration_seconds(out_path)
    assert abs(probed_duration - result.duration_seconds) < 0.5


def test_normalize_narration_produces_valid_output(tmp_path):
    engine = MockVoiceEngine()
    raw_path = str(tmp_path / "raw.wav")
    clean_path = str(tmp_path / "clean.wav")
    engine.synthesize(TTSRequest(text="Testing normalization pipeline behavior."), raw_path)
    normalize_narration(raw_path, clean_path)
    assert os.path.exists(clean_path)
    assert get_audio_duration_seconds(clean_path) > 0
