"""Mock VoiceEngine.

Produces a real, valid WAV file (silence with a very faint low-frequency
hum so it isn't literally all-zero PCM) whose duration matches the
estimated narration length at a realistic speaking pace. This lets every
downstream stage (mixing, ducking, captions sync, FFmpeg muxing) operate
on real audio files rather than fixtures, without needing any TTS model.
"""
from __future__ import annotations

import math
import struct
import wave

from adapters.tts.base import TTSRequest, TTSResult, VoiceEngine

WORDS_PER_MINUTE = 150
SAMPLE_RATE = 22050


def estimate_duration_seconds(text: str, speed: float = 1.0) -> float:
    word_count = max(1, len(text.split()))
    minutes = word_count / WORDS_PER_MINUTE
    return max(0.6, (minutes * 60.0) / max(0.1, speed))


class MockVoiceEngine(VoiceEngine):
    model_name = "mock-tts-v1"
    provider = "mock"

    def synthesize(self, request: TTSRequest, output_path: str) -> TTSResult:
        duration = estimate_duration_seconds(request.text, request.speed)
        n_samples = int(SAMPLE_RATE * duration)

        with wave.open(output_path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(SAMPLE_RATE)
            frames = bytearray()
            amplitude = 200  # very quiet placeholder "voice presence" tone
            freq = 120.0
            for i in range(n_samples):
                t = i / SAMPLE_RATE
                sample = int(amplitude * math.sin(2 * math.pi * freq * t) * (0.5 + 0.5 * math.sin(2 * math.pi * 2 * t)))
                frames += struct.pack("<h", sample)
            wav_file.writeframes(bytes(frames))

        return TTSResult(
            file_path=output_path,
            duration_seconds=round(duration, 2),
            model_name=self.model_name,
            provider=self.provider,
        )
