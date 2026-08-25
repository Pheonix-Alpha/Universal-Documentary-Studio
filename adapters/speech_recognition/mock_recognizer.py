"""Mock SpeechRecognizer: returns a single segment spanning the whole
audio file's duration with placeholder text. Real deployments would
swap in faster-whisper here without touching any calling code.
"""
from __future__ import annotations

import contextlib
import wave

from adapters.speech_recognition.base import SpeechRecognizer, TranscriptSegment


class MockSpeechRecognizer(SpeechRecognizer):
    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        duration = 0.0
        try:
            with contextlib.closing(wave.open(audio_path, "r")) as wav_file:
                frames = wav_file.getnframes()
                rate = wav_file.getframerate()
                duration = frames / float(rate) if rate else 0.0
        except Exception:
            duration = 1.0
        return [TranscriptSegment(start=0.0, end=duration, text="[mock transcript]")]
