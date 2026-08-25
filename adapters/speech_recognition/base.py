"""SpeechRecognizer interface — provider-agnostic ASR (e.g. faster-whisper).

Used mainly to verify narration/caption alignment during QA. Optional in
MOCK_MODE, where a trivial pass-through recognizer is used instead.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TranscriptSegment:
    start: float
    end: float
    text: str


class SpeechRecognizer(ABC):
    @abstractmethod
    def transcribe(self, audio_path: str) -> list[TranscriptSegment]:
        raise NotImplementedError
