"""VoiceEngine interface — provider-agnostic text-to-speech."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TTSRequest:
    text: str
    voice: str = "default"
    language: str = "en"
    speed: float = 1.0


@dataclass
class TTSResult:
    file_path: str
    duration_seconds: float
    model_name: str
    provider: str


class VoiceEngine(ABC):
    @abstractmethod
    def synthesize(self, request: TTSRequest, output_path: str) -> TTSResult:
        raise NotImplementedError
