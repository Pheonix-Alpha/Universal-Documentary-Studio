"""VideoGenerator interface — provider-agnostic AI video generation.

This is deliberately an *enhancement*, never a required dependency
(see spec section 25). Agents must always have a non-AI-video fallback
path (image animation / licensed media) available.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class VideoGenerationRequest:
    prompt: str
    source_image_path: str | None = None
    duration_seconds: float = 4.0
    width: int = 1024
    height: int = 576
    fps: int = 24


@dataclass
class VideoGenerationResult:
    file_path: str
    model_name: str
    provider: str
    metadata: dict


class VideoGenerator(ABC):
    @abstractmethod
    def generate(self, request: VideoGenerationRequest, output_path: str) -> VideoGenerationResult:
        raise NotImplementedError

    def unload(self) -> None:
        """Release any loaded model weights / GPU memory. No-op by default."""
        return None
