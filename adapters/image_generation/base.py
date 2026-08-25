"""ImageGenerator interface — provider-agnostic AI image generation."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class ImageGenerationRequest:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    seed: int | None = None


@dataclass
class ImageGenerationResult:
    file_path: str
    model_name: str
    provider: str
    seed: int | None
    metadata: dict


class ImageGenerator(ABC):
    @abstractmethod
    def generate(self, request: ImageGenerationRequest, output_path: str) -> ImageGenerationResult:
        raise NotImplementedError

    def unload(self) -> None:
        """Release any loaded model weights / GPU memory.

        No-op by default (mocks have nothing to release); real adapters
        override this so ModelLifecycleManager can free GPU/RAM/disk
        between stages without callers needing to know which kind of
        generator they were handed.
        """
        return None
