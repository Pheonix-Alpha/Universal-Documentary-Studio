"""Model capability descriptors.

Every AI model known to the system (image, video, TTS, ...) is described
declaratively so the ModelRegistry can select the best compatible one
without hard-coding provider-specific logic anywhere else.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, ConfigDict


class ModelTask(str, Enum):
    IMAGE_GENERATION = "image_generation"
    VIDEO_GENERATION = "video_generation"
    TTS = "tts"
    SPEECH_RECOGNITION = "speech_recognition"
    EMBEDDING = "embedding"


class ModelStatus(str, Enum):
    AVAILABLE = "available"
    DISABLED = "disabled"
    EXPERIMENTAL = "experimental"


class ModelCapability(BaseModel):
    model_name: str
    provider: str
    version: str = "1.0"
    license: str = "unknown"
    minimum_vram_gb: float = 0.0
    recommended_vram_gb: float = 0.0
    estimated_disk_gb: float = 0.0
    supported_resolution: str = "512x512"
    supported_tasks: list[ModelTask] = []
    commercial_use: bool = False
    status: ModelStatus = ModelStatus.AVAILABLE
    quality_rank: int = 0  # higher = better quality, used to break ties
    hf_repo_id: Optional[str] = None  # HuggingFace repo id, for real (non-mock) providers

    model_config = ConfigDict(use_enum_values=False)

    @property
    def is_free_local(self) -> bool:
        """Whether this can run without any paid API key."""
        return self.provider.lower() not in {"openai", "elevenlabs", "midjourney", "runway", "kling"}
