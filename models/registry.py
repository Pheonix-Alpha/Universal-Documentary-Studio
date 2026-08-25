"""ModelRegistry: resource-aware, provider-agnostic model selection.

The registry never hard-codes "use model X". Instead every model
registers a ModelCapability, and callers ask for the *best compatible*
model given the available VRAM / commercial-use requirements / task.
If nothing fits, the registry returns None and callers must fall back
to a non-AI technique (this is enforced by the agents, not here).
"""
from __future__ import annotations

from core.logging import get_logger
from models.capabilities import ModelCapability, ModelStatus, ModelTask

logger = get_logger(__name__)


class ModelRegistry:
    def __init__(self) -> None:
        self._models: list[ModelCapability] = []
        self._register_defaults()

    # ------------------------------------------------------------------

    def register(self, model: ModelCapability) -> None:
        self._models.append(model)

    def all(self) -> list[ModelCapability]:
        return list(self._models)

    def _register_defaults(self) -> None:
        # A small, deliberately conservative default catalog of well-known
        # open, free-to-run (locally or on Colab) model families. Real
        # deployments can register more via config/models.yaml.
        self.register(ModelCapability(
            model_name="mock-image-v1",
            provider="mock",
            license="internal-testing",
            minimum_vram_gb=0.0,
            recommended_vram_gb=0.0,
            estimated_disk_gb=0.0,
            supported_resolution="1024x1024",
            supported_tasks=[ModelTask.IMAGE_GENERATION],
            commercial_use=True,
            status=ModelStatus.AVAILABLE,
            quality_rank=0,
        ))
        self.register(ModelCapability(
            model_name="sd-turbo-small",
            provider="stability-open",
            license="openrail-m",
            minimum_vram_gb=4.0,
            recommended_vram_gb=6.0,
            estimated_disk_gb=2.5,
            supported_resolution="512x512",
            supported_tasks=[ModelTask.IMAGE_GENERATION],
            commercial_use=True,
            status=ModelStatus.AVAILABLE,
            quality_rank=5,
            hf_repo_id="stabilityai/sd-turbo",
        ))
        self.register(ModelCapability(
            model_name="sdxl-base",
            provider="stability-open",
            license="openrail-m",
            minimum_vram_gb=8.0,
            recommended_vram_gb=12.0,
            estimated_disk_gb=7.0,
            supported_resolution="1024x1024",
            supported_tasks=[ModelTask.IMAGE_GENERATION],
            commercial_use=True,
            status=ModelStatus.AVAILABLE,
            quality_rank=8,
            hf_repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        ))
        self.register(ModelCapability(
            model_name="mock-video-v1",
            provider="mock",
            license="internal-testing",
            supported_tasks=[ModelTask.VIDEO_GENERATION],
            commercial_use=True,
            status=ModelStatus.AVAILABLE,
            quality_rank=0,
        ))
        self.register(ModelCapability(
            model_name="svd-open",
            provider="stability-open",
            license="openrail-m",
            minimum_vram_gb=12.0,
            recommended_vram_gb=16.0,
            estimated_disk_gb=9.0,
            supported_resolution="1024x576",
            supported_tasks=[ModelTask.VIDEO_GENERATION],
            commercial_use=True,
            status=ModelStatus.AVAILABLE,
            quality_rank=7,
            hf_repo_id="stabilityai/stable-video-diffusion-img2vid",
        ))
        self.register(ModelCapability(
            model_name="mock-tts-v1",
            provider="mock",
            license="internal-testing",
            supported_tasks=[ModelTask.TTS],
            commercial_use=True,
            status=ModelStatus.AVAILABLE,
            quality_rank=0,
        ))
        self.register(ModelCapability(
            model_name="piper-tts",
            provider="rhasspy-piper",
            license="mit",
            minimum_vram_gb=0.0,
            recommended_vram_gb=0.0,
            estimated_disk_gb=0.1,
            supported_tasks=[ModelTask.TTS],
            commercial_use=True,
            status=ModelStatus.AVAILABLE,
            quality_rank=4,
        ))
        self.register(ModelCapability(
            model_name="coqui-xtts",
            provider="coqui",
            license="coqui-public-model-license",
            minimum_vram_gb=4.0,
            recommended_vram_gb=6.0,
            estimated_disk_gb=2.0,
            supported_tasks=[ModelTask.TTS],
            commercial_use=False,  # non-commercial license by default - flagged
            status=ModelStatus.AVAILABLE,
            quality_rank=8,
        ))

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _candidates(
        self,
        task: ModelTask,
        available_vram: float,
        commercial_use: bool = True,
        require_free_local: bool = False,
    ) -> list[ModelCapability]:
        out = []
        for m in self._models:
            if task not in m.supported_tasks:
                continue
            if m.status != ModelStatus.AVAILABLE:
                continue
            if commercial_use and not m.commercial_use:
                continue
            if require_free_local and not m.is_free_local:
                continue
            if m.minimum_vram_gb > available_vram:
                continue
            out.append(m)
        out.sort(key=lambda m: m.quality_rank, reverse=True)
        return out

    def get_best_model(
        self,
        task: ModelTask,
        available_vram: float,
        commercial_use: bool = True,
        require_free_local: bool = False,
    ) -> ModelCapability | None:
        candidates = self._candidates(task, available_vram, commercial_use, require_free_local)
        if not candidates:
            logger.warning(
                "No compatible model for task=%s vram=%.1f commercial_use=%s",
                task.value, available_vram, commercial_use,
            )
            return None
        chosen = candidates[0]
        logger.info("Selected model %s for task=%s (vram=%.1f)", chosen.model_name, task.value, available_vram)
        return chosen

    def get_compatible_models(
        self,
        task: ModelTask,
        available_vram: float,
        commercial_use: bool = True,
        require_free_local: bool = False,
    ) -> list[ModelCapability]:
        """All models that fit the given VRAM budget, best quality first.

        Used by ModelLifecycleManager to walk down the list and try the
        next-best model when the top choice fails to download/load (spec
        section 11: automatic fallback rather than a hard crash).
        """
        return self._candidates(task, available_vram, commercial_use, require_free_local)

    def all_for_task(self, task: ModelTask) -> list[ModelCapability]:
        """Every registered model for a task, regardless of VRAM budget."""
        return [m for m in self._models if task in m.supported_tasks]

    def get_best_image_model(self, available_vram: float, commercial_use: bool = True) -> ModelCapability | None:
        return self.get_best_model(ModelTask.IMAGE_GENERATION, available_vram, commercial_use)

    def get_best_video_model(self, available_vram: float, commercial_use: bool = True) -> ModelCapability | None:
        return self.get_best_model(ModelTask.VIDEO_GENERATION, available_vram, commercial_use)

    def get_best_tts_model(self, available_vram: float, commercial_use: bool = True) -> ModelCapability | None:
        return self.get_best_model(ModelTask.TTS, available_vram, commercial_use)
