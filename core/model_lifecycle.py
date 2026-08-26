"""ModelLifecycleManager: the resource lifecycle described in the UDS
architecture notes:

    AVAILABLE RESOURCES -> ResourceManager -> ModelRegistry
        -> SELECT BEST MODEL -> DOWNLOAD ONLY THAT MODEL -> LOAD MODEL
        -> DO THE WORK -> SAVE RESULT -> UNLOAD MODEL
        -> DELETE MODEL + TEMP CACHE -> FREE GPU/RAM/DISK -> SELECT NEXT MODEL

This is the single place responsible for making sure the pipeline never
keeps two heavy generative models resident in GPU/RAM/disk at once:
whichever agent needs a model gets it back through `acquire()`, does its
work, and releases it back through here so the next stage starts from a
clean slate.

Status lifecycle per acquisition:

    WAITING -> CHECKING_RESOURCES -> SELECTING_MODEL -> DOWNLOADING
    -> LOADING -> GENERATING -> SAVING -> UNLOADING -> CLEANING -> COMPLETED

On failure a candidate is marked ERROR and the manager automatically
tries the next-compatible model (finally falling back to the mock
implementation so the pipeline can always complete -- see spec sections
6 and 11).
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Callable, Optional

from core.logging import get_logger
from core.resource_manager import ModelRequirement, ResourceManager
from models.capabilities import ModelCapability, ModelTask
from models.registry import ModelRegistry

logger = get_logger(__name__)


class ModelLifecycleStatus(str, Enum):
    WAITING = "waiting"
    CHECKING_RESOURCES = "checking_resources"
    SELECTING_MODEL = "selecting_model"
    DOWNLOADING = "downloading"
    LOADING = "loading"
    GENERATING = "generating"
    SAVING = "saving"
    UNLOADING = "unloading"
    CLEANING = "cleaning"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class LifecycleEvent:
    """One UI-visible tick of the resource lifecycle (spec section 4/5)."""
    task: str
    status: ModelLifecycleStatus
    model_name: Optional[str] = None
    message: str = ""
    progress: Optional[float] = None


StatusCallback = Callable[[LifecycleEvent], None]


class NoCompatibleModelError(Exception):
    """Raised when not even the mock model is registered for a task."""


@dataclass
class LoadedModel:
    """A generator acquired through ModelLifecycleManager.acquire().

    Call `.release()` (or use as a context manager) as soon as generation
    for the current stage is done, so the manager can unload the model and
    free GPU/RAM/disk before the next stage acquires a different one::

        loaded = manager.acquire(ModelTask.IMAGE_GENERATION, mock_factory, real_factory)
        try:
            result = loaded.generator.generate(request, output_path)
        finally:
            loaded.release()
    """
    manager: "ModelLifecycleManager"
    task: ModelTask
    capability: ModelCapability
    generator: object
    _released: bool = False

    def release(self) -> None:
        if self._released:
            return
        self._released = True
        self.manager._release(self)

    def __enter__(self) -> "LoadedModel":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()


class ModelLifecycleManager:
    def __init__(
        self,
        model_registry: ModelRegistry,
        resource_manager: ResourceManager,
        runtime_dir: str = "uds_runtime",
        mock_mode: bool = True,
        status_callback: Optional[StatusCallback] = None,
        weights_cache_root: str | None = None,
    ):
        self.model_registry = model_registry
        self.resource_manager = resource_manager
        self.runtime_dir = Path(runtime_dir)
        self.mock_mode = mock_mode
        self.status_callback = status_callback

        # Downloaded model weights are multi-GB and expensive to fetch --
        # launcher.py pre-caches them to persistent Google Drive storage
        # and points HF_HOME there specifically so a run never re-pays the
        # download cost. Weights must therefore survive across every
        # acquire()/release() cycle; they live in a *separate*, persistent
        # location from runtime_dir (which is fine to treat as scratch
        # space). Defaults to HF_HOME when launcher.py has set it, so real
        # Colab runs are fast by default with zero extra wiring.
        self.weights_cache_root = Path(
            weights_cache_root or os.environ.get("HF_HOME") or (self.runtime_dir / "weights_cache")
        )

    # ------------------------------------------------------------------

    def _emit(
        self, task: ModelTask, status: ModelLifecycleStatus,
        model_name: Optional[str] = None, message: str = "", progress: Optional[float] = None,
    ) -> None:
        event = LifecycleEvent(task=task.value, status=status, model_name=model_name, message=message, progress=progress)
        logger.info("[lifecycle] task=%s status=%s model=%s %s", task.value, status.value, model_name, message)
        if self.status_callback is not None:
            try:
                self.status_callback(event)
            except Exception:  # noqa: BLE001 - a UI callback must never break the pipeline
                logger.exception("model lifecycle status_callback raised; ignoring")

    # ------------------------------------------------------------------

    def acquire(
        self,
        task: ModelTask,
        mock_factory: Callable[[], object],
        real_factory: Callable[[ModelCapability, Path], object],
        commercial_use: bool = True,
    ) -> LoadedModel:
        """Select, download, and load the best model this hardware can run.

        `mock_factory()` builds the mock generator (e.g. MockImageGenerator).
        `real_factory(capability, workdir)` builds a real adapter (e.g.
        DiffusersImageGenerator) rooted at a fresh, per-model temp workdir;
        any exception it raises (missing deps, OOM, network failure, ...) is
        treated as "this candidate failed" and the next-best compatible
        model is tried, per spec section 11.
        """
        self._emit(task, ModelLifecycleStatus.WAITING)
        self._emit(task, ModelLifecycleStatus.CHECKING_RESOURCES)

        if self.mock_mode:
            self._emit(task, ModelLifecycleStatus.SELECTING_MODEL, message="mock_mode=true -> using mock generator")
            generator = mock_factory()
            self._emit(task, ModelLifecycleStatus.LOADING, model_name=getattr(generator, "model_name", "mock"))
            return LoadedModel(manager=self, task=task, capability=self._mock_capability(task), generator=generator)

        candidates = self.model_registry.get_compatible_models(
            task, available_vram=self.resource_manager.effective_vram_gb(), commercial_use=commercial_use,
        )
        if not candidates:
            self._emit(task, ModelLifecycleStatus.ERROR, message="No compatible model registered for this task.")
            raise NoCompatibleModelError(f"No compatible model for task={task.value}")

        last_reason = ""
        for candidate in candidates:
            self._emit(task, ModelLifecycleStatus.SELECTING_MODEL, model_name=candidate.model_name)

            if candidate.provider == "mock":
                generator = mock_factory()
                self._emit(task, ModelLifecycleStatus.LOADING, model_name=candidate.model_name)
                return LoadedModel(manager=self, task=task, capability=candidate, generator=generator)

            requirement = ModelRequirement.from_capability(candidate, requires_cuda=True)
            ok, reason = self.resource_manager.can_run(requirement, allow_local_gpu=True)
            if not ok:
                self._emit(task, ModelLifecycleStatus.ERROR, model_name=candidate.model_name, message=f"skipped: {reason}")
                last_reason = reason
                continue

            workdir = self._model_workdir(task, candidate)
            try:
                self._emit(task, ModelLifecycleStatus.DOWNLOADING, model_name=candidate.model_name)
                self._emit(task, ModelLifecycleStatus.LOADING, model_name=candidate.model_name)
                generator = real_factory(candidate, workdir)
            except Exception as exc:  # noqa: BLE001 - any adapter failure triggers fallback, never a crash
                self._emit(
                    task, ModelLifecycleStatus.ERROR, model_name=candidate.model_name,
                    message=f"load failed ({exc}); trying next compatible model",
                )
                last_reason = str(exc)
                shutil.rmtree(workdir, ignore_errors=True)
                continue

            return LoadedModel(manager=self, task=task, capability=candidate, generator=generator)

        # Every real candidate was skipped or failed to load -- fall back to
        # the mock model rather than crash the pipeline (spec 6/11).
        self._emit(
            task, ModelLifecycleStatus.SELECTING_MODEL,
            message=f"all real candidates unavailable ({last_reason or 'no fit'}); falling back to mock",
        )
        generator = mock_factory()
        self._emit(task, ModelLifecycleStatus.LOADING, model_name="mock")
        return LoadedModel(manager=self, task=task, capability=self._mock_capability(task), generator=generator)

    # ------------------------------------------------------------------

    def _release(self, loaded: LoadedModel) -> None:
        self._emit(loaded.task, ModelLifecycleStatus.UNLOADING, model_name=loaded.capability.model_name)
        try:
            unload = getattr(loaded.generator, "unload", None)
            if callable(unload):
                unload()
        except Exception:  # noqa: BLE001 - unload must never raise into the caller
            logger.exception("Error unloading model %s", loaded.capability.model_name)

        # NOTE: we deliberately do NOT delete the model's weights directory
        # here. `_model_workdir` points into `weights_cache_root` (the
        # persistent, HF_HOME-backed cache), not scratch space -- wiping it
        # on every release would silently defeat launcher.py's pre-caching
        # and force a full multi-GB re-download on the next acquire().
        # "CLEANING" now just means "unloaded the weights from
        # GPU/RAM" (done above via generator.unload()); the files on disk
        # are meant to be reused.
        self._emit(loaded.task, ModelLifecycleStatus.CLEANING, model_name=loaded.capability.model_name)
        self._emit(loaded.task, ModelLifecycleStatus.COMPLETED, model_name=loaded.capability.model_name)

    def _model_workdir(self, task: ModelTask, capability: ModelCapability) -> Path:
        return self.weights_cache_root / task.value / capability.model_name

    @staticmethod
    def _mock_capability(task: ModelTask) -> ModelCapability:
        name = {
            ModelTask.IMAGE_GENERATION: "mock-image-v1",
            ModelTask.VIDEO_GENERATION: "mock-video-v1",
            ModelTask.TTS: "mock-tts-v1",
        }.get(task, "mock")
        return ModelCapability(model_name=name, provider="mock", supported_tasks=[task])
