"""ProjectManager ties together ProjectStore, StateMachine, and config.

This is the object agents/pipeline code interact with to load/save
checkpoints and move the state machine forward, without needing to know
about file paths or JSON serialization directly.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Type, TypeVar

from pydantic import BaseModel

from core.exceptions import CheckpointError
from core.logging import get_logger
from core.models import ProjectConfig, ProjectState
from core.state import StateMachine
from storage.cache_store import CacheStore
from storage.project_store import ProjectStore

logger = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class ProjectManager:
    def __init__(self, projects_root: str, config: ProjectConfig):
        self.config = config
        self.store = ProjectStore(root=projects_root, project_id=config.project_id)
        self.cache = CacheStore(cache_dir=str(self.store.dir_for("cache")))
        self.state_machine = self._load_or_init_state_machine()
        self._save_config()

    # ------------------------------------------------------------------

    def _load_or_init_state_machine(self) -> StateMachine:
        data = self.store.load_checkpoint("state.json")
        if data:
            logger.info("Resuming project %s from state %s", self.config.project_id, data.get("current_state"))
            return StateMachine.from_dict(data)
        return StateMachine()

    def _save_config(self) -> None:
        if not self.store.has_checkpoint("config.json"):
            self.store.save_checkpoint("config.json", self.config.model_dump())

    def save_state(self) -> None:
        self.store.save_checkpoint("state.json", self.state_machine.to_dict())

    def transition(self, target: ProjectState, force: bool = False) -> ProjectState:
        result = self.state_machine.transition(target, force=force)
        self.save_state()
        return result

    # ------------------------------------------------------------------
    # Typed checkpoint helpers
    # ------------------------------------------------------------------

    def save_model(self, name: str, model: BaseModel) -> str:
        return self.store.save_checkpoint(name, model.model_dump())

    def load_model(self, name: str, model_cls: Type[T]) -> Optional[T]:
        data = self.store.load_checkpoint(name)
        if data is None:
            return None
        try:
            return model_cls.model_validate(data)
        except Exception as exc:
            raise CheckpointError(f"Corrupt checkpoint {name}: {exc}") from exc

    def stage_complete(self, name: str) -> bool:
        return self.store.has_checkpoint(name)

    def resume_summary(self) -> dict:
        return {
            "project_id": self.config.project_id,
            "current_state": self.state_machine.current_state.value,
            "completed_checkpoints": self.store.completed_stages(),
        }
