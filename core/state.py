"""State machine governing project lifecycle.

Every transition is validated against core.models.STATE_TRANSITIONS and
persisted immediately so a crashed/disconnected worker can resume exactly
where it left off (see storage/project_store.py for the on-disk format).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from core.exceptions import StateTransitionError
from core.logging import get_logger
from core.models import STATE_TRANSITIONS, ProjectState

logger = get_logger(__name__)


@dataclass
class StateMachine:
    current_state: ProjectState = ProjectState.CREATED
    history: list[str] = field(default_factory=list)

    def can_transition(self, target: ProjectState) -> bool:
        allowed = STATE_TRANSITIONS.get(self.current_state, [])
        return target in allowed

    def transition(self, target: ProjectState, force: bool = False) -> ProjectState:
        if not force and not self.can_transition(target):
            raise StateTransitionError(
                f"Illegal transition: {self.current_state.value} -> {target.value}. "
                f"Allowed: {[s.value for s in STATE_TRANSITIONS.get(self.current_state, [])]}"
            )
        logger.info("State transition: %s -> %s", self.current_state.value, target.value)
        self.history.append(f"{self.current_state.value}->{target.value}")
        self.current_state = target
        return self.current_state

    def to_dict(self) -> dict:
        return {"current_state": self.current_state.value, "history": self.history}

    @classmethod
    def from_dict(cls, data: dict) -> "StateMachine":
        return cls(
            current_state=ProjectState(data.get("current_state", ProjectState.CREATED.value)),
            history=list(data.get("history", [])),
        )
