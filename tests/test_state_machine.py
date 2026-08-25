from __future__ import annotations

import pytest

from core.exceptions import StateTransitionError
from core.models import ProjectState
from core.state import StateMachine


def test_initial_state_is_created():
    sm = StateMachine()
    assert sm.current_state == ProjectState.CREATED


def test_legal_transition_succeeds():
    sm = StateMachine()
    sm.transition(ProjectState.RESEARCHING)
    assert sm.current_state == ProjectState.RESEARCHING
    assert sm.history == ["CREATED->RESEARCHING"]


def test_illegal_transition_raises():
    sm = StateMachine()
    with pytest.raises(StateTransitionError):
        sm.transition(ProjectState.SCRIPT_COMPLETE)


def test_forced_transition_bypasses_validation():
    sm = StateMachine()
    sm.transition(ProjectState.SCRIPT_COMPLETE, force=True)
    assert sm.current_state == ProjectState.SCRIPT_COMPLETE


def test_full_happy_path_transitions():
    sm = StateMachine()
    path = [
        ProjectState.RESEARCHING, ProjectState.RESEARCH_COMPLETE,
        ProjectState.FACT_CHECKING, ProjectState.FACT_CHECK_COMPLETE,
        ProjectState.SCRIPTING, ProjectState.SCRIPT_COMPLETE,
        ProjectState.SCENE_PLANNING, ProjectState.SCENES_COMPLETE,
        ProjectState.ASSET_GENERATION, ProjectState.ASSETS_COMPLETE,
        ProjectState.AUDIO_GENERATION, ProjectState.AUDIO_COMPLETE,
        ProjectState.RENDERING, ProjectState.RENDER_COMPLETE,
        ProjectState.QA, ProjectState.HUMAN_REVIEW, ProjectState.APPROVED,
        ProjectState.EXPORTED,
    ]
    for target in path:
        sm.transition(target)
    assert sm.current_state == ProjectState.EXPORTED


def test_qa_failed_can_return_to_asset_generation():
    sm = StateMachine()
    sm.current_state = ProjectState.QA_FAILED
    sm.transition(ProjectState.ASSET_GENERATION)
    assert sm.current_state == ProjectState.ASSET_GENERATION


def test_serialization_round_trip():
    sm = StateMachine()
    sm.transition(ProjectState.RESEARCHING)
    data = sm.to_dict()
    restored = StateMachine.from_dict(data)
    assert restored.current_state == ProjectState.RESEARCHING
    assert restored.history == sm.history
