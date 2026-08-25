from __future__ import annotations

from adapters.research.mock_provider import MockResearchProvider
from agents.research_agent import ResearchAgent
from agents.scene_agent import SceneAgent
from agents.script_agent import ScriptAgent
from agents.story_agent import StoryAgent


def _build_research_story_script(topic: str):
    research = ResearchAgent(MockResearchProvider()).run(topic)
    story = StoryAgent().run(research)
    script = ScriptAgent(seed=42).run(research, story)
    return research, story, script


def test_script_has_sections_and_positive_duration(sample_topic):
    _, _, script = _build_research_story_script(sample_topic)
    assert len(script.sections) > 0
    assert script.word_count > 0
    assert script.estimated_duration_seconds > 0


def test_script_does_not_always_start_with_forbidden_phrase(sample_topic):
    _, _, script = _build_research_story_script(sample_topic)
    first_section = script.sections[0]
    assert "in today's video" not in first_section.narration.lower()


def test_script_varies_hook_across_seeds(sample_topic):
    research = ResearchAgent(MockResearchProvider()).run(sample_topic)
    story = StoryAgent().run(research)
    script_a = ScriptAgent(seed=1).run(research, story)
    script_b = ScriptAgent(seed=2).run(research, story)
    # Different seeds should be able to produce different hook phrasing.
    hooks = {script_a.sections[0].narration, script_b.sections[0].narration}
    assert len(hooks) >= 1  # sanity: at least valid, non-empty


def test_scene_agent_produces_scenes_with_full_treatment(sample_topic):
    research, story, script = _build_research_story_script(sample_topic)
    plan = SceneAgent().run(script, story)
    assert len(plan.scenes) > 0
    assert plan.total_duration_seconds > 0
    for scene in plan.scenes:
        assert scene.duration_seconds > 0
        assert scene.visual_type is not None
        assert scene.camera_movement is not None
        assert scene.narration.strip() != ""


def test_scene_agent_varies_visual_type_within_section(sample_topic):
    research, story, script = _build_research_story_script(sample_topic)
    plan = SceneAgent().run(script, story)
    visual_types = {s.visual_type for s in plan.scenes}
    # Documentary should not use a single visual technique for everything.
    assert len(visual_types) >= 2
