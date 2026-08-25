from __future__ import annotations

import os

from core.models import ProjectConfig, ProjectState, QAStatus
from core.pipeline import DocumentaryPipeline
from core.project_manager import ProjectManager


def _make_pipeline(tmp_projects_root, topic, **overrides):
    config = ProjectConfig(topic=topic, mock_mode=True, **overrides)
    pm = ProjectManager(projects_root=tmp_projects_root, config=config)
    pipeline = DocumentaryPipeline(
        pm, video_width=320, video_height=180, fps=10,
        short_width=180, short_height=320, short_fps=10,
    )
    return pm, pipeline


def test_pipeline_stage_by_stage(tmp_projects_root, sample_topic):
    pm, pipeline = _make_pipeline(tmp_projects_root, sample_topic, short_count=3)

    research = pipeline.run_research()
    assert pm.state_machine.current_state == ProjectState.RESEARCH_COMPLETE
    assert len(research.sources) > 0

    fact_check = pipeline.run_fact_check(research)
    assert pm.state_machine.current_state == ProjectState.FACT_CHECK_COMPLETE

    story = pipeline.run_story(research)
    script = pipeline.run_script(research, story)
    assert pm.state_machine.current_state == ProjectState.SCRIPT_COMPLETE
    assert len(script.sections) > 0

    scene_plan = pipeline.run_scenes(script, story)
    assert pm.state_machine.current_state == ProjectState.SCENES_COMPLETE
    assert len(scene_plan.scenes) > 0

    asset_manifest = pipeline.run_assets(scene_plan)
    assert pm.state_machine.current_state == ProjectState.ASSETS_COMPLETE
    assert len(asset_manifest.assets) == len(scene_plan.scenes)
    for asset in asset_manifest.assets:
        if asset.file_path:
            assert os.path.exists(asset.file_path)

    voice_manifest, music_manifest = pipeline.run_audio(scene_plan)
    assert pm.state_machine.current_state == ProjectState.AUDIO_COMPLETE
    assert len(voice_manifest.tracks) == len(scene_plan.scenes)

    render_path = pipeline.run_render(asset_manifest, voice_manifest)
    assert pm.state_machine.current_state == ProjectState.RENDER_COMPLETE
    assert os.path.exists(render_path)
    assert os.path.getsize(render_path) > 0

    qa_report = pipeline.run_qa(research, asset_manifest, render_path, scene_plan)
    assert pm.state_machine.current_state in (ProjectState.HUMAN_REVIEW, ProjectState.QA_FAILED)
    assert qa_report.score >= 0


def test_pipeline_resumes_from_partial_checkpoint(tmp_projects_root, sample_topic):
    pm1, pipeline1 = _make_pipeline(tmp_projects_root, sample_topic)
    research = pipeline1.run_research()
    fact_check = pipeline1.run_fact_check(research)
    story = pipeline1.run_story(research)
    script = pipeline1.run_script(research, story)
    # Stop here — simulate a Colab disconnect before scene planning.

    # Reload the project fresh (as if resuming after a crash).
    config = pm1.config
    pm2 = ProjectManager(projects_root=tmp_projects_root, config=config)
    assert pm2.state_machine.current_state == ProjectState.SCRIPT_COMPLETE

    pipeline2 = DocumentaryPipeline(pm2, video_width=320, video_height=180, fps=10)
    resumed_research = pipeline2.run_research()  # should short-circuit via checkpoint
    assert resumed_research.topic == research.topic

    scene_plan = pipeline2.run_scenes(script, story)
    assert pm2.state_machine.current_state == ProjectState.SCENES_COMPLETE


def test_qa_gate_never_auto_approves(tmp_projects_root, sample_topic):
    pm, pipeline = _make_pipeline(tmp_projects_root, sample_topic)
    result = pipeline.run_full()
    # Regardless of QA score, state must stop at HUMAN_REVIEW or QA_FAILED,
    # never automatically reach APPROVED/EXPORTED.
    assert pm.state_machine.current_state in (ProjectState.HUMAN_REVIEW, ProjectState.QA_FAILED)
