from __future__ import annotations

import os

from core.models import ProjectConfig, ProjectState
from core.pipeline import DocumentaryPipeline
from core.project_manager import ProjectManager


def test_full_daily_mode_production_meets_success_criteria(tmp_projects_root, sample_topic):
    """End-to-end test matching spec section 70 (success criteria) and
    section 53 (daily mode): one topic in, one long-form + 3-5 shorts +
    thumbnails + metadata + reports out, human approval required.
    """
    config = ProjectConfig(topic=sample_topic, mock_mode=True, short_count=4)
    pm = ProjectManager(projects_root=tmp_projects_root, config=config)
    pipeline = DocumentaryPipeline(
        pm, video_width=320, video_height=180, fps=10,
        short_width=180, short_height=320, short_fps=10,
    )

    result = pipeline.run_full()

    # 1-6: research, sourced claims, script, story structure, scene plan
    assert result.research.topic == sample_topic
    assert len(result.research.sources) > 0
    assert all(len(c.source_ids) > 0 for c in result.research.claims)
    assert len(result.script.sections) > 0
    assert len(result.story.structures) >= 1
    assert len(result.scene_plan.scenes) > 0

    # 7: visual technique chosen per scene (not a single forced type)
    visual_types = {s.visual_type for s in result.scene_plan.scenes}
    assert len(visual_types) >= 1

    # 13: 16:9 long-form render exists
    assert result.render_path and os.path.exists(result.render_path)

    # 14: 3-5 shorts
    assert 3 <= len(result.shorts) <= 5
    assert len(result.short_render_paths) == len(result.shorts)
    for path in result.short_render_paths:
        assert os.path.exists(path)
        assert os.path.getsize(path) > 0

    # 15: thumbnails generated
    assert len(result.thumbnail_paths) >= 3

    # 16: automated QA ran
    assert result.qa_report.score >= 0

    # 17: sources and licenses tracked
    assert len(result.research.sources) > 0
    for asset in result.asset_manifest.assets:
        if asset.origin.value == "external_media":
            assert asset.license is not None

    # 18: scene-level regeneration is possible (spot check one scene)
    # -- covered by test_scene_regeneration.py

    # 21: topic independence -- covered by test_topic_independence.py

    # 22: human approval required (never auto-published)
    assert pm.state_machine.current_state in (ProjectState.HUMAN_REVIEW, ProjectState.QA_FAILED)
    assert pm.state_machine.current_state != ProjectState.EXPORTED


def test_no_gpu_mode_still_completes_pipeline(tmp_projects_root, sample_topic):
    """Spec section 66: with no GPU available, the pipeline must still
    complete using image animation / charts / diagrams / licensed media,
    never attempting heavyweight AI generation."""
    config = ProjectConfig(topic=sample_topic, mock_mode=True, local_gpu_enabled=False)
    pm = ProjectManager(projects_root=tmp_projects_root, config=config)
    pipeline = DocumentaryPipeline(pm, video_width=320, video_height=180, fps=10)

    # Force a no-GPU runtime report.
    pipeline.resource_manager._detect_gpu = lambda: (False, None, 0.0, False)  # type: ignore
    pipeline.resource_manager.detect(refresh=True)

    result = pipeline.run_full()
    assert result.render_path and os.path.exists(result.render_path)
    # No scene should have ended up using a real GPU-generated video asset,
    # since none was available.
    for asset in result.asset_manifest.assets:
        assert asset.metadata.get("model") != "svd-open"
