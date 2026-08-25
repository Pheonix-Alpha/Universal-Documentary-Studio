from __future__ import annotations

import os

from core.models import ProjectConfig
from core.pipeline import DocumentaryPipeline
from core.project_manager import ProjectManager


def test_regenerating_one_scene_does_not_touch_others(tmp_projects_root, sample_topic):
    config = ProjectConfig(topic=sample_topic, mock_mode=True)
    pm = ProjectManager(projects_root=tmp_projects_root, config=config)
    pipeline = DocumentaryPipeline(pm, video_width=320, video_height=180, fps=10)

    research = pipeline.run_research()
    fact_check = pipeline.run_fact_check(research)
    story = pipeline.run_story(research)
    script = pipeline.run_script(research, story)
    scene_plan = pipeline.run_scenes(script, story)
    asset_manifest = pipeline.run_assets(scene_plan)

    assert len(asset_manifest.assets) >= 2
    target_scene = scene_plan.scenes[1]
    original_path = next(a.file_path for a in asset_manifest.assets if a.scene_id == target_scene.scene_id)
    other_paths_before = {
        a.scene_id: (a.file_path, os.path.getmtime(a.file_path) if a.file_path and os.path.exists(a.file_path) else None)
        for a in asset_manifest.assets if a.scene_id != target_scene.scene_id
    }

    # Regenerate only the target scene via the VisualAgent directly.
    new_asset = pipeline.visual_agent._generate_for_scene(target_scene)

    assert new_asset.scene_id == target_scene.scene_id
    assert os.path.exists(new_asset.file_path)

    # Other scenes' files must be untouched.
    for a in asset_manifest.assets:
        if a.scene_id == target_scene.scene_id:
            continue
        before_path, before_mtime = other_paths_before[a.scene_id]
        if before_path and os.path.exists(before_path):
            assert os.path.getmtime(before_path) == before_mtime
