from __future__ import annotations

import pytest

from core.models import ProjectConfig
from core.pipeline import DocumentaryPipeline
from core.project_manager import ProjectManager

TOPICS = [
    "The Rise and Fall of a Regional Airline",
    "How CRISPR Gene Editing Was Discovered",
    "The Chernobyl Nuclear Disaster",
    "The Invention of the Transistor",
    "A CEO's Decision That Nearly Sank the Company",
    "The Apollo 13 Mission Investigation",
]


@pytest.mark.parametrize("topic", TOPICS)
def test_pipeline_completes_for_varied_topic_categories(tmp_projects_root, topic):
    """The system must remain topic-agnostic (spec section 67/21):
    no category should be hard-coded as a special case."""
    config = ProjectConfig(topic=topic, mock_mode=True, short_count=3)
    pm = ProjectManager(projects_root=tmp_projects_root, config=config)
    pipeline = DocumentaryPipeline(pm, video_width=320, video_height=180, fps=10,
                                    short_width=180, short_height=320, short_fps=10)
    result = pipeline.run_full()

    assert result.research.topic == topic
    assert len(result.scene_plan.scenes) > 0
    assert result.render_path
    import os
    assert os.path.exists(result.render_path)
    assert len(result.shorts) >= 3
