#!/usr/bin/env bash
# Quick manual smoke test: runs the full pipeline in MOCK_MODE at low
# resolution for a fast sanity check that the whole stack still works.
set -euo pipefail
cd "$(dirname "$0")/.."
PYTHONPATH=. python3 - << 'PYEOF'
from core.models import ProjectConfig
from core.project_manager import ProjectManager
from core.pipeline import DocumentaryPipeline

config = ProjectConfig(
    topic="How a Small Technology Company Became a Global Leader",
    target_duration_minutes=2.0,
    short_count=3,
    mock_mode=True,
)
pm = ProjectManager(projects_root="/tmp/uds_smoke_manual", config=config)
pipeline = DocumentaryPipeline(pm, video_width=640, video_height=360, fps=12,
                                short_width=360, short_height=640, short_fps=12)
result = pipeline.run_full()

print()
print("=== SMOKE TEST RESULT ===")
print("Project:", config.project_id)
print("State:", pm.state_machine.current_state.value)
print("Render:", result.render_path)
print("Shorts:", len(result.short_render_paths))
print("Thumbnails:", len(result.thumbnail_paths))
print("QA:", result.qa_report.score, result.qa_report.status.value)
assert pm.state_machine.current_state.value == "HUMAN_REVIEW"
print("OK")
PYEOF
