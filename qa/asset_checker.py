"""Asset QA: missing files, duplicate visual reuse tracking."""
from __future__ import annotations

from collections import Counter
from pathlib import Path

from core.models import Asset, QAIssue


def check_assets(assets: list[Asset], max_reuse: int = 3) -> list[QAIssue]:
    issues: list[QAIssue] = []

    for asset in assets:
        if asset.file_path and not Path(asset.file_path).exists():
            issues.append(QAIssue(scene_id=asset.scene_id, category="missing_asset", severity="critical",
                                   message=f"Asset file missing on disk: {asset.file_path}"))

    reuse_counter = Counter(a.metadata.get("prompt") or a.file_path for a in assets if a.file_path)
    for key, count in reuse_counter.items():
        if count > max_reuse:
            issues.append(QAIssue(category="visual_reuse", severity="warning",
                                   message=f"Visual reused {count} times (over threshold {max_reuse}): {key}"))

    return issues
