"""License QA: reject any asset with an unknown/missing license."""
from __future__ import annotations

from core.models import Asset, AssetOrigin, QAIssue


def check_licenses(assets: list[Asset]) -> list[QAIssue]:
    issues: list[QAIssue] = []
    for asset in assets:
        if asset.origin == AssetOrigin.EXTERNAL_MEDIA:
            if asset.license is None or not asset.license.license or asset.license.license.lower() == "unknown":
                issues.append(QAIssue(scene_id=asset.scene_id, category="unknown_license", severity="critical",
                                       message=f"External media asset {asset.asset_id} has unknown/missing license."))
    return issues
