"""QAAgent: runs every checker in qa/ and aggregates into a QAReport
(spec section 43/44). Never auto-publishes -- callers gate on
QAStatus.READY_FOR_REVIEW before moving to HUMAN_REVIEW.
"""
from __future__ import annotations

from core.logging import get_logger
from core.models import AssetManifest, QAIssue, QAReport, Research, ScenePlan
from engines.captions.caption_engine import CaptionSegment
from qa.asset_checker import check_assets
from qa.audio_checker import check_audio
from qa.caption_checker import check_captions
from qa.license_checker import check_licenses
from qa.quality_score import build_report
from qa.source_checker import check_sources
from qa.video_checker import check_video

logger = get_logger(__name__)


class QAAgent:
    def __init__(self, ready_threshold: float = 90, review_threshold: float = 80, regenerate_threshold: float = 70):
        self.ready_threshold = ready_threshold
        self.review_threshold = review_threshold
        self.regenerate_threshold = regenerate_threshold

    def run(
        self,
        research: Research,
        asset_manifest: AssetManifest,
        render_path: str | None,
        expected_width: int,
        expected_height: int,
        expected_fps: int,
        caption_segments: list[CaptionSegment],
        total_duration_seconds: float,
    ) -> QAReport:
        issues: list[QAIssue] = []

        issues += check_sources(research.claims, research.sources)
        issues += check_assets(asset_manifest.assets)
        issues += check_licenses(asset_manifest.assets)
        issues += check_captions(caption_segments, total_duration_seconds)

        if render_path:
            issues += check_video(render_path, expected_width, expected_height, expected_fps)

        report = build_report(issues, self.ready_threshold, self.review_threshold, self.regenerate_threshold)
        logger.info("QA complete: score=%.1f status=%s issues=%d", report.score, report.status.value, len(issues))
        return report
