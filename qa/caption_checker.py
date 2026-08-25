"""Caption QA: timing coverage/overlap and rough sync against narration duration."""
from __future__ import annotations

from core.models import QAIssue
from engines.captions.caption_engine import CaptionSegment


def check_captions(segments: list[CaptionSegment], total_duration_seconds: float) -> list[QAIssue]:
    issues: list[QAIssue] = []
    if not segments:
        issues.append(QAIssue(category="captions", severity="critical", message="No caption segments generated."))
        return issues

    prev_end = 0.0
    for i, seg in enumerate(segments):
        if seg.start_seconds < prev_end - 0.01:
            issues.append(QAIssue(category="caption_overlap", severity="warning",
                                   message=f"Caption {i} overlaps previous segment."))
        if seg.end_seconds <= seg.start_seconds:
            issues.append(QAIssue(category="caption_timing", severity="critical",
                                   message=f"Caption {i} has non-positive duration."))
        prev_end = seg.end_seconds

    drift = abs(prev_end - total_duration_seconds)
    if total_duration_seconds > 0 and drift / total_duration_seconds > 0.1:
        issues.append(QAIssue(category="caption_sync", severity="warning",
                               message=f"Caption end ({prev_end:.1f}s) drifts from expected duration "
                                       f"({total_duration_seconds:.1f}s)."))
    return issues
