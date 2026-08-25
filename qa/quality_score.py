"""Aggregates all QA checker issues into a single score + status (spec section 44).

Score starts at 100 and is deducted per issue by severity. Thresholds are
configurable (config/config.yaml `qa:` block) but default to the values
given in the spec.
"""
from __future__ import annotations

from core.models import QAIssue, QAReport, QAStatus

SEVERITY_PENALTY = {"critical": 20, "warning": 6, "info": 1}


def score_issues(issues: list[QAIssue]) -> float:
    score = 100.0
    for issue in issues:
        score -= SEVERITY_PENALTY.get(issue.severity, 2)
    return max(0.0, score)


def status_for_score(score: float, ready: float = 90, review: float = 80, regenerate: float = 70) -> QAStatus:
    if score >= ready:
        return QAStatus.READY_FOR_REVIEW
    if score >= review:
        return QAStatus.REVIEW_REQUIRED
    if score >= regenerate:
        return QAStatus.REGENERATE
    return QAStatus.REJECT


def build_report(issues: list[QAIssue], ready: float = 90, review: float = 80, regenerate: float = 70) -> QAReport:
    score = score_issues(issues)
    status = status_for_score(score, ready, review, regenerate)
    return QAReport(score=score, status=status, issues=issues)
