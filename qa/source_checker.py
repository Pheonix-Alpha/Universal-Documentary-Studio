"""Source QA: every important claim must be backed by at least one source."""
from __future__ import annotations

from core.models import Claim, QAIssue, Source


def check_sources(claims: list[Claim], sources: list[Source], min_sources_for_major_claim: int = 1) -> list[QAIssue]:
    issues: list[QAIssue] = []
    source_ids = {s.source_id for s in sources}

    for claim in claims:
        valid_refs = [sid for sid in claim.source_ids if sid in source_ids]
        if not valid_refs:
            issues.append(QAIssue(category="missing_source", severity="critical",
                                   message=f"Claim '{claim.text[:60]}...' has no valid source reference."))
        elif len(valid_refs) < min_sources_for_major_claim and claim.confidence < 0.6:
            issues.append(QAIssue(category="weak_sourcing", severity="warning",
                                   message=f"Low-confidence claim backed by only {len(valid_refs)} source(s)."))
    return issues
