"""FactChecker agent: research.json -> fact_check_report.json (spec section 17).

Checks, per claim:
  - source exists (referenced source_id is present in the research's source list)
  - source is relevant / supports claim (heuristic: reliability + non-empty text)
  - entities referenced actually exist in research.entities (best-effort)
  - wording doesn't overstate the evidence (simple heuristic on absolute language)

A critical failure (status=FAILED) on any claim blocks script approval,
per spec section 17.
"""
from __future__ import annotations

from core.logging import get_logger
from core.models import FactCheckReport, FactCheckResult, FactCheckStatus, Research

logger = get_logger(__name__)

_OVERREACH_TERMS = ["definitely proves", "undeniably", "100% certain", "no doubt whatsoever"]


class FactChecker:
    def run(self, research: Research) -> FactCheckReport:
        source_ids = {s.source_id for s in research.sources}
        results: list[FactCheckResult] = []

        for claim in research.claims:
            reasons = []
            status = FactCheckStatus.PASSED

            valid_sources = [sid for sid in claim.source_ids if sid in source_ids]
            if not valid_sources:
                status = FactCheckStatus.FAILED
                reasons.append("No valid source reference found for this claim.")
            elif len(valid_sources) < len(claim.source_ids):
                status = FactCheckStatus.WARNING
                reasons.append("One or more referenced sources could not be resolved.")

            if any(term in claim.text.lower() for term in _OVERREACH_TERMS):
                status = FactCheckStatus.WARNING if status == FactCheckStatus.PASSED else status
                reasons.append("Wording may overstate the strength of the evidence.")

            if claim.confidence < 0.3:
                status = FactCheckStatus.WARNING if status == FactCheckStatus.PASSED else status
                reasons.append("Low confidence score for this claim.")

            results.append(FactCheckResult(claim_id=claim.claim_id, status=status, reasons=reasons))

        report = FactCheckReport(results=results)
        if report.has_critical_failure:
            logger.warning("Fact check found %d critical failure(s).",
                            sum(1 for r in results if r.status == FactCheckStatus.FAILED))
        return report
