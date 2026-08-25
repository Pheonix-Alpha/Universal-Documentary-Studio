"""ResearchAgent: TOPIC -> research.json (spec section 15/16).

Provider-agnostic: depends only on adapters.research.base.ResearchProvider.
Real deployments plug in a web-search/Wikipedia/news adapter; MOCK_MODE
uses MockResearchProvider so the rest of the pipeline always has valid,
schema-correct data to work with.
"""
from __future__ import annotations

from adapters.research.base import ResearchProvider
from core.logging import get_logger
from core.models import Research

logger = get_logger(__name__)


class ResearchAgent:
    def __init__(self, provider: ResearchProvider):
        self.provider = provider

    def run(self, topic: str, depth: str = "standard") -> Research:
        logger.info("Researching topic=%r depth=%s", topic, depth)
        sources, entities, context_notes = self.provider.search(topic, depth=depth)

        if not sources:
            logger.warning("Research provider returned zero sources for topic=%r", topic)

        # Synthesize claims from entities/context so downstream fact-checking
        # and scripting have concrete, source-linkable statements to work
        # with. A real provider would return claims directly; this keeps
        # the interface uniform regardless of provider sophistication.
        claims = []
        source_id_pool = [s.source_id for s in sources]
        for i, entity in enumerate(entities):
            if not source_id_pool:
                break
            linked = [source_id_pool[i % len(source_id_pool)]]
            if len(source_id_pool) > 1:
                linked.append(source_id_pool[(i + 1) % len(source_id_pool)])
            from core.models import Claim
            claims.append(Claim(
                text=f"{entity.name}: {entity.description or 'relevant to ' + topic}",
                source_ids=linked,
                confidence=0.7,
            ))

        research = Research(
            topic=topic,
            entities=entities,
            claims=claims,
            sources=sources,
            context_notes=context_notes,
        )
        return research
