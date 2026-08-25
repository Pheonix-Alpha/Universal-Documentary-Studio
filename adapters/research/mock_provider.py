"""Deterministic, offline ResearchProvider used in MOCK_MODE and tests.

It never calls the network. It synthesizes plausible source/entity
structures from the topic string so the rest of the pipeline (fact
checking, scripting, scene planning...) has real, schema-valid data to
work with without needing live web access or paid research APIs.
"""
from __future__ import annotations

from core.models import ResearchEntity, Source, SourceType

DEPTH_SOURCE_COUNT = {"quick": 3, "standard": 6, "deep": 10}


class MockResearchProvider:
    def search(self, topic: str, depth: str = "standard") -> tuple[list[Source], list[ResearchEntity], list[str]]:
        n_sources = DEPTH_SOURCE_COUNT.get(depth, 6)
        publishers = ["Reuters", "Official Archives", "Academic Journal",
                      "Government Record", "Reference Encyclopedia", "Trade Press",
                      "Industry Report", "Historical Society", "Technical Review",
                      "News Wire"]
        source_types = [SourceType.JOURNALISM, SourceType.OFFICIAL, SourceType.ACADEMIC,
                         SourceType.GOVERNMENT, SourceType.REFERENCE]

        sources = []
        for i in range(n_sources):
            publisher = publishers[i % len(publishers)]
            sources.append(Source(
                title=f"{topic} — coverage from {publisher}",
                url=f"https://example-archive.org/{topic.lower().replace(' ', '-')}/{i}",
                publisher=publisher,
                author=None,
                published_date=None,
                source_type=source_types[i % len(source_types)],
                reliability=0.6 + (0.3 * (i % 3) / 2),
            ))

        entities = [
            ResearchEntity(name=f"{topic} - key figure", entity_type="person",
                            description="Primary individual central to the story."),
            ResearchEntity(name=f"{topic} - organization", entity_type="organization",
                            description="Primary organization involved."),
            ResearchEntity(name="Founding period", entity_type="date",
                            description="Approximate period the story begins."),
            ResearchEntity(name="Primary location", entity_type="location",
                            description="Where the central events took place."),
            ResearchEntity(name="Key statistic", entity_type="statistic",
                            description="A quantitative measure central to the narrative."),
        ]

        context_notes = [
            f"{topic} sits within a broader industry/historical context that shaped its outcome.",
            "Multiple independent sources corroborate the core sequence of events.",
            "Some details remain disputed between competing accounts and should be framed with appropriate caveats.",
        ]

        return sources, entities, context_notes
