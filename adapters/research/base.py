"""ResearchProvider interface. Concrete providers (web search, Wikipedia,
news APIs, ...) implement `search`; the research agent only depends on
this interface, never on a specific provider.
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from core.models import ResearchEntity, Source


class ResearchProvider(ABC):
    @abstractmethod
    def search(self, topic: str, depth: str = "standard") -> tuple[list[Source], list[ResearchEntity], list[str]]:
        """Return (sources, entities, context_notes) for a topic."""
        raise NotImplementedError
