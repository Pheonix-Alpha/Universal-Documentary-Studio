"""MediaProvider interface — provider-agnostic licensed/public-domain media search.

Every result MUST carry full license metadata (spec section 26/27).
Unknown-license media must never be returned by a real implementation.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from core.models import LicenseRecord


@dataclass
class MediaSearchResult:
    url: str
    creator: str | None
    license: LicenseRecord
    media_type: str  # image / video
    retrieved_date: str


class MediaProvider(ABC):
    @abstractmethod
    def search(self, query: str, media_type: str = "image") -> list[MediaSearchResult]:
        raise NotImplementedError
