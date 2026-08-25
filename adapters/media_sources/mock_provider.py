"""Mock MediaProvider.

Returns zero results by design in most cases — real licensed-media
search requires actual network access to library APIs (e.g. Wikimedia
Commons, NASA archives). Returning nothing (rather than fabricating
license metadata) forces the VisualAgent to fall back to generated
images/animation, which is the safe default in MOCK_MODE and tests.
"""
from __future__ import annotations

from adapters.media_sources.base import MediaProvider, MediaSearchResult


class MockMediaProvider(MediaProvider):
    def search(self, query: str, media_type: str = "image") -> list[MediaSearchResult]:
        return []
