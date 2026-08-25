"""Pipeline-facing cache facade.

Agents import from `core.cache` rather than reaching into `storage`
directly, keeping the storage backend swappable.
"""
from __future__ import annotations

from storage.cache_store import CacheStore, make_cache_key

__all__ = ["CacheStore", "make_cache_key"]
