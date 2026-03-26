"""Ports for the LLM Caching Layer."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .domain import CacheEntry, CacheStats


@runtime_checkable
class CacheBackendPort(Protocol):
    """Abstract cache backend for LLM responses.

    Implementations are responsible for storage, eviction, and statistics.
    Thread-safety guarantees depend on the concrete backend.
    """

    def get(self, key: str) -> CacheEntry | None:
        """Retrieve a cached entry by key.

        Args:
          key: The deterministic cache key.

        Returns:
          The cached entry if found, else ``None``.
        """
        ...

    def put(self, entry: CacheEntry) -> None:
        """Store a cache entry.

        If the cache is at capacity, the backend may evict entries
        according to its eviction policy.

        Args:
          entry: The cache entry to store.
        """
        ...

    def stats(self) -> CacheStats:
        """Return aggregated cache statistics.

        Returns:
          A snapshot of current hit/miss/size counters.
        """
        ...

    def clear(self) -> None:
        """Remove all entries from the cache."""
        ...
