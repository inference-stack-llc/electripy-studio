"""Domain models for the LLM Caching Layer."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CacheEntry:
    """A single cached LLM response.

    Attributes:
      key: The deterministic cache key (hex digest).
      response_text: The cached response text.
      model: LLM model identifier used for the original request.
      hit_count: Number of times this entry has been served.
    """

    key: str
    response_text: str
    model: str
    hit_count: int = 0


@dataclass(frozen=True, slots=True)
class CacheStats:
    """Aggregated cache performance statistics.

    Attributes:
      hits: Total cache hits.
      misses: Total cache misses.
      size: Current number of entries in the cache.
      hit_rate: Ratio of hits to total lookups (0.0–1.0).
    """

    hits: int
    misses: int
    size: int

    @property
    def hit_rate(self) -> float:
        """Return the cache hit rate as a float between 0.0 and 1.0."""
        total = self.hits + self.misses
        if total == 0:
            return 0.0
        return self.hits / total
