"""LLM Caching Layer — reduce cost and latency with pluggable response caches.

Purpose:
  - Cache LLM completions by exact request fingerprint.
  - Pluggable backends: in-memory (default), SQLite, or user-supplied.
  - Track hit/miss statistics and estimated cost savings.

Guarantees:
  - Cache keys are deterministic (model + messages + temperature).
  - No provider-specific types; wraps any ``SyncLlmPort``.
  - Thread-safe in-memory backend; SQLite backend is process-safe.
  - All domain models are immutable.

Usage::

    from electripy.ai.llm_cache import CachedLlmPort, InMemoryCacheBackend

    cache = InMemoryCacheBackend(max_size=1000)
    cached_port = CachedLlmPort(inner=my_llm_port, backend=cache)
    response = cached_port.complete(request)  # cached on second call
    print(cache.stats())
"""

from __future__ import annotations

from .adapters import InMemoryCacheBackend, SqliteCacheBackend
from .domain import CacheEntry, CacheStats
from .ports import CacheBackendPort
from .services import CachedLlmPort

__all__ = [
    # Domain models
    "CacheEntry",
    "CacheStats",
    # Ports
    "CacheBackendPort",
    # Adapters
    "InMemoryCacheBackend",
    "SqliteCacheBackend",
    # Services
    "CachedLlmPort",
]
