# LLM Caching Layer

The **LLM Caching Layer** reduces cost and latency by storing LLM
responses keyed on a deterministic fingerprint of the request.  Wrap any
`SyncLlmPort` in `CachedLlmPort` to get transparent caching with zero
changes to calling code.

## When to use it

- You call the same (or similar) prompts repeatedly — development loops,
  regression tests, batch processing.
- You want to **cut API spend** without changing your prompts or models.
- You need a local, offline fallback for previously-seen queries.

## Core concepts

- **Domain models**:
    - `CacheEntry` — frozen record of a cached response plus the
      original request hash and timestamp.
    - `CacheStats` — frozen hit/miss counter with a `hit_rate` property.
- **Ports**:
    - `CacheBackendPort` — protocol that any backend must implement
      (`get`, `put`, `stats`, `clear`).
- **Adapters (built-in backends)**:
    - `InMemoryCacheBackend` — thread-safe, LRU eviction, zero config.
    - `SqliteCacheBackend` — process-safe, WAL mode, persistent across
      restarts.
- **Services**:
    - `CachedLlmPort` — decorator that implements `SyncLlmPort`, checks
      the cache first, falls back to the inner port on miss.

## Basic example: in-memory cache

```python
from electripy.ai.llm_cache import CachedLlmPort, InMemoryCacheBackend
from electripy.ai.llm_gateway import build_llm_sync_client

inner = build_llm_sync_client("openai")
cache = InMemoryCacheBackend(max_size=500)
client = CachedLlmPort(inner=inner, backend=cache)

# First call — miss, hits the real API
response = client.complete(request)

# Second call — hit, returns instantly from cache
response = client.complete(request)

print(cache.stats())          # CacheStats(hits=1, misses=1)
print(cache.stats().hit_rate)  # 0.5
```

## SQLite persistent cache

For caches that survive process restarts, use the SQLite backend:

```python
from electripy.ai.llm_cache import CachedLlmPort, SqliteCacheBackend

cache = SqliteCacheBackend(db_path="llm_cache.db")
client = CachedLlmPort(inner=inner, backend=cache)
```

SQLite uses WAL mode for concurrent read performance and is safe across
multiple processes.

## Cache key mechanics

Cache keys are SHA-256 hashes computed from:

- Model name
- Temperature
- Message content (role + text for each message)

This means identical prompts with the same model and temperature always
produce the same key, regardless of other metadata.

## Custom backends

Implement `CacheBackendPort` to plug in Redis, DynamoDB, or any other
store:

```python
from electripy.ai.llm_cache import CacheBackendPort, CacheEntry, CacheStats


class RedisCacheBackend(CacheBackendPort):
    def get(self, key: str) -> CacheEntry | None: ...
    def put(self, key: str, entry: CacheEntry) -> None: ...
    def stats(self) -> CacheStats: ...
    def clear(self) -> None: ...
```

## Integration with other components

- **Structured Output** — cache structured extraction calls by wrapping
  the inner port before passing to `StructuredOutputExtractor`.
- **Replay Tape** — combine caching with recording: cache in prod, record
  in staging for offline test suites.
