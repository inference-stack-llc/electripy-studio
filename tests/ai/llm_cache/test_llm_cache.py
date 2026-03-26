"""Tests for the LLM Caching Layer."""

from __future__ import annotations

import pytest

from electripy.ai.llm_cache import (
    CachedLlmPort,
    CacheEntry,
    CacheStats,
    InMemoryCacheBackend,
    SqliteCacheBackend,
)
from electripy.ai.llm_cache.services import compute_cache_key
from electripy.ai.llm_gateway.domain import LlmMessage, LlmRequest, LlmResponse, LlmRole

# ---------------------------------------------------------------------------
# Fake LLM port — tracks call count
# ---------------------------------------------------------------------------


class _FakeLlmPort:
    def __init__(self, text: str = "response") -> None:
        self._text = text
        self.call_count = 0

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        self.call_count += 1
        return LlmResponse(text=self._text, model=request.model)


def _make_request(content: str = "hello", model: str = "gpt-4o-mini") -> LlmRequest:
    return LlmRequest(
        model=model,
        messages=[LlmMessage(role=LlmRole.USER, content=content)],
    )


# ---------------------------------------------------------------------------
# compute_cache_key tests
# ---------------------------------------------------------------------------


class TestComputeCacheKey:
    def test_deterministic(self) -> None:
        req = _make_request("hello")
        assert compute_cache_key(req) == compute_cache_key(req)

    def test_different_content_different_key(self) -> None:
        k1 = compute_cache_key(_make_request("hello"))
        k2 = compute_cache_key(_make_request("world"))
        assert k1 != k2

    def test_different_model_different_key(self) -> None:
        k1 = compute_cache_key(_make_request(model="gpt-4o-mini"))
        k2 = compute_cache_key(_make_request(model="gpt-4"))
        assert k1 != k2

    def test_key_is_hex_sha256(self) -> None:
        key = compute_cache_key(_make_request())
        assert len(key) == 64
        int(key, 16)  # validates hex


# ---------------------------------------------------------------------------
# InMemoryCacheBackend tests
# ---------------------------------------------------------------------------


class TestInMemoryCacheBackend:
    def test_get_miss(self) -> None:
        backend = InMemoryCacheBackend()
        assert backend.get("nonexistent") is None

    def test_put_and_get(self) -> None:
        backend = InMemoryCacheBackend()
        entry = CacheEntry(key="k1", response_text="hello", model="m")
        backend.put(entry)
        result = backend.get("k1")
        assert result is not None
        assert result.response_text == "hello"
        assert result.hit_count == 1

    def test_lru_eviction(self) -> None:
        backend = InMemoryCacheBackend(max_size=2)
        backend.put(CacheEntry(key="a", response_text="1", model="m"))
        backend.put(CacheEntry(key="b", response_text="2", model="m"))
        backend.put(CacheEntry(key="c", response_text="3", model="m"))
        # "a" should be evicted
        assert backend.get("a") is None
        assert backend.get("b") is not None
        assert backend.get("c") is not None

    def test_lru_promotion(self) -> None:
        backend = InMemoryCacheBackend(max_size=2)
        backend.put(CacheEntry(key="a", response_text="1", model="m"))
        backend.put(CacheEntry(key="b", response_text="2", model="m"))
        backend.get("a")  # promote "a" to most-recently-used
        backend.put(CacheEntry(key="c", response_text="3", model="m"))
        # "b" should be evicted, not "a"
        assert backend.get("a") is not None
        assert backend.get("b") is None

    def test_stats_tracking(self) -> None:
        backend = InMemoryCacheBackend()
        backend.put(CacheEntry(key="k", response_text="v", model="m"))
        backend.get("k")  # hit
        backend.get("miss")  # miss

        stats = backend.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 1

    def test_clear(self) -> None:
        backend = InMemoryCacheBackend()
        backend.put(CacheEntry(key="k", response_text="v", model="m"))
        backend.clear()
        assert backend.get("k") is None
        stats = backend.stats()
        assert stats.size == 0
        assert stats.hits == 0


# ---------------------------------------------------------------------------
# SqliteCacheBackend tests
# ---------------------------------------------------------------------------


class TestSqliteCacheBackend:
    def test_get_miss(self) -> None:
        backend = SqliteCacheBackend(db_path=":memory:")
        assert backend.get("nonexistent") is None

    def test_put_and_get(self) -> None:
        backend = SqliteCacheBackend(db_path=":memory:")
        entry = CacheEntry(key="k1", response_text="hello", model="m")
        backend.put(entry)
        result = backend.get("k1")
        assert result is not None
        assert result.response_text == "hello"
        assert result.hit_count == 1

    def test_eviction_at_capacity(self) -> None:
        backend = SqliteCacheBackend(db_path=":memory:", max_size=2)
        backend.put(CacheEntry(key="a", response_text="1", model="m"))
        backend.put(CacheEntry(key="b", response_text="2", model="m"))
        backend.put(CacheEntry(key="c", response_text="3", model="m"))
        # oldest ("a") should be evicted
        assert backend.get("a") is None
        assert backend.get("c") is not None

    def test_stats(self) -> None:
        backend = SqliteCacheBackend(db_path=":memory:")
        backend.put(CacheEntry(key="k", response_text="v", model="m"))
        backend.get("k")
        backend.get("miss")
        stats = backend.stats()
        assert stats.hits == 1
        assert stats.misses == 1
        assert stats.size == 1

    def test_clear(self) -> None:
        backend = SqliteCacheBackend(db_path=":memory:")
        backend.put(CacheEntry(key="k", response_text="v", model="m"))
        backend.clear()
        assert backend.stats().size == 0


# ---------------------------------------------------------------------------
# CacheStats tests
# ---------------------------------------------------------------------------


class TestCacheStats:
    def test_hit_rate_no_lookups(self) -> None:
        stats = CacheStats(hits=0, misses=0, size=0)
        assert stats.hit_rate == 0.0

    def test_hit_rate_all_hits(self) -> None:
        stats = CacheStats(hits=10, misses=0, size=5)
        assert stats.hit_rate == 1.0

    def test_hit_rate_mixed(self) -> None:
        stats = CacheStats(hits=3, misses=7, size=3)
        assert stats.hit_rate == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# CachedLlmPort integration tests
# ---------------------------------------------------------------------------


class TestCachedLlmPort:
    def test_cache_miss_calls_inner(self) -> None:
        port = _FakeLlmPort("hello")
        backend = InMemoryCacheBackend()
        cached = CachedLlmPort(inner=port, backend=backend)

        response = cached.complete(_make_request())
        assert response.text == "hello"
        assert port.call_count == 1

    def test_cache_hit_skips_inner(self) -> None:
        port = _FakeLlmPort("hello")
        backend = InMemoryCacheBackend()
        cached = CachedLlmPort(inner=port, backend=backend)

        req = _make_request()
        cached.complete(req)  # miss
        cached.complete(req)  # hit

        assert port.call_count == 1

    def test_cached_response_matches_original(self) -> None:
        port = _FakeLlmPort("result")
        backend = InMemoryCacheBackend()
        cached = CachedLlmPort(inner=port, backend=backend)

        req = _make_request()
        r1 = cached.complete(req)
        r2 = cached.complete(req)

        assert r1.text == r2.text == "result"

    def test_different_requests_not_cached(self) -> None:
        port = _FakeLlmPort("resp")
        backend = InMemoryCacheBackend()
        cached = CachedLlmPort(inner=port, backend=backend)

        cached.complete(_make_request("a"))
        cached.complete(_make_request("b"))

        assert port.call_count == 2

    def test_stats_reflect_hits_and_misses(self) -> None:
        port = _FakeLlmPort("resp")
        backend = InMemoryCacheBackend()
        cached = CachedLlmPort(inner=port, backend=backend)

        req = _make_request()
        cached.complete(req)  # miss
        cached.complete(req)  # hit
        cached.complete(req)  # hit

        stats = backend.stats()
        assert stats.hits == 2
        assert stats.misses == 1
