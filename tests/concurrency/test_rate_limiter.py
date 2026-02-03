"""Tests for concurrency.rate_limiter module."""

import asyncio
import time

import pytest

from electripy.concurrency.rate_limiter import AsyncTokenBucketRateLimiter


@pytest.mark.asyncio
async def test_rate_limiter_basic() -> None:
    """Test basic rate limiter operation."""
    limiter = AsyncTokenBucketRateLimiter(rate=10, capacity=10)
    
    # Should acquire immediately
    await limiter.acquire(1)
    assert limiter.available_tokens < 10


@pytest.mark.asyncio
async def test_rate_limiter_context_manager() -> None:
    """Test rate limiter as context manager."""
    limiter = AsyncTokenBucketRateLimiter(rate=10, capacity=10)
    
    async with limiter:
        pass  # Token acquired and released
    
    # Should have tokens available
    assert limiter.available_tokens > 0


@pytest.mark.asyncio
async def test_rate_limiter_exceeds_capacity() -> None:
    """Test rate limiter raises error when tokens exceed capacity."""
    limiter = AsyncTokenBucketRateLimiter(rate=5, capacity=5)
    
    with pytest.raises(ValueError):
        await limiter.acquire(10)


@pytest.mark.asyncio
async def test_rate_limiter_token_replenishment() -> None:
    """Test that tokens replenish over time."""
    limiter = AsyncTokenBucketRateLimiter(rate=10, capacity=10)
    
    # Consume all tokens
    await limiter.acquire(10)
    assert limiter.available_tokens < 1
    
    # Wait for replenishment
    await asyncio.sleep(0.5)
    assert limiter.available_tokens >= 4


@pytest.mark.asyncio
async def test_rate_limiter_waits_when_empty() -> None:
    """Test rate limiter waits when bucket is empty."""
    limiter = AsyncTokenBucketRateLimiter(rate=10, capacity=5)
    
    # Consume all tokens
    await limiter.acquire(5)
    
    # Next acquire should wait
    start = time.monotonic()
    await limiter.acquire(1)
    elapsed = time.monotonic() - start
    
    # Should have waited at least 0.1 seconds (1 token / 10 rate)
    assert elapsed >= 0.08  # Allow some tolerance
