from __future__ import annotations

import asyncio
import time

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from electripy.concurrency.rate_limiter import AsyncTokenBucketRateLimiter


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=50)
@given(
    rate=st.floats(min_value=0.1, max_value=100.0),
    capacity_multiplier=st.floats(min_value=0.5, max_value=5.0),
)
async def test_available_tokens_never_exceeds_capacity(
    rate: float, capacity_multiplier: float
) -> None:
    capacity = max(rate * capacity_multiplier, rate)
    limiter = AsyncTokenBucketRateLimiter(rate=rate, capacity=capacity)

    # Drain the bucket entirely.
    await limiter.acquire(tokens=capacity)

    # After waiting, available_tokens must never exceed capacity.
    await asyncio.sleep(0.05)
    assert limiter.available_tokens <= capacity + 1e-6


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=25, deadline=None)
@given(
    rate=st.floats(min_value=1.0, max_value=50.0),
)
async def test_acquire_respects_rate_over_time(rate: float) -> None:
    capacity = rate
    limiter = AsyncTokenBucketRateLimiter(rate=rate, capacity=capacity)

    # Immediately consume full capacity.
    await limiter.acquire(tokens=capacity)
    start = time.monotonic()

    # Next acquire of 1 token should take at least ~1 / rate seconds.
    await limiter.acquire(tokens=1.0)
    elapsed = time.monotonic() - start

    assert elapsed >= (1.0 / rate) * 0.5
