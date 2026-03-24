import asyncio
from collections.abc import Awaitable

import pytest

from electripy.concurrency.task_groups import gather_limited, map_limited


async def _recording_coro(value: int, seen: list[int], delay: float = 0.01) -> int:
    await asyncio.sleep(delay)
    seen.append(value)
    return value * 2


@pytest.mark.asyncio
async def test_gather_limited_preserves_order_and_bounds_concurrency() -> None:
    max_concurrency = 3
    in_flight = 0
    peak_in_flight = 0

    async def wrapped(v: int) -> int:
        nonlocal in_flight, peak_in_flight
        in_flight += 1
        peak_in_flight = max(peak_in_flight, in_flight)
        try:
            await asyncio.sleep(0.01)
            return v
        finally:
            in_flight -= 1

    coros: list[Awaitable[int]] = [wrapped(i) for i in range(10)]
    results = await gather_limited(coros, concurrency=max_concurrency)

    assert results == list(range(10))
    assert 1 <= peak_in_flight <= max_concurrency


@pytest.mark.asyncio
async def test_map_limited_applies_function() -> None:
    seen: list[int] = []
    items = [1, 2, 3, 4]

    async def fn(x: int) -> int:
        return await _recording_coro(x, seen)

    results = await map_limited(fn, items, concurrency=2)

    assert results == [2, 4, 6, 8]
    assert sorted(seen) == items
