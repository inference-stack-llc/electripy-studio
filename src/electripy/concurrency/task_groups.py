"""Async task group and bounded worker-pool utilities.

These helpers provide a thin, typed layer over :class:`asyncio.TaskGroup`
with bounded concurrency. They are intended for higher-level components
such as RAG pipelines or batch LLM calls that need backpressure-aware
fan-out.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from typing import TypeVar

T = TypeVar("T")
U = TypeVar("U")


async def gather_limited(
    coros: Iterable[Awaitable[T]],
    *,
    concurrency: int,
) -> list[T]:
    """Run awaitables with a maximum concurrency limit.

    Args:
        coros: Iterable of awaitables to run.
        concurrency: Maximum number of in-flight tasks.

    Returns:
        List of results in the original order.
    """

    if concurrency <= 0:
        raise ValueError("concurrency must be positive")

    semaphore = asyncio.Semaphore(concurrency)
    coros_list = list(coros)
    results: list[T | None] = [None] * len(coros_list)

    async def _runner(idx: int, aw: Awaitable[T]) -> None:
        async with semaphore:
            results[idx] = await aw

    async with asyncio.TaskGroup() as tg:  # type: ignore[attr-defined]
        for idx, aw in enumerate(coros_list):
            tg.create_task(_runner(idx, aw))

    # At this point all tasks have completed successfully or raised.
    return [r for r in results if r is not None]


async def map_limited(
    fn: Callable[[T], Awaitable[U]],
    items: Iterable[T],
    *,
    concurrency: int,
) -> list[U]:
    """Apply an async function to items with bounded concurrency.

    Args:
        fn: Async function to apply.
        items: Iterable of input items.
        concurrency: Maximum number of concurrent tasks.

    Returns:
        List of results in the order of ``items``.
    """

    return await gather_limited((fn(item) for item in items), concurrency=concurrency)
