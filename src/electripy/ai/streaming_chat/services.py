"""Services for consuming and normalizing streaming chat output."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterable, AsyncIterator, Iterable, Iterator

from .domain import StreamChunk

__all__ = [
    "async_collect_text",
    "collect_text",
    "iter_text_deltas",
    "with_timeout",
]


def iter_text_deltas(chunks: Iterable[StreamChunk]) -> Iterator[str]:
    """Yield non-empty text deltas from chunks in order."""

    for chunk in chunks:
        if chunk.delta_text:
            yield chunk.delta_text


def collect_text(chunks: Iterable[StreamChunk]) -> str:
    """Collect all text deltas from a sync stream into one string."""

    return "".join(iter_text_deltas(chunks))


async def async_collect_text(chunks: AsyncIterable[StreamChunk]) -> str:
    """Collect all text deltas from an async stream into one string."""

    parts: list[str] = []
    async for chunk in chunks:
        if chunk.delta_text:
            parts.append(chunk.delta_text)
    return "".join(parts)


async def with_timeout(
    chunks: AsyncIterable[StreamChunk],
    *,
    timeout_seconds: float,
) -> AsyncIterator[StreamChunk]:
    """Wrap an async chunk stream with a total timeout.

    Raises:
      TimeoutError: If stream iteration exceeds the timeout.
    """

    iterator = aiter(chunks)
    while True:
        try:
            async with asyncio.timeout(timeout_seconds):
                chunk = await anext(iterator)
        except StopAsyncIteration:
            return
        except TimeoutError as exc:
            raise TimeoutError("stream timed out") from exc
        yield chunk
