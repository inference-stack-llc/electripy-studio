"""Ports for provider-agnostic streaming chat adapters."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from typing import Protocol, runtime_checkable

from .domain import StreamChunk


@runtime_checkable
class SyncChatStreamPort(Protocol):
    """Protocol for synchronous streaming chat providers."""

    def stream_text(self, prompt: str) -> Iterator[StreamChunk]:
        """Yield stream chunks for a prompt."""

        ...


@runtime_checkable
class AsyncChatStreamPort(Protocol):
    """Protocol for asynchronous streaming chat providers."""

    async def stream_text(self, prompt: str) -> AsyncIterator[StreamChunk]:
        """Yield stream chunks for a prompt."""

        ...
