"""Fake LLM provider for the LLM Gateway recipe.

This module implements small in-memory providers that satisfy the
`SyncLlmPort` and `AsyncLlmPort` Protocols. They are useful for demos
and tests where you do not want to hit real LLM APIs.
"""

from __future__ import annotations

from dataclasses import dataclass

from electripy.ai.llm_gateway import AsyncLlmPort, LlmRequest, LlmResponse, SyncLlmPort


@dataclass
class FakeSyncProvider(SyncLlmPort):
    """Deterministic synchronous fake provider.

    It simply echoes the last user message with a fixed prefix.
    """

    prefix: str = "FakeSync: "

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        # For the recipe we keep this deliberately simple: just echo the
        # content of the last message in the conversation.
        last = request.messages[-1]
        text = f"{self.prefix}{last.content}"
        return LlmResponse(text=text, model=request.model)


@dataclass
class FakeAsyncProvider(AsyncLlmPort):
    """Deterministic asynchronous fake provider."""

    prefix: str = "FakeAsync: "

    async def complete(
        self,
        request: LlmRequest,
        *,
        timeout: float | None = None,
    ) -> LlmResponse:
        last = request.messages[-1]
        text = f"{self.prefix}{last.content}"
        return LlmResponse(text=text, model=request.model)
