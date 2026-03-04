"""Ports (Protocols) for the LLM Gateway.

Purpose:
  - Define the minimal capabilities required from LLM providers and
    safety components (redactor and prompt guard).

Guarantees:
  - Business logic depends only on these Protocols.
  - Adapters are free to use any third-party libraries to implement them.

Usage:
  Basic example::

    class MyProvider(SyncLlmPort):
        def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
            ...

    client = LlmGatewaySyncClient(port=MyProvider())
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from .domain import LlmMessage, LlmRequest, LlmResponse


@runtime_checkable
class SyncLlmPort(Protocol):
    """Synchronous LLM provider port."""

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        """Perform a single LLM completion synchronously.

        This is a protocol method; concrete implementations must provide the
        behavior. The ellipsis here intentionally has no body.
        """

        ...


@runtime_checkable
class AsyncLlmPort(Protocol):
    """Asynchronous LLM provider port."""

    async def complete(
        self,
        request: LlmRequest,
        *,
        timeout: float | None = None,
    ) -> LlmResponse:
      """Perform a single LLM completion asynchronously.

      This is a protocol method; concrete implementations must provide the
      behavior. The ellipsis here intentionally has no body.
      """

      ...


@runtime_checkable
class RedactorPort(Protocol):
    """Redactor port for scrubbing PII/PHI before logging or persistence."""

    def redact(self, text: str) -> str:
      """Redact sensitive information in the given text.

      Implementations should return a redacted version of ``text``.
      """

      ...


@dataclass(slots=True)
class GuardResult:
    """Result of a prompt guard assessment.

    Attributes:
      allowed: Whether the request is allowed to proceed.
      score: Optional score between 0 and 1, where lower is more suspicious.
      reasons: Optional human-readable reasons for the decision.
    """

    allowed: bool
    score: float | None = None
    reasons: tuple[str, ...] = ()


@runtime_checkable
class PromptGuardPort(Protocol):
    """Prompt guard port for lightweight safety checks."""

    def assess(self, messages: Sequence[LlmMessage]) -> GuardResult:
        """Assess whether the given messages should be allowed.

        Returns:
          GuardResult indicating allow/deny and optional reasons.
        """

        ...
