"""Configuration models for the LLM Gateway.

Purpose:
  - Provide typed configuration for retries, timeouts, defaults, and safety hooks.

Guarantees:
  - Default settings are conservative and safe for production use.
  - All fields are fully typed and immutable where appropriate.

Usage:
  Basic example::

    from electripy.ai.llm_gateway import RetryPolicy, LlmGatewaySettings

    settings = LlmGatewaySettings(
        retry_policy=RetryPolicy(max_attempts=4),
        default_model="gpt-4o-mini",
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - import-time only
    from .domain import LlmRequest, LlmResponse
    from .ports import PromptGuardPort, RedactorPort

__all__ = [
    "LlmCallHook",
    "LlmGatewaySettings",
    "LlmRequestHook",
    "LlmResponseHook",
    "RetryPolicy",
]


LlmCallHook = Callable[["LlmRequest", "LlmResponse", float], None]
LlmRequestHook = Callable[["LlmRequest"], "LlmRequest"]
LlmResponseHook = Callable[["LlmRequest", "LlmResponse"], "LlmResponse"]


@dataclass(slots=True)
class RetryPolicy:
    """Retry policy for LLM calls.

    Attributes:
      max_attempts: Maximum number of attempts (including the first).
      initial_backoff_seconds: Initial backoff delay.
      max_backoff_seconds: Maximum delay between attempts.
      total_timeout_seconds: Maximum total wall-clock time across attempts.

    Latency trade-off:
      Higher ``max_attempts`` and ``total_timeout_seconds`` increase robustness but
      also increase worst-case latency. For interactive use, consider keeping
      ``max_attempts`` <= 3 and ``total_timeout_seconds`` <= 30.
    """

    max_attempts: int = 3
    initial_backoff_seconds: float = 0.5
    max_backoff_seconds: float = 8.0
    total_timeout_seconds: float = 30.0


@dataclass(slots=True)
class LlmGatewaySettings:
    """Settings for LlmGatewaySyncClient and LlmGatewayAsyncClient."""

    retry_policy: RetryPolicy = field(default_factory=RetryPolicy)
    default_model: str = "gpt-4o-mini"
    default_temperature: float = 0.2
    default_max_input_chars: int = 8000

    # Safety & logging
    enable_safe_logging: bool = False
    redactor: RedactorPort | None = None
    prompt_guard: PromptGuardPort | None = None

    # Observability hooks
    # Optional callback invoked after each successful LLM call with
    # (request, response, latency_ms). This is intentionally
    # decoupled from any particular telemetry backend so that
    # callers can integrate metrics/tracing however they like.
    on_llm_call: LlmCallHook | None = None

    # Structured output
    structured_output_max_repair_attempts: int = 1

    # Optional transform hooks for request/response policy seams.
    # request_hook can sanitize or reject by raising an exception.
    request_hook: LlmRequestHook | None = None
    # response_hook can sanitize or reject by raising an exception.
    response_hook: LlmResponseHook | None = None
