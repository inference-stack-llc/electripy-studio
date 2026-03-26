"""Provider adapters — OpenAI, Anthropic, and Ollama implementations of SyncLlmPort.

Purpose:
  - Canonical home for all provider-specific LLM adapters.
  - Provide OpenAI adapter for GPT models via the Chat Completions API.
  - Provide Anthropic adapter for Claude models via the Messages API.
  - Provide zero-config Ollama adapter for local/self-hosted models.

Guarantees:
  - Lazy-import of ``openai``, ``anthropic``, and ``httpx`` — no hard dependency
    at import time.
  - Provider-specific exceptions are mapped to domain exceptions.
  - Adapters accept a test-injected client for deterministic offline tests.

Usage::

    from electripy.ai.llm_gateway.provider_adapters import (
        OpenAiSyncAdapter,
        AnthropicSyncAdapter,
        OllamaSyncAdapter,
    )

    # OpenAI (requires `pip install openai`)
    adapter = OpenAiSyncAdapter(api_key="sk-...")
    response = adapter.complete(request)

    # Anthropic (requires `pip install anthropic`)
    adapter = AnthropicSyncAdapter(api_key="sk-ant-...")
    response = adapter.complete(request)

    # Ollama (requires a running Ollama server)
    adapter = OllamaSyncAdapter(base_url="http://localhost:11434")
    response = adapter.complete(request)
"""

from __future__ import annotations

import importlib
from typing import Any

import httpx

from .adapters import OpenAiSyncAdapter
from .domain import LlmRequest, LlmResponse
from .errors import RateLimitedError, TransientLlmError
from .ports import SyncLlmPort

__all__ = [
    "OpenAiSyncAdapter",
    "AnthropicSyncAdapter",
    "OllamaSyncAdapter",
]


class AnthropicSyncAdapter(SyncLlmPort):
    """Synchronous Anthropic adapter implementing SyncLlmPort.

    Translates the provider-agnostic ``LlmRequest`` to the Anthropic
    Messages API format and maps responses back.

    Notes:
      - Requires the official Anthropic Python SDK (``anthropic>=0.18``).
      - System messages are extracted and sent via the ``system`` parameter.
      - Only the minimal ``messages.create`` surface is used.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ) -> None:
        if client is not None:
            self._client = client
            return
        try:
            anthropic_module = importlib.import_module("anthropic")
        except ImportError as exc:  # pragma: no cover — import-time dependency
            raise ImportError(
                "AnthropicSyncAdapter requires the `anthropic` package. "
                "Install with `pip install anthropic`.",
            ) from exc

        kwargs: dict[str, Any] = {}
        if api_key is not None:
            kwargs["api_key"] = api_key
        if base_url is not None:
            kwargs["base_url"] = base_url
        self._client = anthropic_module.Anthropic(**kwargs)

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        """Perform a completion via the Anthropic Messages API.

        System messages are extracted from the message list and sent
        as the ``system`` parameter.  All other messages are forwarded
        as user/assistant turns.

        Args:
          request: Normalised LLM request.
          timeout: Optional per-call timeout in seconds.

        Returns:
          An :class:`LlmResponse` with the assistant text.
        """
        system_parts: list[str] = []
        api_messages: list[dict[str, str]] = []

        for msg in request.messages:
            if msg.role.value == "system":
                system_parts.append(msg.content)
            else:
                api_messages.append({"role": msg.role.value, "content": msg.content})

        kwargs: dict[str, Any] = {
            "model": request.model,
            "messages": api_messages,
            "temperature": request.temperature,
            "max_tokens": request.max_output_tokens or 1024,
        }
        if system_parts:
            kwargs["system"] = "\n\n".join(system_parts)
        if timeout is not None:
            kwargs["timeout"] = timeout

        try:
            response = self._client.messages.create(**kwargs)
        except Exception as exc:
            _map_anthropic_exception(exc)
            raise  # pragma: no cover — _map always raises

        text = ""
        if hasattr(response, "content") and response.content:
            block = response.content[0]
            text = getattr(block, "text", "")

        return LlmResponse(
            text=text,
            model=getattr(response, "model", request.model),
            usage_total_tokens=_extract_anthropic_usage(response),
            finish_reason=getattr(response, "stop_reason", None),
            request_id=getattr(response, "id", None),
        )


class OllamaSyncAdapter(SyncLlmPort):
    """Synchronous Ollama adapter implementing SyncLlmPort.

    Communicates with a running Ollama server via its HTTP chat API.
    No external SDK is required — uses :mod:`httpx` directly.

    Notes:
      - Ollama must be running locally (default ``http://localhost:11434``).
      - Only the ``/api/chat`` endpoint is used.
    """

    def __init__(
        self,
        *,
        base_url: str = "http://localhost:11434",
        client: httpx.Client | None = None,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=120.0)

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        """Perform a completion via the Ollama chat API.

        Args:
          request: Normalised LLM request.
          timeout: Optional per-call timeout in seconds.

        Returns:
          An :class:`LlmResponse` with the assistant text.
        """
        messages = [{"role": m.role.value, "content": m.content} for m in request.messages]

        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": request.temperature,
            },
        }
        if request.max_output_tokens is not None:
            payload["options"]["num_predict"] = request.max_output_tokens

        try:
            resp = self._client.post(
                f"{self._base_url}/api/chat",
                json=payload,
                timeout=timeout,
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            _map_http_exception(exc)
            raise  # pragma: no cover
        except httpx.HTTPError as exc:
            raise TransientLlmError(
                message=f"Ollama connection error: {exc}", status_code=None
            ) from None

        data = resp.json()
        text = data.get("message", {}).get("content", "")
        total_tokens = None
        if "eval_count" in data and "prompt_eval_count" in data:
            total_tokens = data["prompt_eval_count"] + data["eval_count"]

        return LlmResponse(
            text=text,
            model=data.get("model", request.model),
            usage_total_tokens=total_tokens,
            finish_reason=data.get("done_reason"),
        )


# ---------------------------------------------------------------------------
# Exception mapping helpers
# ---------------------------------------------------------------------------


def _map_anthropic_exception(exc: Exception) -> None:
    """Map an Anthropic exception to a domain error and raise it."""
    status_code: int | None = getattr(exc, "status_code", None)

    if status_code == 429:
        raise RateLimitedError(
            message="Anthropic rate limit encountered",
            status_code=429,
        ) from None

    if status_code is not None and status_code >= 500:
        raise TransientLlmError(
            message="Anthropic transient error",
            status_code=status_code,
        ) from None

    raise TransientLlmError(message=str(exc), status_code=status_code) from None


def _map_http_exception(exc: httpx.HTTPStatusError) -> None:
    """Map an HTTP status error to a domain error and raise it."""
    status = exc.response.status_code

    if status == 429:
        raise RateLimitedError(
            message="Provider rate limit encountered",
            status_code=429,
        ) from None

    if status >= 500:
        raise TransientLlmError(
            message=f"Provider error (HTTP {status})",
            status_code=status,
        ) from None

    raise TransientLlmError(
        message=f"Provider error (HTTP {status}): {exc}",
        status_code=status,
    ) from None


def _extract_anthropic_usage(response: Any) -> int | None:
    """Extract total token usage from an Anthropic response."""
    usage = getattr(response, "usage", None)
    if usage is None:
        return None
    input_tokens = getattr(usage, "input_tokens", 0) or 0
    output_tokens = getattr(usage, "output_tokens", 0) or 0
    return input_tokens + output_tokens
