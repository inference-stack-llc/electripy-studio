from __future__ import annotations

"""Provider factory helpers for the LLM Gateway.

Purpose:
  - Create ready-to-use sync/async LLM gateway clients from a provider name.
  - Demonstrate how to plug in OpenAI and generic HTTP JSON providers
    (suitable for OpenRouter, Copilot, Grok, or custom gateways).

Guarantees:
  - The factory returns gateway clients that depend only on the port
    Protocols and domain models.

Usage:
  Basic example (OpenAI)::

    from electripy.ai.llm_gateway import build_llm_sync_client

    client = build_llm_sync_client("openai")

  HTTP JSON example (OpenRouter-like)::

    from electripy.ai.llm_gateway import build_llm_sync_client

    client = build_llm_sync_client(
        "http-json",
        base_url="https://api.openrouter.ai",
        path="/v1/chat/completions",
        api_key="sk-...",
    )
"""

from dataclasses import dataclass
from typing import Any

from .adapters import (
    HttpJsonChatAsyncAdapter,
    HttpJsonChatSyncAdapter,
    OpenAiAsyncAdapter,
    OpenAiSyncAdapter,
)
from .config import LlmGatewaySettings
from .services import LlmGatewayAsyncClient, LlmGatewaySyncClient


def _normalise_provider_name(provider: str) -> str:
    return provider.strip().lower()


def build_llm_sync_client(
    provider: str,
    *,
    settings: LlmGatewaySettings | None = None,
    **kwargs: Any,
) -> LlmGatewaySyncClient:
    """Build a synchronous LLM gateway client for the given provider.

    Supported provider names (case-insensitive):

    - "openai": Uses :class:`OpenAiSyncAdapter`.
    - "http-json", "openrouter", "copilot", "grok", "claude-http":
      Use :class:`HttpJsonChatSyncAdapter` with HTTP+JSON via ``httpx``.

    Additional keyword arguments are passed directly to the underlying adapter
    constructor. For example, the HTTP JSON adapter accepts ``base_url``,
    ``path``, and ``api_key``.

    Args:
      provider: Provider identifier.
      settings: Optional gateway settings. If omitted, defaults are used.
      **kwargs: Provider-specific keyword arguments.

    Returns:
      Configured :class:`LlmGatewaySyncClient` instance.
    """

    name = _normalise_provider_name(provider)
    resolved_settings = settings or LlmGatewaySettings()

    if name == "openai":
        port = OpenAiSyncAdapter(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
            organization=kwargs.get("organization"),
        )
        return LlmGatewaySyncClient(port=port, settings=resolved_settings)

    if name in {"http-json", "openrouter", "copilot", "grok", "claude-http"}:
        port = HttpJsonChatSyncAdapter(
            base_url=kwargs["base_url"],
            path=kwargs.get("path", "/v1/chat/completions"),
            api_key=kwargs.get("api_key"),
            auth_header_name=kwargs.get("auth_header_name", "Authorization"),
            client=kwargs.get("client"),
        )
        return LlmGatewaySyncClient(port=port, settings=resolved_settings)

    raise ValueError(f"Unknown provider: {provider!r}")


def build_llm_async_client(
    provider: str,
    *,
    settings: LlmGatewaySettings | None = None,
    **kwargs: Any,
) -> LlmGatewayAsyncClient:
    """Build an asynchronous LLM gateway client for the given provider.

    See :func:`build_llm_sync_client` for supported provider names and
    keyword arguments.
    """

    name = _normalise_provider_name(provider)
    resolved_settings = settings or LlmGatewaySettings()

    if name == "openai":
        port = OpenAiAsyncAdapter(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
            organization=kwargs.get("organization"),
        )
        return LlmGatewayAsyncClient(port=port, settings=resolved_settings)

    if name in {"http-json", "openrouter", "copilot", "grok", "claude-http"}:
        port = HttpJsonChatAsyncAdapter(
            base_url=kwargs["base_url"],
            path=kwargs.get("path", "/v1/chat/completions"),
            api_key=kwargs.get("api_key"),
            auth_header_name=kwargs.get("auth_header_name", "Authorization"),
            client=kwargs.get("client"),
        )
        return LlmGatewayAsyncClient(port=port, settings=resolved_settings)

    raise ValueError(f"Unknown provider: {provider!r}")
