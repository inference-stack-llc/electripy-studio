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

from __future__ import annotations

from collections.abc import Callable
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


@dataclass(slots=True)
class _ProviderFactories:
    """Registered factories for building gateway clients.

    This allows external code to register custom providers without
    modifying the core factory logic. Factories receive the resolved
    :class:`LlmGatewaySettings` plus any provider-specific keyword
    arguments.
    """

    sync_factory: Callable[[LlmGatewaySettings, dict[str, Any]], LlmGatewaySyncClient] | None
    async_factory: Callable[[LlmGatewaySettings, dict[str, Any]], LlmGatewayAsyncClient] | None


_PROVIDER_REGISTRY: dict[str, _ProviderFactories] = {}


def _normalise_provider_name(provider: str) -> str:
    return provider.strip().lower()


def register_llm_provider(
    name: str,
    *,
    sync_factory: Callable[[LlmGatewaySettings, dict[str, Any]], LlmGatewaySyncClient] | None = None,
    async_factory: Callable[[LlmGatewaySettings, dict[str, Any]], LlmGatewayAsyncClient] | None = None,
) -> None:
    """Register custom LLM provider factories.

    Args:
        name: Provider identifier (case-insensitive).
        sync_factory: Optional factory returning a configured
            :class:`LlmGatewaySyncClient`.
        async_factory: Optional factory returning a configured
            :class:`LlmGatewayAsyncClient`.
    """

    key = _normalise_provider_name(name)
    _PROVIDER_REGISTRY[key] = _ProviderFactories(
        sync_factory=sync_factory,
        async_factory=async_factory,
    )


def list_registered_llm_providers() -> list[str]:
    """Return a sorted list of registered provider names."""

    return sorted(_PROVIDER_REGISTRY.keys())


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

    # Prefer registered custom providers when present.
    registered = _PROVIDER_REGISTRY.get(name)
    if registered is not None and registered.sync_factory is not None:
        return registered.sync_factory(resolved_settings, dict(kwargs))

    if name == "openai":
        openai_port = OpenAiSyncAdapter(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
            organization=kwargs.get("organization"),
        )
        return LlmGatewaySyncClient(port=openai_port, settings=resolved_settings)

    if name in {"http-json", "openrouter", "copilot", "grok", "claude-http"}:
        http_port = HttpJsonChatSyncAdapter(
            base_url=kwargs["base_url"],
            path=kwargs.get("path", "/v1/chat/completions"),
            api_key=kwargs.get("api_key"),
            auth_header_name=kwargs.get("auth_header_name", "Authorization"),
            client=kwargs.get("client"),
        )
        return LlmGatewaySyncClient(port=http_port, settings=resolved_settings)

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

    # Prefer registered custom providers when present.
    registered = _PROVIDER_REGISTRY.get(name)
    if registered is not None and registered.async_factory is not None:
        return registered.async_factory(resolved_settings, dict(kwargs))

    if name == "openai":
        openai_port = OpenAiAsyncAdapter(
            api_key=kwargs.get("api_key"),
            base_url=kwargs.get("base_url"),
            organization=kwargs.get("organization"),
        )
        return LlmGatewayAsyncClient(port=openai_port, settings=resolved_settings)

    if name in {"http-json", "openrouter", "copilot", "grok", "claude-http"}:
        http_port = HttpJsonChatAsyncAdapter(
            base_url=kwargs["base_url"],
            path=kwargs.get("path", "/v1/chat/completions"),
            api_key=kwargs.get("api_key"),
            auth_header_name=kwargs.get("auth_header_name", "Authorization"),
            client=kwargs.get("client"),
        )
        return LlmGatewayAsyncClient(port=http_port, settings=resolved_settings)

    raise ValueError(f"Unknown provider: {provider!r}")
