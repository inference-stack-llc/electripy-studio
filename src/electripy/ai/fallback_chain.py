"""Fallback chain — try LLM providers in order until one succeeds.

Purpose:
  - Declaratively specify a ranked list of ``SyncLlmPort`` providers.
  - On failure the chain advances to the next provider automatically.
  - Surface which provider ultimately handled the request.

Guarantees:
  - No provider-specific code — works with any ``SyncLlmPort``.
  - Exceptions from non-final providers are swallowed; the last provider's
    exception propagates if all fail.
  - Thread-safe and stateless — safe for concurrent use.

Usage::

    from electripy.ai.fallback_chain import FallbackChainPort

    chain = FallbackChainPort(
        providers=[openai_adapter, anthropic_adapter, ollama_adapter],
    )
    response = chain.complete(request)  # tries in order
    print(response.metadata["_fallback_provider_index"])  # 0, 1, or 2
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

from electripy.ai.llm_gateway.domain import LlmRequest, LlmResponse
from electripy.ai.llm_gateway.ports import SyncLlmPort

__all__ = [
    "FallbackChainPort",
]

_logger = logging.getLogger(__name__)


class FallbackChainPort(SyncLlmPort):
    """Wraps N ``SyncLlmPort`` providers and tries each in order.

    On the first success the response is returned immediately with
    ``metadata["_fallback_provider_index"]`` set to the winning index.
    If a provider raises any exception the chain moves to the next one.
    The last provider's exception is always re-raised.
    """

    __slots__ = ("_providers",)

    def __init__(self, *, providers: Sequence[SyncLlmPort]) -> None:
        if not providers:
            raise ValueError("FallbackChainPort requires at least one provider.")
        self._providers = tuple(providers)

    def complete(
        self,
        request: LlmRequest,
        *,
        timeout: float | None = None,
    ) -> LlmResponse:
        last_exc: Exception | None = None
        for idx, provider in enumerate(self._providers):
            try:
                response = provider.complete(request, timeout=timeout)
                response.metadata["_fallback_provider_index"] = idx
                return response
            except Exception as exc:  # noqa: BLE001
                _logger.debug(
                    "Fallback chain: provider %d failed (%s), trying next.",
                    idx,
                    type(exc).__name__,
                )
                last_exc = exc
        # All failed — re-raise the last provider's exception.
        raise last_exc  # type: ignore[misc]
