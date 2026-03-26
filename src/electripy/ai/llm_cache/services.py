"""Services for the LLM Caching Layer."""

from __future__ import annotations

import hashlib
import json

from electripy.ai.llm_gateway.domain import LlmRequest, LlmResponse
from electripy.ai.llm_gateway.ports import SyncLlmPort

from .domain import CacheEntry
from .ports import CacheBackendPort


def compute_cache_key(request: LlmRequest) -> str:
    """Compute a deterministic SHA-256 cache key from a request.

    The key is derived from model, temperature, and the full message
    sequence.  Two identical requests always produce the same key.

    Args:
      request: The LLM request to hash.

    Returns:
      Hex-encoded SHA-256 digest.
    """
    payload = json.dumps(
        {
            "model": request.model,
            "temperature": request.temperature,
            "messages": [{"role": m.role.value, "content": m.content} for m in request.messages],
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


class CachedLlmPort:
    """Caching wrapper around an existing LLM port.

    Transparently intercepts ``complete()`` calls, checking the cache
    backend before forwarding to the inner port.  Responses from the
    inner port are stored for future lookups.

    This class itself implements the ``SyncLlmPort`` protocol so it can
    be used anywhere a port is expected.

    Args:
      inner: The underlying LLM port to delegate cache misses to.
      backend: The cache backend for storage and retrieval.
    """

    __slots__ = ("_inner", "_backend")

    def __init__(
        self,
        *,
        inner: SyncLlmPort,
        backend: CacheBackendPort,
    ) -> None:
        self._inner = inner
        self._backend = backend

    def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
        """Complete a request, returning a cached response if available.

        Args:
          request: The LLM request.
          timeout: Optional timeout forwarded to the inner port.

        Returns:
          An ``LlmResponse`` — either from cache or from the inner port.
        """
        key = compute_cache_key(request)
        cached = self._backend.get(key)
        if cached is not None:
            return LlmResponse(text=cached.response_text, model=cached.model)

        response = self._inner.complete(request, timeout=timeout)
        entry = CacheEntry(
            key=key,
            response_text=response.text,
            model=response.model or request.model,
        )
        self._backend.put(entry)
        return response
