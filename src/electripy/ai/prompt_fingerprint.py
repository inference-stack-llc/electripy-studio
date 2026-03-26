"""Prompt Fingerprint — deterministic hash of an LLM request for versioning and dedup.

Purpose:
  - Produce a stable, deterministic identifier for any LLM request.
  - Useful for caching, deduplication, A/B bucketing, audit trails.
  - Works with any ``LlmRequest`` — no provider-specific logic.

Guarantees:
  - Same (model, temperature, messages) → same fingerprint, always.
  - SHA-256 based — collision-resistant and URL-safe.
  - Pure function with no side effects.

Usage::

    from electripy.ai.prompt_fingerprint import prompt_fingerprint

    fp = prompt_fingerprint(request)
    print(fp)  # "a3f2c8..."  (64-char hex digest)
"""

from __future__ import annotations

import hashlib
import json

from electripy.ai.llm_gateway.domain import LlmRequest

__all__ = [
    "prompt_fingerprint",
    "prompt_fingerprint_short",
]


def prompt_fingerprint(request: LlmRequest) -> str:
    """Return a SHA-256 hex digest uniquely identifying the request.

    The hash is computed from:

    - ``model``
    - ``temperature``
    - ``messages`` (role + content for each)

    This is identical to the cache key algorithm used by
    ``electripy.ai.llm_cache`` but exposed as a standalone utility.
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


def prompt_fingerprint_short(request: LlmRequest, *, length: int = 12) -> str:
    """Return a truncated fingerprint for display / logging.

    Args:
        request: The LLM request to fingerprint.
        length: Number of hex characters to return (default 12).

    Returns:
        First *length* characters of the full SHA-256 digest.
    """
    return prompt_fingerprint(request)[:length]
