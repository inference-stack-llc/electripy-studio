"""Streaming chat primitives for sync and async text generation.

Purpose:
  - Provide provider-agnostic stream chunk models and helper utilities.
  - Keep stream handling deterministic and testable across adapters.

Guarantees:
  - Public APIs are typed and backend-agnostic.
  - Sync and async helpers behave consistently.
"""

from __future__ import annotations

from .domain import StreamChunk
from .ports import AsyncChatStreamPort, SyncChatStreamPort
from .services import (
    async_collect_text,
    collect_text,
    iter_text_deltas,
    with_timeout,
)

__all__ = [
    "StreamChunk",
    "SyncChatStreamPort",
    "AsyncChatStreamPort",
    "iter_text_deltas",
    "collect_text",
    "async_collect_text",
    "with_timeout",
]
