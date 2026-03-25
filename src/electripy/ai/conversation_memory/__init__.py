"""Token-budget-aware conversation memory management for chat applications.

Purpose:
  - Manage chat history with sliding window and token-aware truncation.
  - Provide deterministic, testable conversation state management.

Guarantees:
  - No mutable global state; all state lives in ConversationWindow instances.
  - Uses the TokenizerPort from token_budget for consistent counting.
"""

from __future__ import annotations

from .domain import ConversationWindow, Turn, TurnRole
from .errors import ConversationMemoryError
from .services import (
    append_turn,
    recent_turns,
    sliding_window,
    trim_to_budget,
)

__all__ = [
    "Turn",
    "TurnRole",
    "ConversationWindow",
    "ConversationMemoryError",
    "append_turn",
    "recent_turns",
    "sliding_window",
    "trim_to_budget",
]
