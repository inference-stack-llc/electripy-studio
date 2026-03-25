"""Domain models for conversation memory."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class TurnRole(Enum):
    """Role of a conversation turn."""

    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


@dataclass(slots=True)
class Turn:
    """A single turn in a conversation.

    Attributes:
        role: The speaker role.
        content: The message text.
        token_count: Cached token count (populated during budget operations).
        metadata: Optional non-sensitive metadata.
    """

    role: TurnRole
    content: str
    token_count: int = 0
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class ConversationWindow:
    """An immutable snapshot of a conversation history.

    Attributes:
        turns: Ordered list of conversation turns.
        total_tokens: Total token count across all turns.
    """

    turns: list[Turn] = field(default_factory=list)
    total_tokens: int = 0

    def to_dicts(self) -> list[dict[str, str]]:
        """Convert turns to a list of plain dicts."""
        return [{"role": t.role.value, "content": t.content} for t in self.turns]
