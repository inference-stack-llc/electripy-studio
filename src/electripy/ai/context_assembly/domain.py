"""Domain models for context assembly."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class ContextPriority(IntEnum):
    """Priority levels for context blocks (higher = more important)."""

    LOW = 10
    MEDIUM = 20
    HIGH = 30
    CRITICAL = 40


@dataclass(slots=True)
class ContextBlock:
    """A single block of content to include in the context window.

    Attributes:
        label: Human-readable label for this block (e.g. "system_prompt").
        content: The text content.
        priority: Priority level; higher values survive truncation.
        token_count: Cached token count (populated during assembly).
    """

    label: str
    content: str
    priority: ContextPriority = ContextPriority.MEDIUM
    token_count: int = 0


@dataclass(slots=True)
class AssembledContext:
    """Result of assembling context blocks within a budget.

    Attributes:
        blocks: Blocks that survived assembly, in original insertion order.
        total_tokens: Total token count of assembled blocks.
        dropped_labels: Labels of blocks that were dropped due to budget.
        budget: The token budget used for assembly.
    """

    blocks: list[ContextBlock]
    total_tokens: int
    dropped_labels: list[str] = field(default_factory=list)
    budget: int = 0

    @property
    def text(self) -> str:
        """Concatenate all surviving block contents with double newlines."""
        return "\n\n".join(b.content for b in self.blocks)
