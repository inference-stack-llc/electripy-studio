"""Domain models for token budget management."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class TruncationStrategy(Enum):
    """Strategy for truncating text to fit a token budget."""

    TAIL = "tail"
    HEAD = "head"
    MIDDLE = "middle"


@dataclass(slots=True)
class TokenCount:
    """Result of counting tokens in a text.

    Attributes:
        text: The original text that was counted.
        token_count: Number of tokens.
    """

    text: str
    token_count: int


@dataclass(slots=True)
class TruncationResult:
    """Result of truncating text to fit a budget.

    Attributes:
        text: The truncated text.
        original_tokens: Token count before truncation.
        final_tokens: Token count after truncation.
        was_truncated: Whether truncation was necessary.
    """

    text: str
    original_tokens: int
    final_tokens: int
    was_truncated: bool
