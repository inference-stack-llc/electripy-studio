"""Domain models for the LLM Replay Tape."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class DiffStatus(StrEnum):
    """Status of a single entry comparison in a tape diff."""

    IDENTICAL = "identical"
    CHANGED = "changed"
    ADDED = "added"
    REMOVED = "removed"


@dataclass(frozen=True, slots=True)
class TapeEntry:
    """A single recorded LLM interaction.

    Attributes:
      index: 0-based position in the tape.
      model: The LLM model identifier used.
      messages: The request messages as a list of role/content dicts.
      temperature: Sampling temperature used.
      response_text: The raw text returned by the LLM.
      metadata: Optional caller-supplied metadata.
    """

    index: int
    model: str
    messages: tuple[dict[str, str], ...]
    temperature: float
    response_text: str
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DiffResult:
    """Result of comparing a single tape entry pair.

    Attributes:
      index: The tape index being compared.
      status: Whether the entry is identical, changed, added, or removed.
      text_a: Response text from tape A (None if added).
      text_b: Response text from tape B (None if removed).
    """

    index: int
    status: DiffStatus
    text_a: str | None = None
    text_b: str | None = None
