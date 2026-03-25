"""Domain models for the prompt engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class PromptRole(Enum):
    """Standard chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(slots=True)
class FewShotExample:
    """A single few-shot example pair.

    Attributes:
        user: The user message in the example.
        assistant: The expected assistant response.
        metadata: Optional non-sensitive tags for filtering examples.
    """

    user: str
    assistant: str
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class RenderedMessage:
    """A single rendered chat message.

    Attributes:
        role: The message role (system, user, assistant).
        content: The fully rendered message text.
    """

    role: PromptRole
    content: str

    def to_dict(self) -> dict[str, str]:
        """Convert to a plain dict suitable for LLM API payloads."""
        return {"role": self.role.value, "content": self.content}


@dataclass(slots=True)
class RenderedPrompt:
    """A fully rendered prompt consisting of ordered messages.

    Attributes:
        messages: Ordered list of rendered messages.
    """

    messages: list[RenderedMessage]

    def to_dicts(self) -> list[dict[str, str]]:
        """Convert all messages to plain dicts."""
        return [m.to_dict() for m in self.messages]
