from __future__ import annotations

"""Domain models for the LLM Gateway.

Purpose:
  - Represent LLM requests and responses in a provider-agnostic way.
  - Capture structured output specifications for strict JSON mode.

Guarantees:
  - No provider-specific types appear in this module.
  - Data models are fully typed and safe for use with type checkers.

Usage:
  Basic example::

    from electripy.ai.llm_gateway import LlmMessage, LlmRequest

    request = LlmRequest(
        model="gpt-4o-mini",
        messages=[LlmMessage.user("Hello!")],
    )
"""

from dataclasses import dataclass, field, replace
from enum import Enum
from typing import Any, List, Mapping, MutableMapping, Optional, Sequence


class LlmRole(str, Enum):
    """Role of a message in a chat-style LLM interaction."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(slots=True)
class LlmMessage:
    """Single message in a chat-style LLM conversation.

    Attributes:
      role: Role of the message author.
      content: Text content of the message.
    """

    role: LlmRole
    content: str

    @staticmethod
    def system(content: str) -> "LlmMessage":
        """Create a system message."""

        return LlmMessage(role=LlmRole.SYSTEM, content=content)

    @staticmethod
    def user(content: str) -> "LlmMessage":
        """Create a user message."""

        return LlmMessage(role=LlmRole.USER, content=content)

    @staticmethod
    def assistant(content: str) -> "LlmMessage":
        """Create an assistant message."""

        return LlmMessage(role=LlmRole.ASSISTANT, content=content)


@dataclass(slots=True)
class StructuredOutputSpec:
    """Specification for strict structured JSON output.

    This is a lightweight, schema-like specification focused on JSON objects
    with simple value types.

    Attributes:
      name: Human-readable name for the schema (for prompts and diagnostics).
      field_types: Mapping from field name to expected Python type.
        Supported types are: str, int, float, bool, list, dict.
      description: Optional free-text description for prompt context.
    """

    name: str
    field_types: Mapping[str, type]
    description: str | None = None

    def describe_for_prompt(self) -> str:
        """Return a human-readable description for use in prompts."""

        parts: list[str] = [f"Schema {self.name}:"]
        for key, value_type in self.field_types.items():
            parts.append(f"- {key}: {value_type.__name__}")
        if self.description:
            parts.append(f"Description: {self.description}")
        return "\n".join(parts)


@dataclass(slots=True)
class LlmRequest:
    """Normalized LLM request.

    Attributes:
      model: Provider-specific model identifier (for example, "gpt-4o-mini").
      messages: Ordered list of messages comprising the prompt.
      temperature: Sampling temperature between 0 and 2; defaults to 0.2.
      max_output_tokens: Maximum tokens the model may generate, if supported.
      max_input_chars: Hard limit on the total number of characters across
        all messages. If exceeded, the gateway raises TokenBudgetExceededError.
      metadata: Optional, caller-defined metadata associated with the request.
    """

    model: str
    messages: List[LlmMessage]
    temperature: float = 0.2
    max_output_tokens: Optional[int] = None
    max_input_chars: Optional[int] = None
    metadata: Mapping[str, Any] | None = None

    def clone_with_messages(self, messages: Sequence[LlmMessage]) -> "LlmRequest":
        """Return a shallow copy of this request with different messages."""

        return replace(self, messages=list(messages))


@dataclass(slots=True)
class LlmResponse:
    """Normalized LLM response.

    Attributes:
      text: Primary text content for plain-text mode or JSON text for
        structured mode.
      raw_json: Parsed JSON object if structured mode is used, else None.
      usage_total_tokens: Total tokens as reported by the provider, if known.
      finish_reason: Provider-reported finish reason (for example, "stop").
      request_id: Provider request identifier, if available.
      model: Effective model used by the provider.
      metadata: Mutable mapping for gateway-added metadata such as
        safety evaluation results.
    """

    text: str
    raw_json: Optional[Mapping[str, Any]] = None
    usage_total_tokens: Optional[int] = None
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    model: Optional[str] = None
    metadata: MutableMapping[str, Any] = field(default_factory=dict)


# Deviation: "Any" is used for raw_json and metadata value types
# because JSON structures are inherently heterogeneous and cannot be
# precisely typed here without a full JSON Schema implementation.
