"""Domain models for the Structured Output Engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Generic, TypeVar

T = TypeVar("T")


class ExtractionStrategy(StrEnum):
    """Strategy for extracting structured output from LLM text."""

    JSON_MODE = "json_mode"
    PROMPT_COERCE = "prompt_coerce"


@dataclass(frozen=True, slots=True)
class ExtractionAttempt:
    """Record of a single extraction attempt.

    Attributes:
      attempt_number: 1-based index of this attempt.
      raw_text: The raw text returned by the LLM.
      temperature: The temperature used for this attempt.
      error: Validation or parse error message, if the attempt failed.
      success: Whether the attempt produced a valid parsed object.
    """

    attempt_number: int
    raw_text: str
    temperature: float
    error: str | None = None
    success: bool = False


@dataclass(frozen=True, slots=True)
class ExtractionResult(Generic[T]):
    """Result of a structured output extraction pipeline.

    Attributes:
      parsed: The validated and typed output object.
      attempts: Full history of attempts (including failures).
      model: The LLM model identifier used.
      schema_prompt: The JSON-schema snippet injected into the prompt.
      total_attempts: Number of attempts made before success.
    """

    parsed: T
    attempts: tuple[ExtractionAttempt, ...]
    model: str
    schema_prompt: str
    total_attempts: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "total_attempts", len(self.attempts))
