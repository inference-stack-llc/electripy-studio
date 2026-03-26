"""Errors for the Structured Output Engine."""

from __future__ import annotations

from dataclasses import dataclass


class StructuredOutputEngineError(Exception):
    """Base exception for all structured output engine errors."""


class SchemaGenerationError(StructuredOutputEngineError):
    """Raised when a JSON schema cannot be generated from the output model."""


@dataclass(slots=True)
class ExtractionError(StructuredOutputEngineError):
    """Raised when a single extraction attempt fails.

    Attributes:
      message: Human-readable description.
      raw_text: The raw LLM output that could not be parsed.
      attempt: 1-based attempt index.
    """

    message: str
    raw_text: str
    attempt: int

    def __str__(self) -> str:  # pragma: no cover — trivial formatting
        return f"attempt {self.attempt}: {self.message}"


@dataclass(slots=True)
class ExtractionExhaustedError(StructuredOutputEngineError):
    """Raised when all extraction attempts have been exhausted.

    Attributes:
      message: Human-readable summary.
      attempts: Number of attempts made.
      last_error: The error from the final attempt.
    """

    message: str
    attempts: int
    last_error: str

    def __str__(self) -> str:  # pragma: no cover — trivial formatting
        return f"{self.message} (after {self.attempts} attempts: {self.last_error})"
