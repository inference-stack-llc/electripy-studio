"""Domain exceptions for the LLM Gateway.

Purpose:
  - Provide explicit, typed failure modes for the LLM gateway.
  - Shield callers from provider-specific exceptions.

Guarantees:
  - No raw third-party exceptions cross the public boundary.
  - Exceptions carry enough context for diagnostics without exposing prompts.

Usage:
  Basic example::

    from electripy.ai.llm_gateway import LlmGatewayError

    try:
        client.complete(request)
    except LlmGatewayError as exc:
        handle(exc)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final


class LlmGatewayError(Exception):
    """Base exception for all LLM gateway related errors."""


@dataclass(slots=True)
class RateLimitedError(LlmGatewayError):
    """Raised when the provider signals a rate limit.

    Attributes:
      message: Human-readable description of the error.
      status_code: Optional HTTP-like status code, typically 429.
      retry_after_seconds: Parsed Retry-After hint if provided by the provider.

    Example::

      raise RateLimitedError(
          "Rate limited by provider", status_code=429, retry_after_seconds=2.0
      )
    """

    message: str
    status_code: int | None = None
    retry_after_seconds: float | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        base = self.message
        if self.status_code is not None:
            base += f" (status={self.status_code})"
        if self.retry_after_seconds is not None:
            base += f", retry_after={self.retry_after_seconds}s"
        return base


@dataclass(slots=True)
class RetryExhaustedError(LlmGatewayError):
    """Raised when retry attempts are exhausted.

    Attributes:
      attempts: Number of attempts that were made.
      last_error_message: Message from the last underlying error.

    Example::

      raise RetryExhaustedError(attempts=3, last_error_message="Timeout")
    """

    attempts: int
    last_error_message: str

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return (
            f"Retry attempts exhausted after {self.attempts} attempts: "
            f"{self.last_error_message}"
        )


@dataclass(slots=True)
class StructuredOutputError(LlmGatewayError):
    """Raised when strict structured output validation fails.

    Attributes:
      details: Human-readable description of the validation failure.
      last_raw_output: Last raw text returned by the model (truncated if needed).
      validation_errors: List of machine-readable validation error messages.

    Security:
      - Does not contain the original prompt.
      - Raw output should be truncated or scrubbed by callers before logging.
    """

    details: str
    last_raw_output: str | None = None
    validation_errors: tuple[str, ...] = ()

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return f"StructuredOutputError: {self.details}"


@dataclass(slots=True)
class TokenBudgetExceededError(LlmGatewayError):
    """Raised when an LLM request exceeds the configured input budget.

    Attributes:
      input_chars: Number of characters observed in the request.
      max_input_chars: Configured maximum allowed characters.

    Example::

      raise TokenBudgetExceededError(input_chars=12000, max_input_chars=8000)
    """

    input_chars: int
    max_input_chars: int

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return (
            f"Token budget exceeded: {self.input_chars} chars > "
            f"limit of {self.max_input_chars} chars"
        )


@dataclass(slots=True)
class PromptRejectedError(LlmGatewayError):
    """Raised when the prompt guard rejects a request.

    Attributes:
      reasons: Reasons provided by the guard.

    Example::

      raise PromptRejectedError(reasons=("Prompt contained exfiltration intent",))
    """

    reasons: tuple[str, ...]

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        reasons_text = "; ".join(self.reasons) or "no reasons provided"
        return f"Prompt rejected by guard: {reasons_text}"


@dataclass(slots=True)
class TransientLlmError(LlmGatewayError):
    """Internal error class for transient provider failures.

    This is not part of the public API and is used only to distinguish
    retryable failures inside the gateway's retry logic.
    """

    message: str
    status_code: int | None = None

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        if self.status_code is None:
            return self.message
        return f"{self.message} (status={self.status_code})"


TRANSIENT_STATUS_CODES: Final[tuple[int, ...]] = (500, 502, 503, 504)
