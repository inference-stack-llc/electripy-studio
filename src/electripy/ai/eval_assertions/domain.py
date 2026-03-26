"""Domain models for Eval Assertions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class AssertionSeverity(StrEnum):
    """Severity level for assertion checks."""

    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True, slots=True)
class AssertionCheck:
    """A single declarative assertion to run against LLM output.

    Attributes:
      name: Short human-readable label (e.g. "contains_keywords").
      check_fn: Callable accepting a string and returning True if the
        check passes.
      severity: Whether a failure is an error or a warning.
      description: Optional detailed explanation of the check.
    """

    name: str
    check_fn: Callable[[str], bool]
    severity: AssertionSeverity = AssertionSeverity.ERROR
    description: str = ""


@dataclass(frozen=True, slots=True)
class AssertionResult:
    """Result of running a single assertion check.

    Attributes:
      check_name: Name of the check that was run.
      passed: Whether the check passed.
      severity: Severity of the check.
      message: Human-readable diagnostic message.
    """

    check_name: str
    passed: bool
    severity: AssertionSeverity
    message: str = ""
