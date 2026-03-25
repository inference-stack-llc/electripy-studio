"""Exception hierarchy for the prompt engine."""

from __future__ import annotations


class PromptEngineError(Exception):
    """Base exception for prompt engine errors."""


class MissingVariableError(PromptEngineError):
    """Raised when a required template variable is not provided."""

    def __init__(self, variable: str) -> None:
        self.variable = variable
        super().__init__(f"Missing template variable: {variable!r}")


class TemplateSyntaxError(PromptEngineError):
    """Raised when a template string has invalid syntax."""
