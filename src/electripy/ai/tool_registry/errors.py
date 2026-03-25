"""Exception hierarchy for the tool registry."""

from __future__ import annotations


class ToolRegistryError(Exception):
    """Base exception for tool registry errors."""


class ToolValidationError(ToolRegistryError):
    """Raised when tool arguments fail validation."""

    def __init__(self, tool_name: str, details: str) -> None:
        self.tool_name = tool_name
        self.details = details
        super().__init__(f"Validation failed for tool {tool_name!r}: {details}")
