"""Declarative tool definitions and JSON schema generation for agent frameworks.

Purpose:
  - Define agent tools from plain Python functions with automatic schema generation.
  - Provide argument validation and a type-safe tool registry.

Guarantees:
  - No external dependencies; uses stdlib inspect + typing for schema inference.
  - Generated schemas follow the JSON Schema subset used by OpenAI function calling.
"""

from __future__ import annotations

from .domain import ToolDefinition, ToolParameter, ToolSchema
from .errors import ToolRegistryError, ToolValidationError
from .services import (
    ToolRegistry,
    generate_schema,
    tool_from_function,
    validate_arguments,
)

__all__ = [
    "ToolDefinition",
    "ToolParameter",
    "ToolSchema",
    "ToolRegistryError",
    "ToolValidationError",
    "ToolRegistry",
    "tool_from_function",
    "generate_schema",
    "validate_arguments",
]
