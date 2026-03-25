"""Prompt templating, composition, and few-shot example management.

Purpose:
  - Provide a lightweight, type-safe prompt template engine for LLM applications.
  - Support variable injection, few-shot example blocks, and message composition.

Guarantees:
  - No external dependencies beyond the standard library.
  - Templates are validated at render time; missing variables raise explicit errors.
"""

from __future__ import annotations

from .domain import FewShotExample, PromptRole, RenderedMessage, RenderedPrompt
from .errors import MissingVariableError, PromptEngineError, TemplateSyntaxError
from .services import (
    build_few_shot_block,
    compose_messages,
    render_template,
)

__all__ = [
    "FewShotExample",
    "PromptRole",
    "RenderedMessage",
    "RenderedPrompt",
    "PromptEngineError",
    "MissingVariableError",
    "TemplateSyntaxError",
    "render_template",
    "build_few_shot_block",
    "compose_messages",
]
