"""Priority-based context window assembly for LLM prompts.

Purpose:
  - Pack system prompts, documents, examples, and user queries into a
    token-limited context window with explicit priority ordering.
  - Automatically trim lower-priority blocks when the budget is exceeded.

Guarantees:
  - Higher-priority blocks are never dropped before lower-priority ones.
  - Uses the TokenizerPort from token_budget for consistent counting.
"""

from __future__ import annotations

from .domain import AssembledContext, ContextBlock, ContextPriority
from .errors import AssemblyError, EmptyAssemblyError
from .services import assemble_context

__all__ = [
    "ContextBlock",
    "ContextPriority",
    "AssembledContext",
    "AssemblyError",
    "EmptyAssemblyError",
    "assemble_context",
]
