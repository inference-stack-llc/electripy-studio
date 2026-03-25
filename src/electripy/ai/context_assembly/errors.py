"""Exception hierarchy for context assembly."""

from __future__ import annotations


class AssemblyError(Exception):
    """Base exception for context assembly errors."""


class EmptyAssemblyError(AssemblyError):
    """Raised when no blocks can fit within the budget."""
