"""Exception hierarchy for model routing."""

from __future__ import annotations


class ModelRouterError(Exception):
    """Base exception for model routing errors."""


class NoMatchingModelError(ModelRouterError):
    """Raised when no model matches the given routing rules."""
