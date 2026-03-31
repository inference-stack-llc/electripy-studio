"""Custom exceptions for the observe package.

All exceptions inherit from :class:`electripy.core.errors.ElectriPyError`
to integrate cleanly with the project-wide exception hierarchy.
"""

from __future__ import annotations

from electripy.core.errors import ElectriPyError

__all__ = [
    "ObserveError",
    "SpanError",
    "TracerConfigError",
    "RedactionError",
]


class ObserveError(ElectriPyError):
    """Base exception for all observe-package errors."""


class SpanError(ObserveError):
    """Raised when a span operation is invalid or fails."""


class TracerConfigError(ObserveError):
    """Raised when a tracer cannot be configured correctly."""


class RedactionError(ObserveError):
    """Raised when a redaction rule cannot be applied."""
