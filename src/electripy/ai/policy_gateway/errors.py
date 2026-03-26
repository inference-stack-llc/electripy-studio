"""Errors for policy gateway evaluation and enforcement."""

from __future__ import annotations

__all__ = [
    "PolicyConfigurationError",
    "PolicyEvaluationError",
    "PolicyGatewayError",
]


class PolicyGatewayError(Exception):
    """Base exception for policy gateway failures."""


class PolicyConfigurationError(PolicyGatewayError):
    """Raised when policy rules or settings are invalid."""


class PolicyEvaluationError(PolicyGatewayError):
    """Raised when policy detection/sanitization fails unexpectedly."""
