"""Errors for the policy and approval engine."""

from __future__ import annotations

from electripy.core.errors import ElectriPyError

__all__ = [
    "ApprovalError",
    "EscalationError",
    "EvidenceError",
    "PolicyEngineError",
    "PolicyPackError",
]


class PolicyEngineError(ElectriPyError):
    """Base exception for all policy engine failures."""


class PolicyPackError(PolicyEngineError):
    """Raised when a policy pack is invalid or contains conflicting rules."""


class EvidenceError(PolicyEngineError):
    """Raised when required evidence is missing or invalid."""


class ApprovalError(PolicyEngineError):
    """Raised when an approval flow fails (expired, invalid token, etc.)."""


class EscalationError(PolicyEngineError):
    """Raised when an escalation directive cannot be fulfilled."""
