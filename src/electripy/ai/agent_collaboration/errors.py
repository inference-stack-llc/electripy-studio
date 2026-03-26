"""Errors for multi-agent collaboration runtime."""

from __future__ import annotations

__all__ = [
    "AgentCollaborationError",
    "HopLimitExceededError",
    "UnknownAgentError",
]


class AgentCollaborationError(Exception):
    """Base error for collaboration runtime failures."""


class UnknownAgentError(AgentCollaborationError):
    """Raised when a message targets an unregistered agent."""


class HopLimitExceededError(AgentCollaborationError):
    """Raised when collaboration exceeds configured hop budget."""
