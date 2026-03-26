"""Ports for pluggable multi-agent implementations."""

from __future__ import annotations

from typing import Protocol

from .domain import AgentMessage, AgentTurnResult, CollaborationTask

__all__ = ["CollaborationAgentPort"]


class CollaborationAgentPort(Protocol):
    """Protocol for agent handlers used by collaboration runtime."""

    def handle(self, message: AgentMessage, *, task: CollaborationTask) -> AgentTurnResult:
        """Handle one incoming message and return next messages."""
