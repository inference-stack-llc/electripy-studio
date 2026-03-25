"""Deterministic runtime for agent-to-agent collaboration.

Purpose:
  - Coordinate specialist agents with explicit message contracts.
  - Keep handoffs bounded and auditable for enterprise workflows.
"""

from __future__ import annotations

from .domain import AgentMessage, AgentTurnResult, CollaborationRunResult, CollaborationTask
from .errors import AgentCollaborationError, HopLimitExceededError, UnknownAgentError
from .ports import CollaborationAgentPort
from .services import AgentCollaborationRuntime, CollaborationRuntimeSettings, make_message

__all__ = [
    "CollaborationTask",
    "AgentMessage",
    "AgentTurnResult",
    "CollaborationRunResult",
    "CollaborationAgentPort",
    "AgentCollaborationRuntime",
    "CollaborationRuntimeSettings",
    "make_message",
    "AgentCollaborationError",
    "UnknownAgentError",
    "HopLimitExceededError",
]
