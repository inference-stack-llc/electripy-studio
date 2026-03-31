"""Ports (Protocols) for the policy and approval engine.

All ports are runtime-checkable Protocols so adapters can be
substituted without inheritance coupling.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from .domain import (
    ApprovalRequest,
    ApprovalToken,
    EscalationDirective,
    PolicyContext,
    PolicyDecision,
    PolicyRule,
)

__all__ = [
    "ApprovalStorePort",
    "EscalationHandlerPort",
    "PolicyEvaluatorPort",
    "PolicyRepositoryPort",
    "PolicyObserverPort",
]


@runtime_checkable
class PolicyEvaluatorPort(Protocol):
    """Evaluates a policy context against a set of rules."""

    def evaluate(
        self,
        context: PolicyContext,
        rules: Sequence[PolicyRule],
    ) -> PolicyDecision:
        """Return a decision for the given context and rules."""
        ...


@runtime_checkable
class PolicyRepositoryPort(Protocol):
    """Loads and stores policy packs and rules."""

    def list_rules(self, *, resource_type: str | None = None) -> list[PolicyRule]:
        """Return all enabled rules, optionally filtered by resource type."""
        ...


@runtime_checkable
class ApprovalStorePort(Protocol):
    """Persists and retrieves approval requests and tokens."""

    def save_request(self, request: ApprovalRequest) -> None:
        """Persist an approval request."""
        ...

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        """Retrieve an approval request by ID."""
        ...

    def save_token(self, token: ApprovalToken) -> None:
        """Persist an approval token."""
        ...

    def get_token(self, token_id: str) -> ApprovalToken | None:
        """Retrieve an approval token by ID."""
        ...


@runtime_checkable
class EscalationHandlerPort(Protocol):
    """Dispatches escalation directives to appropriate channels."""

    def escalate(
        self,
        directive: EscalationDirective,
        context: PolicyContext,
    ) -> None:
        """Send an escalation notification."""
        ...


@runtime_checkable
class PolicyObserverPort(Protocol):
    """Receives notifications about policy evaluation events."""

    def on_decision(
        self,
        context: PolicyContext,
        decision: PolicyDecision,
    ) -> None:
        """Called after a policy decision is produced."""
        ...
