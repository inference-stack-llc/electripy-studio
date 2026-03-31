"""Policy and approval engine for AI governance.

Purpose:
  - Enforce governance rules for AI/agent/tool operations.
  - Manage approval workflows with time-bound tokens.
  - Support evidence requirements, escalation, and redaction directives.
  - Complement the lower-level ``policy_gateway`` package.
"""

from __future__ import annotations

from .adapters import (
    DefaultPolicyEvaluator,
    InMemoryApprovalStore,
    InMemoryPolicyRepository,
    LoggingEscalationHandler,
)
from .domain import (
    ApprovalRequest,
    ApprovalStatus,
    ApprovalToken,
    DecisionOutcome,
    DecisionReason,
    EscalationDirective,
    EscalationLevel,
    EvidenceItem,
    EvidenceKind,
    PolicyAction,
    PolicyContext,
    PolicyDecision,
    PolicyPack,
    PolicyResource,
    PolicyRule,
    PolicySubject,
    PolicyViolation,
    RedactionDirective,
    RulePriority,
)
from .errors import (
    ApprovalError,
    EscalationError,
    EvidenceError,
    PolicyEngineError,
    PolicyPackError,
)
from .ports import (
    ApprovalStorePort,
    EscalationHandlerPort,
    PolicyEvaluatorPort,
    PolicyObserverPort,
    PolicyRepositoryPort,
)
from .services import PolicyEngine, PolicyEngineSettings

__all__ = [
    # Domain — enumerations
    "DecisionOutcome",
    "RulePriority",
    "EvidenceKind",
    "ApprovalStatus",
    "EscalationLevel",
    # Domain — value objects
    "PolicySubject",
    "PolicyResource",
    "PolicyAction",
    "EvidenceItem",
    "PolicyContext",
    "RedactionDirective",
    "EscalationDirective",
    "PolicyRule",
    "PolicyPack",
    "DecisionReason",
    "PolicyViolation",
    "PolicyDecision",
    "ApprovalRequest",
    "ApprovalToken",
    # Errors
    "PolicyEngineError",
    "PolicyPackError",
    "EvidenceError",
    "ApprovalError",
    "EscalationError",
    # Ports
    "PolicyEvaluatorPort",
    "PolicyRepositoryPort",
    "ApprovalStorePort",
    "EscalationHandlerPort",
    "PolicyObserverPort",
    # Adapters
    "DefaultPolicyEvaluator",
    "InMemoryPolicyRepository",
    "InMemoryApprovalStore",
    "LoggingEscalationHandler",
    # Services
    "PolicyEngine",
    "PolicyEngineSettings",
]
