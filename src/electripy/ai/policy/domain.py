"""Domain models for the policy and approval engine.

Complements the lower-level ``policy_gateway`` package with
approval workflows, evidence requirements, escalation directives,
and versioned policy packs for enterprise AI governance.

All value objects are frozen dataclasses for immutability and
hashability.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from uuid import uuid4

__all__ = [
    "ApprovalRequest",
    "ApprovalStatus",
    "ApprovalToken",
    "DecisionOutcome",
    "DecisionReason",
    "EscalationDirective",
    "EscalationLevel",
    "EvidenceItem",
    "EvidenceKind",
    "PolicyAction",
    "PolicyContext",
    "PolicyDecision",
    "PolicyPack",
    "PolicyResource",
    "PolicyRule",
    "PolicySubject",
    "PolicyViolation",
    "RedactionDirective",
    "RulePriority",
]


# ── Enumerations ─────────────────────────────────────────────────────


class DecisionOutcome(StrEnum):
    """Possible outcomes of a policy evaluation."""

    ALLOW = "allow"
    DENY = "deny"
    REDACT = "redact"
    REQUIRE_APPROVAL = "require_approval"
    ESCALATE = "escalate"


class RulePriority(StrEnum):
    """Priority tiers for policy rules."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EvidenceKind(StrEnum):
    """Types of evidence that may satisfy a policy requirement."""

    JUSTIFICATION = "justification"
    TICKET_REFERENCE = "ticket_reference"
    APPROVAL_TOKEN = "approval_token"
    MFA_CHALLENGE = "mfa_challenge"
    AUDIT_LOG = "audit_log"


class ApprovalStatus(StrEnum):
    """Status of an approval request."""

    PENDING = "pending"
    APPROVED = "approved"
    DENIED = "denied"
    EXPIRED = "expired"


class EscalationLevel(StrEnum):
    """Escalation tier for policy violations."""

    TEAM_LEAD = "team_lead"
    MANAGER = "manager"
    SECURITY = "security"
    EXECUTIVE = "executive"


# ── Value objects ────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class PolicySubject:
    """Who is requesting the action.

    Attributes:
        actor_id: Unique actor identifier.
        roles: Roles assigned to the actor.
        teams: Teams the actor belongs to.
        attributes: Arbitrary actor metadata.
    """

    actor_id: str
    roles: tuple[str, ...] = ()
    teams: tuple[str, ...] = ()
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyResource:
    """What the action targets.

    Attributes:
        resource_type: Category such as ``tool``, ``model``, ``endpoint``.
        resource_id: Stable identifier for the resource.
        attributes: Arbitrary resource metadata.
    """

    resource_type: str
    resource_id: str
    attributes: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyAction:
    """The operation the subject wants to perform.

    Attributes:
        action_type: Verb such as ``invoke``, ``read``, ``write``, ``delete``.
        parameters: Arbitrary parameters describing the action.
    """

    action_type: str
    parameters: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True, slots=True)
class EvidenceItem:
    """A single piece of evidence supporting a request.

    Attributes:
        kind: Type of evidence provided.
        value: Content of the evidence.
        provided_at: When the evidence was supplied.
    """

    kind: EvidenceKind
    value: str
    provided_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


@dataclass(frozen=True, slots=True)
class PolicyContext:
    """Full context for a policy evaluation request.

    Attributes:
        subject: The actor requesting the action.
        resource: The target of the action.
        action: The operation being performed.
        evidence: Evidence items supporting the request.
        request_id: Correlation identifier.
        tags: Arbitrary context tags.
        timestamp: When the evaluation was requested.
    """

    subject: PolicySubject
    resource: PolicyResource
    action: PolicyAction
    evidence: tuple[EvidenceItem, ...] = ()
    request_id: str = field(default_factory=lambda: uuid4().hex)
    tags: tuple[tuple[str, str], ...] = ()
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


# ── Rules and packs ─────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RedactionDirective:
    """Instructions for redacting specific fields in outputs.

    Attributes:
        field_path: Dot-separated path to the field to redact.
        replacement: Text to replace the field value with.
    """

    field_path: str
    replacement: str = "[REDACTED]"


@dataclass(frozen=True, slots=True)
class EscalationDirective:
    """Instructions for escalating a policy violation.

    Attributes:
        level: Escalation tier.
        reason: Human-readable explanation.
        notify_channels: Channels to notify (e.g. ``slack:#security``).
    """

    level: EscalationLevel
    reason: str
    notify_channels: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class PolicyRule:
    """A single governance rule in the policy engine.

    Attributes:
        rule_id: Stable identifier for this rule.
        description: Human-readable description.
        priority: How severe a violation of this rule is.
        outcome: Default decision outcome when this rule triggers.
        resource_types: Resource types this rule applies to (empty = all).
        action_types: Action types this rule applies to (empty = all).
        required_roles: Roles that satisfy this rule (empty = any role).
        required_evidence: Evidence kinds that must be present.
        redaction: Redaction instructions when outcome is ``redact``.
        escalation: Escalation instructions when outcome is ``escalate``.
        ttl_seconds: Time limit in seconds for a granted approval.
        enabled: Whether the rule is active.
    """

    rule_id: str
    description: str
    priority: RulePriority = RulePriority.MEDIUM
    outcome: DecisionOutcome = DecisionOutcome.DENY
    resource_types: tuple[str, ...] = ()
    action_types: tuple[str, ...] = ()
    required_roles: tuple[str, ...] = ()
    required_evidence: tuple[EvidenceKind, ...] = ()
    redaction: RedactionDirective | None = None
    escalation: EscalationDirective | None = None
    ttl_seconds: int | None = None
    enabled: bool = True


@dataclass(frozen=True, slots=True)
class PolicyPack:
    """A versioned collection of policy rules.

    Attributes:
        pack_id: Stable identifier for this pack.
        version: Semantic version string.
        rules: Ordered tuple of rules in this pack.
        description: Human-readable description.
    """

    pack_id: str
    version: str
    rules: tuple[PolicyRule, ...]
    description: str = ""


# ── Decisions and violations ─────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class DecisionReason:
    """Human-readable explanation for a policy decision.

    Attributes:
        rule_id: Rule that produced this reason.
        message: Explanation text.
    """

    rule_id: str
    message: str


@dataclass(frozen=True, slots=True)
class PolicyViolation:
    """A specific rule violation found during evaluation.

    Attributes:
        rule_id: The rule that was violated.
        priority: Severity of the violation.
        message: Human-readable violation description.
    """

    rule_id: str
    priority: RulePriority
    message: str


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Result of a policy evaluation.

    Attributes:
        outcome: The decision outcome.
        violations: Violations found during evaluation.
        reasons: Human-readable explanations for the decision.
        redactions: Redaction directives to apply.
        escalation: Escalation directive, if triggered.
        policy_version: Version of the policy pack used.
        request_id: Correlation identifier from the context.
        evaluated_at: When the decision was produced.
    """

    outcome: DecisionOutcome
    violations: tuple[PolicyViolation, ...] = ()
    reasons: tuple[DecisionReason, ...] = ()
    redactions: tuple[RedactionDirective, ...] = ()
    escalation: EscalationDirective | None = None
    policy_version: str = ""
    request_id: str = ""
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @property
    def blocked(self) -> bool:
        """Return ``True`` when the outcome blocks execution."""
        return self.outcome in (
            DecisionOutcome.DENY,
            DecisionOutcome.REQUIRE_APPROVAL,
            DecisionOutcome.ESCALATE,
        )

    @property
    def requires_evidence(self) -> bool:
        """Return ``True`` when the outcome requires additional evidence."""
        return self.outcome == DecisionOutcome.REQUIRE_APPROVAL


@dataclass(frozen=True, slots=True)
class ApprovalRequest:
    """An approval request generated by a require_approval decision.

    Attributes:
        request_id: Correlation identifier.
        subject: The actor requesting approval.
        resource: The target resource.
        action: The requested action.
        required_evidence: Evidence kinds needed for approval.
        ttl: Duration before the approval request expires.
        status: Current status of the approval request.
        created_at: When the request was created.
    """

    request_id: str = field(default_factory=lambda: uuid4().hex)
    subject: PolicySubject | None = None
    resource: PolicyResource | None = None
    action: PolicyAction | None = None
    required_evidence: tuple[EvidenceKind, ...] = ()
    ttl: timedelta = field(default_factory=lambda: timedelta(hours=1))
    status: ApprovalStatus = ApprovalStatus.PENDING
    created_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @property
    def expired(self) -> bool:
        """Return ``True`` when the request has exceeded its TTL."""
        return datetime.now(tz=UTC) > self.created_at + self.ttl


@dataclass(frozen=True, slots=True)
class ApprovalToken:
    """A time-bound token granting approved access.

    Attributes:
        token_id: Unique token identifier.
        request_id: The approval request this token satisfies.
        granted_by: Identifier of the approver.
        granted_at: When the token was issued.
        expires_at: When the token expires.
        evidence: Evidence items submitted for approval.
    """

    token_id: str = field(default_factory=lambda: uuid4().hex)
    request_id: str = ""
    granted_by: str = ""
    granted_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))
    expires_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC) + timedelta(hours=1))
    evidence: tuple[EvidenceItem, ...] = ()

    @property
    def valid(self) -> bool:
        """Return ``True`` when the token has not yet expired."""
        return datetime.now(tz=UTC) < self.expires_at
