"""Domain models for policy gateway decisions and findings."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum


class PolicyAction(StrEnum):
    """Action returned by policy evaluation."""

    ALLOW = "allow"
    SANITIZE = "sanitize"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class PolicyStage(StrEnum):
    """Evaluation stage for policy checks."""

    PREFLIGHT = "preflight"
    POSTFLIGHT = "postflight"
    STREAM = "stream"
    TOOL_CALL = "tool_call"


class PolicySeverity(StrEnum):
    """Severity level of a policy finding."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass(slots=True)
class PolicyContext:
    """Correlation context for policy evaluations."""

    request_id: str | None = None
    actor_id: str | None = None
    tenant_id: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(slots=True)
class PolicyRule:
    """Rule definition used by detector adapters.

    Attributes:
        rule_id: Stable identifier for this rule.
        code: Stable violation code.
        description: Human-readable description.
        stage: Stage where this rule applies.
        pattern: Regex pattern to detect violations.
        severity: Finding severity when pattern matches.
        action: Recommended action for this rule.
        replacement: Replacement text used by sanitizers.
    """

    rule_id: str
    code: str
    description: str
    stage: PolicyStage
    pattern: str
    severity: PolicySeverity = PolicySeverity.HIGH
    action: PolicyAction = PolicyAction.SANITIZE
    replacement: str = "[REDACTED]"


@dataclass(slots=True)
class PolicyInput:
    """Input payload passed to policy detectors."""

    stage: PolicyStage
    text: str
    tool_name: str | None = None
    tool_args: dict[str, object] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)
    context: PolicyContext | None = None


@dataclass(slots=True)
class PolicyFinding:
    """Single policy finding from detector evaluation."""

    rule_id: str
    code: str
    message: str
    severity: PolicySeverity
    action: PolicyAction
    start: int | None = None
    end: int | None = None


@dataclass(slots=True)
class PolicyDecision:
    """Final policy decision for a given request/response."""

    action: PolicyAction
    policy_version: str
    findings: list[PolicyFinding] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)
    sanitized_text: str | None = None
    requires_approval: bool = False
    evaluated_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))

    @property
    def blocked(self) -> bool:
        """Return ``True`` when action blocks execution."""

        return self.action in (PolicyAction.DENY, PolicyAction.REQUIRE_APPROVAL)
