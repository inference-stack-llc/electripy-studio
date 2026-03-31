"""Adapter implementations for policy engine ports."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from .domain import (
    ApprovalRequest,
    ApprovalToken,
    DecisionOutcome,
    DecisionReason,
    EscalationDirective,
    PolicyContext,
    PolicyDecision,
    PolicyRule,
    PolicyViolation,
    RedactionDirective,
)

__all__ = [
    "DefaultPolicyEvaluator",
    "InMemoryApprovalStore",
    "InMemoryPolicyRepository",
    "LoggingEscalationHandler",
]


@dataclass(slots=True)
class DefaultPolicyEvaluator:
    """Deterministic rule-based policy evaluator.

    Evaluates rules in priority order (critical → low).  The first
    blocking rule determines the outcome.  When multiple rules match
    and none block, the highest-priority non-allow outcome wins.
    """

    def evaluate(
        self,
        context: PolicyContext,
        rules: Sequence[PolicyRule],
    ) -> PolicyDecision:
        """Evaluate *context* against *rules* and return a decision."""
        matching_rules = self._match_rules(context, rules)
        if not matching_rules:
            return PolicyDecision(
                outcome=DecisionOutcome.ALLOW,
                request_id=context.request_id,
            )

        violations: list[PolicyViolation] = []
        reasons: list[DecisionReason] = []
        redactions: list[RedactionDirective] = []
        escalation: EscalationDirective | None = None

        for rule in matching_rules:
            evidence_satisfied = self._check_evidence(context, rule)
            role_satisfied = self._check_roles(context, rule)

            if role_satisfied and evidence_satisfied:
                continue  # Rule conditions met — no violation.

            violation_msg = self._build_violation_message(rule, role_satisfied, evidence_satisfied)
            violations.append(
                PolicyViolation(
                    rule_id=rule.rule_id,
                    priority=rule.priority,
                    message=violation_msg,
                )
            )
            reasons.append(DecisionReason(rule_id=rule.rule_id, message=violation_msg))
            if rule.redaction is not None:
                redactions.append(rule.redaction)
            if rule.escalation is not None and escalation is None:
                escalation = rule.escalation

        if not violations:
            return PolicyDecision(
                outcome=DecisionOutcome.ALLOW,
                request_id=context.request_id,
            )

        outcome = self._resolve_outcome(violations, matching_rules)
        return PolicyDecision(
            outcome=outcome,
            violations=tuple(violations),
            reasons=tuple(reasons),
            redactions=tuple(redactions),
            escalation=escalation,
            request_id=context.request_id,
        )

    # ── private helpers ──────────────────────────────────────────────

    def _match_rules(
        self,
        context: PolicyContext,
        rules: Sequence[PolicyRule],
    ) -> list[PolicyRule]:
        matched: list[PolicyRule] = []
        for rule in rules:
            if not rule.enabled:
                continue
            if rule.resource_types and context.resource.resource_type not in rule.resource_types:
                continue
            if rule.action_types and context.action.action_type not in rule.action_types:
                continue
            matched.append(rule)
        # Sort critical-first for deterministic evaluation.
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        matched.sort(key=lambda r: priority_order.get(r.priority.value, 99))
        return matched

    def _check_evidence(self, context: PolicyContext, rule: PolicyRule) -> bool:
        if not rule.required_evidence:
            return True
        provided_kinds = {e.kind for e in context.evidence}
        return all(kind in provided_kinds for kind in rule.required_evidence)

    def _check_roles(self, context: PolicyContext, rule: PolicyRule) -> bool:
        if not rule.required_roles:
            return True
        return bool(set(context.subject.roles) & set(rule.required_roles))

    def _build_violation_message(self, rule: PolicyRule, role_ok: bool, evidence_ok: bool) -> str:
        parts: list[str] = [rule.description]
        if not role_ok:
            parts.append(f"Required roles: {', '.join(rule.required_roles)}")
        if not evidence_ok:
            parts.append(f"Required evidence: {', '.join(e.value for e in rule.required_evidence)}")
        return "; ".join(parts)

    def _resolve_outcome(
        self,
        violations: list[PolicyViolation],
        matched_rules: list[PolicyRule],
    ) -> DecisionOutcome:
        rule_map = {r.rule_id: r for r in matched_rules}
        # Collect outcomes from violated rules.
        outcomes: set[DecisionOutcome] = set()
        for v in violations:
            rule = rule_map.get(v.rule_id)
            if rule is not None:
                outcomes.add(rule.outcome)
        # Highest-severity outcome wins.
        if DecisionOutcome.DENY in outcomes:
            return DecisionOutcome.DENY
        if DecisionOutcome.ESCALATE in outcomes:
            return DecisionOutcome.ESCALATE
        if DecisionOutcome.REQUIRE_APPROVAL in outcomes:
            return DecisionOutcome.REQUIRE_APPROVAL
        if DecisionOutcome.REDACT in outcomes:
            return DecisionOutcome.REDACT
        return DecisionOutcome.DENY


@dataclass(slots=True)
class InMemoryPolicyRepository:
    """In-memory policy rule store for testing and development."""

    _rules: list[PolicyRule] = field(default_factory=list)

    def add_rule(self, rule: PolicyRule) -> None:
        """Add a rule to the repository."""
        self._rules.append(rule)

    def list_rules(self, *, resource_type: str | None = None) -> list[PolicyRule]:
        """Return all enabled rules, optionally filtered by resource type."""
        result: list[PolicyRule] = []
        for rule in self._rules:
            if not rule.enabled:
                continue
            if resource_type and rule.resource_types and resource_type not in rule.resource_types:
                continue
            result.append(rule)
        return result


@dataclass(slots=True)
class InMemoryApprovalStore:
    """In-memory approval store for testing and development."""

    _requests: dict[str, ApprovalRequest] = field(default_factory=dict)
    _tokens: dict[str, ApprovalToken] = field(default_factory=dict)

    def save_request(self, request: ApprovalRequest) -> None:
        self._requests[request.request_id] = request

    def get_request(self, request_id: str) -> ApprovalRequest | None:
        return self._requests.get(request_id)

    def save_token(self, token: ApprovalToken) -> None:
        self._tokens[token.token_id] = token

    def get_token(self, token_id: str) -> ApprovalToken | None:
        return self._tokens.get(token_id)


@dataclass(slots=True)
class LoggingEscalationHandler:
    """Escalation handler that records directives in-memory for testing."""

    escalations: list[tuple[EscalationDirective, PolicyContext]] = field(default_factory=list)

    def escalate(
        self,
        directive: EscalationDirective,
        context: PolicyContext,
    ) -> None:
        self.escalations.append((directive, context))
