"""Tests for policy engine adapters."""

from __future__ import annotations

from electripy.ai.policy import (
    DecisionOutcome,
    EscalationDirective,
    EscalationLevel,
    EvidenceItem,
    EvidenceKind,
    PolicyAction,
    PolicyContext,
    PolicyResource,
    PolicyRule,
    PolicySubject,
    PolicyViolation,
    RedactionDirective,
    RulePriority,
)
from electripy.ai.policy.adapters import (
    DefaultPolicyEvaluator,
    InMemoryApprovalStore,
    InMemoryPolicyRepository,
    LoggingEscalationHandler,
)
from electripy.ai.policy.domain import ApprovalRequest, ApprovalToken


# ── Helpers ──────────────────────────────────────────────────────────


def _make_context(
    *,
    actor_id: str = "user-1",
    roles: tuple[str, ...] = (),
    resource_type: str = "tool",
    resource_id: str = "web_search",
    action_type: str = "invoke",
    evidence: tuple[EvidenceItem, ...] = (),
) -> PolicyContext:
    return PolicyContext(
        subject=PolicySubject(actor_id=actor_id, roles=roles),
        resource=PolicyResource(resource_type=resource_type, resource_id=resource_id),
        action=PolicyAction(action_type=action_type),
        evidence=evidence,
    )


# ── DefaultPolicyEvaluator ──────────────────────────────────────────


class TestDefaultPolicyEvaluatorAllow:
    def test_no_rules_returns_allow(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        decision = evaluator.evaluate(_make_context(), [])
        assert decision.outcome == DecisionOutcome.ALLOW

    def test_no_matching_rules_returns_allow(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rule = PolicyRule(
            rule_id="r1",
            description="Only applies to models",
            resource_types=("model",),
        )
        decision = evaluator.evaluate(_make_context(resource_type="tool"), [rule])
        assert decision.outcome == DecisionOutcome.ALLOW

    def test_disabled_rule_is_skipped(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rule = PolicyRule(rule_id="r1", description="Disabled", enabled=False)
        decision = evaluator.evaluate(_make_context(), [rule])
        assert decision.outcome == DecisionOutcome.ALLOW

    def test_role_satisfied_returns_allow(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rule = PolicyRule(
            rule_id="r1",
            description="Admins only",
            required_roles=("admin",),
            outcome=DecisionOutcome.DENY,
        )
        ctx = _make_context(roles=("admin",))
        decision = evaluator.evaluate(ctx, [rule])
        assert decision.outcome == DecisionOutcome.ALLOW

    def test_evidence_satisfied_returns_allow(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rule = PolicyRule(
            rule_id="r1",
            description="Need justification",
            required_evidence=(EvidenceKind.JUSTIFICATION,),
            outcome=DecisionOutcome.DENY,
        )
        ctx = _make_context(
            evidence=(EvidenceItem(kind=EvidenceKind.JUSTIFICATION, value="testing"),)
        )
        decision = evaluator.evaluate(ctx, [rule])
        assert decision.outcome == DecisionOutcome.ALLOW


class TestDefaultPolicyEvaluatorDeny:
    def test_missing_role_triggers_deny(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rule = PolicyRule(
            rule_id="r1",
            description="Admins only",
            required_roles=("admin",),
            outcome=DecisionOutcome.DENY,
        )
        ctx = _make_context(roles=("viewer",))
        decision = evaluator.evaluate(ctx, [rule])
        assert decision.outcome == DecisionOutcome.DENY
        assert len(decision.violations) == 1
        assert decision.violations[0].rule_id == "r1"

    def test_missing_evidence_triggers_deny(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rule = PolicyRule(
            rule_id="r1",
            description="Need justification",
            required_evidence=(EvidenceKind.JUSTIFICATION,),
            outcome=DecisionOutcome.DENY,
        )
        ctx = _make_context()
        decision = evaluator.evaluate(ctx, [rule])
        assert decision.outcome == DecisionOutcome.DENY

    def test_deny_takes_precedence_over_require_approval(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rules = [
            PolicyRule(
                rule_id="r1",
                description="Deny rule",
                required_roles=("superadmin",),
                outcome=DecisionOutcome.DENY,
            ),
            PolicyRule(
                rule_id="r2",
                description="Approval rule",
                required_roles=("approver",),
                outcome=DecisionOutcome.REQUIRE_APPROVAL,
            ),
        ]
        ctx = _make_context(roles=("viewer",))
        decision = evaluator.evaluate(ctx, rules)
        assert decision.outcome == DecisionOutcome.DENY


class TestDefaultPolicyEvaluatorRequireApproval:
    def test_require_approval_outcome(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rule = PolicyRule(
            rule_id="r1",
            description="Needs approval",
            required_roles=("admin",),
            outcome=DecisionOutcome.REQUIRE_APPROVAL,
        )
        ctx = _make_context(roles=("viewer",))
        decision = evaluator.evaluate(ctx, [rule])
        assert decision.outcome == DecisionOutcome.REQUIRE_APPROVAL
        assert decision.blocked


class TestDefaultPolicyEvaluatorEscalate:
    def test_escalate_outcome_with_directive(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rule = PolicyRule(
            rule_id="r1",
            description="High risk",
            required_roles=("security-team",),
            outcome=DecisionOutcome.ESCALATE,
            escalation=EscalationDirective(
                level=EscalationLevel.SECURITY,
                reason="Sensitive operation",
            ),
        )
        ctx = _make_context(roles=("viewer",))
        decision = evaluator.evaluate(ctx, [rule])
        assert decision.outcome == DecisionOutcome.ESCALATE
        assert decision.escalation is not None
        assert decision.escalation.level == EscalationLevel.SECURITY


class TestDefaultPolicyEvaluatorRedact:
    def test_redact_outcome_with_directive(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rule = PolicyRule(
            rule_id="r1",
            description="Redact output",
            required_roles=("privileged",),
            outcome=DecisionOutcome.REDACT,
            redaction=RedactionDirective(field_path="response.body"),
        )
        ctx = _make_context(roles=("viewer",))
        decision = evaluator.evaluate(ctx, [rule])
        assert decision.outcome == DecisionOutcome.REDACT
        assert len(decision.redactions) == 1
        assert decision.redactions[0].field_path == "response.body"


class TestDefaultPolicyEvaluatorMultipleRules:
    def test_violations_from_multiple_rules(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rules = [
            PolicyRule(
                rule_id="r1",
                description="Role check",
                required_roles=("admin",),
                outcome=DecisionOutcome.REQUIRE_APPROVAL,
            ),
            PolicyRule(
                rule_id="r2",
                description="Evidence check",
                required_evidence=(EvidenceKind.TICKET_REFERENCE,),
                outcome=DecisionOutcome.REQUIRE_APPROVAL,
            ),
        ]
        ctx = _make_context(roles=("viewer",))
        decision = evaluator.evaluate(ctx, rules)
        assert decision.outcome == DecisionOutcome.REQUIRE_APPROVAL
        assert len(decision.violations) == 2

    def test_action_type_filtering(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rule = PolicyRule(
            rule_id="r1",
            description="Only for delete",
            action_types=("delete",),
            required_roles=("admin",),
            outcome=DecisionOutcome.DENY,
        )
        ctx = _make_context(action_type="invoke")
        decision = evaluator.evaluate(ctx, [rule])
        assert decision.outcome == DecisionOutcome.ALLOW

    def test_priority_ordering(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rules = [
            PolicyRule(
                rule_id="low",
                description="Low priority",
                priority=RulePriority.LOW,
                required_roles=("admin",),
                outcome=DecisionOutcome.DENY,
            ),
            PolicyRule(
                rule_id="critical",
                description="Critical",
                priority=RulePriority.CRITICAL,
                required_roles=("admin",),
                outcome=DecisionOutcome.DENY,
            ),
        ]
        ctx = _make_context(roles=("viewer",))
        decision = evaluator.evaluate(ctx, rules)
        # Both violated, but critical should appear first in violations.
        assert decision.violations[0].rule_id == "critical"

    def test_reasons_contain_violation_messages(self) -> None:
        evaluator = DefaultPolicyEvaluator()
        rule = PolicyRule(
            rule_id="r1",
            description="Must be admin",
            required_roles=("admin",),
            outcome=DecisionOutcome.DENY,
        )
        ctx = _make_context(roles=("viewer",))
        decision = evaluator.evaluate(ctx, [rule])
        assert len(decision.reasons) == 1
        assert "admin" in decision.reasons[0].message


# ── InMemoryPolicyRepository ────────────────────────────────────────


class TestInMemoryPolicyRepository:
    def test_add_and_list(self) -> None:
        repo = InMemoryPolicyRepository()
        rule = PolicyRule(rule_id="r1", description="Test")
        repo.add_rule(rule)
        assert len(repo.list_rules()) == 1

    def test_filter_by_resource_type(self) -> None:
        repo = InMemoryPolicyRepository()
        repo.add_rule(PolicyRule(rule_id="r1", description="A", resource_types=("tool",)))
        repo.add_rule(PolicyRule(rule_id="r2", description="B", resource_types=("model",)))
        assert len(repo.list_rules(resource_type="tool")) == 1

    def test_disabled_rules_excluded(self) -> None:
        repo = InMemoryPolicyRepository()
        repo.add_rule(PolicyRule(rule_id="r1", description="Active"))
        repo.add_rule(PolicyRule(rule_id="r2", description="Off", enabled=False))
        assert len(repo.list_rules()) == 1

    def test_empty_resource_types_matches_all(self) -> None:
        repo = InMemoryPolicyRepository()
        repo.add_rule(PolicyRule(rule_id="r1", description="Universal"))
        assert len(repo.list_rules(resource_type="anything")) == 1


# ── InMemoryApprovalStore ───────────────────────────────────────────


class TestInMemoryApprovalStore:
    def test_save_and_get_request(self) -> None:
        store = InMemoryApprovalStore()
        req = ApprovalRequest(request_id="req-1")
        store.save_request(req)
        assert store.get_request("req-1") is req

    def test_get_missing_request_returns_none(self) -> None:
        store = InMemoryApprovalStore()
        assert store.get_request("nope") is None

    def test_save_and_get_token(self) -> None:
        store = InMemoryApprovalStore()
        token = ApprovalToken(token_id="tok-1")
        store.save_token(token)
        assert store.get_token("tok-1") is token

    def test_get_missing_token_returns_none(self) -> None:
        store = InMemoryApprovalStore()
        assert store.get_token("nope") is None


# ── LoggingEscalationHandler ────────────────────────────────────────


class TestLoggingEscalationHandler:
    def test_records_escalation(self) -> None:
        handler = LoggingEscalationHandler()
        directive = EscalationDirective(
            level=EscalationLevel.MANAGER, reason="Test"
        )
        ctx = _make_context()
        handler.escalate(directive, ctx)
        assert len(handler.escalations) == 1
        assert handler.escalations[0][0].level == EscalationLevel.MANAGER
