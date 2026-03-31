"""Tests for the PolicyEngine service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from electripy.ai.policy import (
    ApprovalError,
    DecisionOutcome,
    EscalationDirective,
    EscalationLevel,
    EvidenceError,
    EvidenceItem,
    EvidenceKind,
    InMemoryApprovalStore,
    InMemoryPolicyRepository,
    LoggingEscalationHandler,
    PolicyAction,
    PolicyContext,
    PolicyEngine,
    PolicyEngineSettings,
    PolicyPack,
    PolicyPackError,
    PolicyResource,
    PolicyRule,
    PolicySubject,
    RedactionDirective,
    RulePriority,
)
from electripy.ai.policy.domain import ApprovalRequest, ApprovalStatus


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


def _make_engine(
    rules: list[PolicyRule] | None = None,
    **kwargs: object,
) -> PolicyEngine:
    repo = InMemoryPolicyRepository()
    for rule in rules or []:
        repo.add_rule(rule)
    return PolicyEngine(repository=repo, **kwargs)  # type: ignore[arg-type]


# ── Basic evaluation ─────────────────────────────────────────────────


class TestPolicyEngineEvaluate:
    def test_allow_when_no_rules(self) -> None:
        engine = _make_engine([])
        decision = engine.evaluate(_make_context())
        assert decision.outcome == DecisionOutcome.ALLOW

    def test_deny_when_role_missing(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Admin only",
                required_roles=("admin",),
                outcome=DecisionOutcome.DENY,
            )
        ])
        decision = engine.evaluate(_make_context(roles=("viewer",)))
        assert decision.outcome == DecisionOutcome.DENY
        assert decision.blocked

    def test_allow_when_role_satisfied(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Admin only",
                required_roles=("admin",),
                outcome=DecisionOutcome.DENY,
            )
        ])
        decision = engine.evaluate(_make_context(roles=("admin",)))
        assert decision.outcome == DecisionOutcome.ALLOW

    def test_policy_version_attached(self) -> None:
        engine = _make_engine(
            [],
            settings=PolicyEngineSettings(policy_version="v2.1"),
        )
        decision = engine.evaluate(_make_context())
        assert decision.policy_version == "v2.1"

    def test_request_id_propagated(self) -> None:
        engine = _make_engine([])
        ctx = _make_context()
        decision = engine.evaluate(ctx)
        assert decision.request_id == ctx.request_id


# ── Evidence enforcement ─────────────────────────────────────────────


class TestPolicyEngineEvidence:
    def test_deny_when_evidence_missing(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Need ticket",
                required_evidence=(EvidenceKind.TICKET_REFERENCE,),
                outcome=DecisionOutcome.DENY,
            )
        ])
        decision = engine.evaluate(_make_context())
        assert decision.outcome == DecisionOutcome.DENY

    def test_allow_when_evidence_provided(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Need ticket",
                required_evidence=(EvidenceKind.TICKET_REFERENCE,),
                outcome=DecisionOutcome.DENY,
            )
        ])
        ctx = _make_context(
            evidence=(EvidenceItem(kind=EvidenceKind.TICKET_REFERENCE, value="JIRA-123"),)
        )
        decision = engine.evaluate(ctx)
        assert decision.outcome == DecisionOutcome.ALLOW

    def test_multiple_evidence_requirements(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Need ticket + justification",
                required_evidence=(
                    EvidenceKind.TICKET_REFERENCE,
                    EvidenceKind.JUSTIFICATION,
                ),
                outcome=DecisionOutcome.DENY,
            )
        ])
        # Only ticket, missing justification.
        ctx = _make_context(
            evidence=(EvidenceItem(kind=EvidenceKind.TICKET_REFERENCE, value="J-1"),)
        )
        decision = engine.evaluate(ctx)
        assert decision.outcome == DecisionOutcome.DENY


# ── Escalation ───────────────────────────────────────────────────────


class TestPolicyEngineEscalation:
    def test_escalation_handler_called(self) -> None:
        handler = LoggingEscalationHandler()
        engine = _make_engine(
            [
                PolicyRule(
                    rule_id="r1",
                    description="Escalate",
                    required_roles=("security",),
                    outcome=DecisionOutcome.ESCALATE,
                    escalation=EscalationDirective(
                        level=EscalationLevel.SECURITY,
                        reason="Sensitive op",
                    ),
                )
            ],
            escalation_handler=handler,
        )
        decision = engine.evaluate(_make_context(roles=("viewer",)))
        assert decision.outcome == DecisionOutcome.ESCALATE
        assert len(handler.escalations) == 1

    def test_no_escalation_handler_is_safe(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Escalate",
                required_roles=("security",),
                outcome=DecisionOutcome.ESCALATE,
                escalation=EscalationDirective(
                    level=EscalationLevel.SECURITY, reason="Test"
                ),
            )
        ])
        # Should not raise even without handler.
        decision = engine.evaluate(_make_context(roles=("viewer",)))
        assert decision.outcome == DecisionOutcome.ESCALATE


# ── Redaction directives ─────────────────────────────────────────────


class TestPolicyEngineRedaction:
    def test_redaction_directives_returned(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Redact response",
                required_roles=("privileged",),
                outcome=DecisionOutcome.REDACT,
                redaction=RedactionDirective(field_path="response.body"),
            )
        ])
        decision = engine.evaluate(_make_context(roles=("viewer",)))
        assert decision.outcome == DecisionOutcome.REDACT
        assert len(decision.redactions) == 1
        assert decision.redactions[0].field_path == "response.body"


# ── Observer hook ────────────────────────────────────────────────────


class TestPolicyEngineObserver:
    def test_observer_called_on_decision(self) -> None:
        observations: list[tuple] = []

        class TestObserver:
            def on_decision(self, context, decision):
                observations.append((context, decision))

        engine = _make_engine([], observer=TestObserver())  # type: ignore[arg-type]
        ctx = _make_context()
        engine.evaluate(ctx)
        assert len(observations) == 1
        assert observations[0][0] is ctx


# ── Approval workflow ────────────────────────────────────────────────


class TestPolicyEngineApproval:
    def test_request_approval_creates_request(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Approval needed",
                required_roles=("admin",),
                outcome=DecisionOutcome.REQUIRE_APPROVAL,
                required_evidence=(EvidenceKind.JUSTIFICATION,),
                ttl_seconds=600,
            )
        ])
        approval = engine.request_approval(_make_context(roles=("viewer",)))
        assert approval.status == ApprovalStatus.PENDING
        assert EvidenceKind.JUSTIFICATION in approval.required_evidence
        assert approval.ttl == timedelta(seconds=600)

    def test_request_approval_fails_when_not_required(self) -> None:
        engine = _make_engine([])
        with pytest.raises(ApprovalError, match="not require_approval"):
            engine.request_approval(_make_context())

    def test_grant_approval_produces_token(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Needs approval",
                required_roles=("admin",),
                outcome=DecisionOutcome.REQUIRE_APPROVAL,
            )
        ])
        approval = engine.request_approval(_make_context(roles=("viewer",)))
        token = engine.grant_approval(
            approval.request_id,
            granted_by="manager-1",
        )
        assert token.granted_by == "manager-1"
        assert token.valid

    def test_grant_approval_with_evidence(self) -> None:
        engine = _make_engine(
            [
                PolicyRule(
                    rule_id="r1",
                    description="Needs approval + evidence",
                    required_roles=("admin",),
                    outcome=DecisionOutcome.REQUIRE_APPROVAL,
                    required_evidence=(EvidenceKind.JUSTIFICATION,),
                )
            ],
            settings=PolicyEngineSettings(enforce_evidence=True),
        )
        approval = engine.request_approval(_make_context(roles=("viewer",)))

        # Missing evidence should raise.
        with pytest.raises(EvidenceError, match="Missing required evidence"):
            engine.grant_approval(approval.request_id, granted_by="mgr")

        # With evidence should succeed.
        token = engine.grant_approval(
            approval.request_id,
            granted_by="mgr",
            evidence=(EvidenceItem(kind=EvidenceKind.JUSTIFICATION, value="approved"),),
        )
        assert token.valid

    def test_grant_approval_missing_request_raises(self) -> None:
        engine = _make_engine([])
        with pytest.raises(ApprovalError, match="not found"):
            engine.grant_approval("nonexistent", granted_by="mgr")

    def test_validate_token_success(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Needs approval",
                required_roles=("admin",),
                outcome=DecisionOutcome.REQUIRE_APPROVAL,
            )
        ])
        approval = engine.request_approval(_make_context(roles=("viewer",)))
        token = engine.grant_approval(approval.request_id, granted_by="mgr")
        validated = engine.validate_token(token.token_id)
        assert validated.token_id == token.token_id

    def test_validate_expired_token_raises(self) -> None:
        engine = _make_engine([])
        store = engine.approval_store
        from electripy.ai.policy.domain import ApprovalToken

        expired_token = ApprovalToken(
            token_id="tok-expired",
            expires_at=datetime.now(tz=UTC) - timedelta(seconds=1),
        )
        store.save_token(expired_token)  # type: ignore[union-attr]
        with pytest.raises(ApprovalError, match="expired"):
            engine.validate_token("tok-expired")

    def test_validate_missing_token_raises(self) -> None:
        engine = _make_engine([])
        with pytest.raises(ApprovalError, match="not found"):
            engine.validate_token("nonexistent")


# ── Policy packs ─────────────────────────────────────────────────────


class TestPolicyEnginePacks:
    def test_load_pack_adds_rules(self) -> None:
        engine = _make_engine([])
        pack = PolicyPack(
            pack_id="enterprise",
            version="1.0.0",
            rules=(
                PolicyRule(rule_id="p1", description="Rule A"),
                PolicyRule(rule_id="p2", description="Rule B"),
            ),
        )
        engine.load_pack(pack)
        rules = engine.repository.list_rules()
        assert len(rules) == 2

    def test_load_empty_pack_raises(self) -> None:
        engine = _make_engine([])
        pack = PolicyPack(pack_id="empty", version="1.0.0", rules=())
        with pytest.raises(PolicyPackError, match="no rules"):
            engine.load_pack(pack)

    def test_load_pack_with_duplicate_ids_raises(self) -> None:
        engine = _make_engine([])
        pack = PolicyPack(
            pack_id="dup",
            version="1.0.0",
            rules=(
                PolicyRule(rule_id="same", description="A"),
                PolicyRule(rule_id="same", description="B"),
            ),
        )
        with pytest.raises(PolicyPackError, match="duplicate"):
            engine.load_pack(pack)

    def test_load_pack_conflicting_with_existing_raises(self) -> None:
        engine = _make_engine([PolicyRule(rule_id="existing", description="Already here")])
        pack = PolicyPack(
            pack_id="conflict",
            version="1.0.0",
            rules=(PolicyRule(rule_id="existing", description="Conflict"),),
        )
        with pytest.raises(PolicyPackError, match="conflicts"):
            engine.load_pack(pack)


# ── Determinism ──────────────────────────────────────────────────────


class TestPolicyEngineDeterminism:
    def test_same_input_same_output(self) -> None:
        rules = [
            PolicyRule(
                rule_id="r1",
                description="Admin only",
                required_roles=("admin",),
                outcome=DecisionOutcome.DENY,
            )
        ]
        engine = _make_engine(rules)
        ctx = _make_context(roles=("viewer",))
        d1 = engine.evaluate(ctx)
        d2 = engine.evaluate(ctx)
        assert d1.outcome == d2.outcome
        assert d1.violations == d2.violations
        assert d1.reasons == d2.reasons

    def test_evaluation_is_idempotent(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Redact",
                required_roles=("privileged",),
                outcome=DecisionOutcome.REDACT,
                redaction=RedactionDirective(field_path="out"),
            )
        ])
        ctx = _make_context(roles=("viewer",))
        results = [engine.evaluate(ctx) for _ in range(5)]
        outcomes = {r.outcome for r in results}
        assert outcomes == {DecisionOutcome.REDACT}


# ── TTL on approval ──────────────────────────────────────────────────


class TestPolicyEngineApprovalTTL:
    def test_ttl_from_rule(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Short TTL",
                required_roles=("admin",),
                outcome=DecisionOutcome.REQUIRE_APPROVAL,
                ttl_seconds=120,
            )
        ])
        approval = engine.request_approval(_make_context(roles=("viewer",)))
        assert approval.ttl == timedelta(seconds=120)

    def test_default_ttl_used_when_rule_has_none(self) -> None:
        engine = _make_engine(
            [
                PolicyRule(
                    rule_id="r1",
                    description="Default TTL",
                    required_roles=("admin",),
                    outcome=DecisionOutcome.REQUIRE_APPROVAL,
                )
            ],
            settings=PolicyEngineSettings(default_approval_ttl=timedelta(minutes=30)),
        )
        approval = engine.request_approval(_make_context(roles=("viewer",)))
        assert approval.ttl == timedelta(minutes=30)

    def test_shortest_ttl_wins(self) -> None:
        engine = _make_engine([
            PolicyRule(
                rule_id="r1",
                description="Long",
                required_roles=("admin",),
                outcome=DecisionOutcome.REQUIRE_APPROVAL,
                ttl_seconds=7200,
            ),
            PolicyRule(
                rule_id="r2",
                description="Short",
                required_roles=("admin",),
                outcome=DecisionOutcome.REQUIRE_APPROVAL,
                ttl_seconds=300,
            ),
        ])
        approval = engine.request_approval(_make_context(roles=("viewer",)))
        assert approval.ttl == timedelta(seconds=300)
