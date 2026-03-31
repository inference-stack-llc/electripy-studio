"""Tests for policy engine domain models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from electripy.ai.policy import (
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


class TestPolicySubject:
    def test_creation_with_defaults(self) -> None:
        subject = PolicySubject(actor_id="user-1")
        assert subject.actor_id == "user-1"
        assert subject.roles == ()
        assert subject.teams == ()
        assert subject.attributes == ()

    def test_creation_with_all_fields(self) -> None:
        subject = PolicySubject(
            actor_id="user-2",
            roles=("admin", "editor"),
            teams=("engineering",),
            attributes=(("department", "platform"),),
        )
        assert subject.roles == ("admin", "editor")
        assert subject.teams == ("engineering",)

    def test_immutability(self) -> None:
        subject = PolicySubject(actor_id="user-1")
        try:
            subject.actor_id = "changed"  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestPolicyResource:
    def test_creation(self) -> None:
        resource = PolicyResource(resource_type="tool", resource_id="web_search")
        assert resource.resource_type == "tool"
        assert resource.resource_id == "web_search"
        assert resource.attributes == ()


class TestPolicyAction:
    def test_creation(self) -> None:
        action = PolicyAction(action_type="invoke")
        assert action.action_type == "invoke"
        assert action.parameters == ()

    def test_with_parameters(self) -> None:
        action = PolicyAction(
            action_type="write",
            parameters=(("target", "production"),),
        )
        assert action.parameters == (("target", "production"),)


class TestEvidenceItem:
    def test_creation(self) -> None:
        evidence = EvidenceItem(kind=EvidenceKind.JUSTIFICATION, value="Testing in staging")
        assert evidence.kind == EvidenceKind.JUSTIFICATION
        assert evidence.value == "Testing in staging"
        assert evidence.provided_at is not None

    def test_all_evidence_kinds(self) -> None:
        kinds = list(EvidenceKind)
        assert len(kinds) == 5
        assert EvidenceKind.JUSTIFICATION in kinds
        assert EvidenceKind.TICKET_REFERENCE in kinds
        assert EvidenceKind.APPROVAL_TOKEN in kinds
        assert EvidenceKind.MFA_CHALLENGE in kinds
        assert EvidenceKind.AUDIT_LOG in kinds


class TestPolicyContext:
    def test_creation(self) -> None:
        ctx = PolicyContext(
            subject=PolicySubject(actor_id="user-1"),
            resource=PolicyResource(resource_type="tool", resource_id="web_search"),
            action=PolicyAction(action_type="invoke"),
        )
        assert ctx.subject.actor_id == "user-1"
        assert ctx.resource.resource_type == "tool"
        assert ctx.action.action_type == "invoke"
        assert ctx.request_id  # Auto-generated UUID hex
        assert ctx.evidence == ()

    def test_with_evidence(self) -> None:
        evidence = (EvidenceItem(kind=EvidenceKind.JUSTIFICATION, value="need it"),)
        ctx = PolicyContext(
            subject=PolicySubject(actor_id="user-1"),
            resource=PolicyResource(resource_type="tool", resource_id="t"),
            action=PolicyAction(action_type="invoke"),
            evidence=evidence,
        )
        assert len(ctx.evidence) == 1


class TestDecisionOutcome:
    def test_all_outcomes(self) -> None:
        outcomes = list(DecisionOutcome)
        assert len(outcomes) == 5
        assert DecisionOutcome.ALLOW.value == "allow"
        assert DecisionOutcome.DENY.value == "deny"
        assert DecisionOutcome.REDACT.value == "redact"
        assert DecisionOutcome.REQUIRE_APPROVAL.value == "require_approval"
        assert DecisionOutcome.ESCALATE.value == "escalate"


class TestRulePriority:
    def test_all_priorities(self) -> None:
        assert len(list(RulePriority)) == 4
        assert RulePriority.CRITICAL.value == "critical"


class TestPolicyRule:
    def test_basic_rule(self) -> None:
        rule = PolicyRule(
            rule_id="r1",
            description="No delete",
            outcome=DecisionOutcome.DENY,
        )
        assert rule.rule_id == "r1"
        assert rule.priority == RulePriority.MEDIUM
        assert rule.enabled is True
        assert rule.resource_types == ()
        assert rule.action_types == ()
        assert rule.required_roles == ()
        assert rule.required_evidence == ()

    def test_rule_with_all_fields(self) -> None:
        rule = PolicyRule(
            rule_id="r2",
            description="Only admins may invoke tools",
            priority=RulePriority.HIGH,
            outcome=DecisionOutcome.REQUIRE_APPROVAL,
            resource_types=("tool",),
            action_types=("invoke",),
            required_roles=("admin",),
            required_evidence=(EvidenceKind.JUSTIFICATION,),
            redaction=RedactionDirective(field_path="response.body"),
            escalation=EscalationDirective(
                level=EscalationLevel.SECURITY, reason="Tool access"
            ),
            ttl_seconds=3600,
            enabled=True,
        )
        assert rule.required_roles == ("admin",)
        assert rule.redaction is not None
        assert rule.escalation is not None
        assert rule.ttl_seconds == 3600


class TestPolicyPack:
    def test_creation(self) -> None:
        rules = (
            PolicyRule(rule_id="r1", description="Rule 1"),
            PolicyRule(rule_id="r2", description="Rule 2"),
        )
        pack = PolicyPack(
            pack_id="enterprise-v1",
            version="1.0.0",
            rules=rules,
            description="Enterprise rule set",
        )
        assert pack.pack_id == "enterprise-v1"
        assert len(pack.rules) == 2
        assert pack.version == "1.0.0"


class TestRedactionDirective:
    def test_defaults(self) -> None:
        rd = RedactionDirective(field_path="output.text")
        assert rd.field_path == "output.text"
        assert rd.replacement == "[REDACTED]"

    def test_custom_replacement(self) -> None:
        rd = RedactionDirective(field_path="output.text", replacement="***")
        assert rd.replacement == "***"


class TestEscalationDirective:
    def test_creation(self) -> None:
        ed = EscalationDirective(
            level=EscalationLevel.MANAGER,
            reason="High-risk operation",
            notify_channels=("slack:#alerts",),
        )
        assert ed.level == EscalationLevel.MANAGER
        assert ed.reason == "High-risk operation"
        assert ed.notify_channels == ("slack:#alerts",)


class TestPolicyDecision:
    def test_allow_decision(self) -> None:
        d = PolicyDecision(outcome=DecisionOutcome.ALLOW)
        assert not d.blocked
        assert not d.requires_evidence

    def test_deny_decision_is_blocked(self) -> None:
        d = PolicyDecision(outcome=DecisionOutcome.DENY)
        assert d.blocked

    def test_require_approval_is_blocked_and_requires_evidence(self) -> None:
        d = PolicyDecision(outcome=DecisionOutcome.REQUIRE_APPROVAL)
        assert d.blocked
        assert d.requires_evidence

    def test_escalate_is_blocked(self) -> None:
        d = PolicyDecision(outcome=DecisionOutcome.ESCALATE)
        assert d.blocked

    def test_redact_is_not_blocked(self) -> None:
        d = PolicyDecision(outcome=DecisionOutcome.REDACT)
        assert not d.blocked


class TestPolicyViolation:
    def test_creation(self) -> None:
        v = PolicyViolation(
            rule_id="r1",
            priority=RulePriority.HIGH,
            message="Access denied",
        )
        assert v.rule_id == "r1"
        assert v.priority == RulePriority.HIGH


class TestDecisionReason:
    def test_creation(self) -> None:
        r = DecisionReason(rule_id="r1", message="Missing admin role")
        assert r.rule_id == "r1"
        assert r.message == "Missing admin role"


class TestApprovalRequest:
    def test_creation(self) -> None:
        req = ApprovalRequest(
            subject=PolicySubject(actor_id="user-1"),
            resource=PolicyResource(resource_type="tool", resource_id="t"),
            action=PolicyAction(action_type="invoke"),
        )
        assert req.status == ApprovalStatus.PENDING
        assert req.request_id  # Auto-generated
        assert not req.expired

    def test_expired_request(self) -> None:
        req = ApprovalRequest(
            ttl=timedelta(seconds=0),
            created_at=datetime.now(tz=UTC) - timedelta(seconds=1),
        )
        assert req.expired


class TestApprovalToken:
    def test_valid_token(self) -> None:
        token = ApprovalToken(
            request_id="req-1",
            granted_by="approver-1",
            expires_at=datetime.now(tz=UTC) + timedelta(hours=1),
        )
        assert token.valid
        assert token.granted_by == "approver-1"

    def test_expired_token(self) -> None:
        token = ApprovalToken(
            expires_at=datetime.now(tz=UTC) - timedelta(seconds=1),
        )
        assert not token.valid


class TestEscalationLevel:
    def test_all_levels(self) -> None:
        levels = list(EscalationLevel)
        assert len(levels) == 4
        assert EscalationLevel.TEAM_LEAD.value == "team_lead"
        assert EscalationLevel.EXECUTIVE.value == "executive"
