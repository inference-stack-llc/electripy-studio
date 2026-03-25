from __future__ import annotations

from electripy.ai.policy_gateway import (
    PolicyAction,
    PolicyGateway,
    PolicyRule,
    PolicySeverity,
    PolicyStage,
)


def test_policy_gateway_sanitizes_preflight_text() -> None:
    gateway = PolicyGateway(
        rules=[
            PolicyRule(
                rule_id="email-redact",
                code="PII_EMAIL",
                description="Email must be redacted",
                stage=PolicyStage.PREFLIGHT,
                pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+",
                action=PolicyAction.SANITIZE,
                severity=PolicySeverity.MEDIUM,
            )
        ]
    )

    decision = gateway.evaluate_preflight("Contact admin@example.com")

    assert decision.action == PolicyAction.SANITIZE
    assert decision.sanitized_text == "Contact [REDACTED]"
    assert decision.reason_codes == ["PII_EMAIL"]


def test_policy_gateway_denies_critical_findings() -> None:
    gateway = PolicyGateway(
        rules=[
            PolicyRule(
                rule_id="secrets",
                code="SECRET_TOKEN",
                description="Secret token marker",
                stage=PolicyStage.POSTFLIGHT,
                pattern=r"SECRET_[A-Z0-9]+",
                severity=PolicySeverity.CRITICAL,
                action=PolicyAction.SANITIZE,
            )
        ]
    )

    decision = gateway.evaluate_postflight("Do not expose SECRET_ABC123")

    assert decision.action == PolicyAction.DENY
    assert decision.blocked is True


def test_policy_gateway_tool_call_requires_approval_on_high() -> None:
    gateway = PolicyGateway(
        rules=[
            PolicyRule(
                rule_id="risky-delete",
                code="TOOL_DELETE",
                description="Delete operations require approval",
                stage=PolicyStage.TOOL_CALL,
                pattern=r"delete|drop",
                action=PolicyAction.SANITIZE,
                severity=PolicySeverity.HIGH,
            )
        ]
    )

    decision = gateway.evaluate_tool_call("db.execute", {"sql": "drop table users"})

    assert decision.action == PolicyAction.REQUIRE_APPROVAL
    assert decision.requires_approval is True
