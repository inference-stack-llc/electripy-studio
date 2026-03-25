from __future__ import annotations

import pytest

from electripy.ai.llm_gateway import LlmMessage, LlmRequest, LlmResponse
from electripy.ai.llm_gateway.errors import PolicyViolationError
from electripy.ai.policy_gateway import (
    PolicyAction,
    PolicyGateway,
    PolicyRule,
    PolicySeverity,
    PolicyStage,
    build_llm_policy_hooks,
)


def test_build_llm_policy_hooks_sanitizes_request_text() -> None:
    gateway = PolicyGateway(
        rules=[
            PolicyRule(
                rule_id="email",
                code="PII_EMAIL",
                description="Mask email",
                stage=PolicyStage.PREFLIGHT,
                pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+",
                action=PolicyAction.SANITIZE,
                severity=PolicySeverity.MEDIUM,
            )
        ]
    )

    request_hook, _ = build_llm_policy_hooks(gateway)
    request = LlmRequest(
        model="gpt-test",
        messages=[LlmMessage.user("send to admin@example.com")],
    )

    sanitized = request_hook(request)

    assert sanitized.messages[0].content == "send to [REDACTED]"


def test_build_llm_policy_hooks_blocks_postflight_response() -> None:
    gateway = PolicyGateway(
        rules=[
            PolicyRule(
                rule_id="secret",
                code="SECRET_LEAK",
                description="Block leaked secrets",
                stage=PolicyStage.POSTFLIGHT,
                pattern=r"SECRET_[A-Z0-9]+",
                action=PolicyAction.DENY,
            )
        ]
    )

    request_hook, response_hook = build_llm_policy_hooks(gateway)
    request = request_hook(LlmRequest(model="gpt-test", messages=[LlmMessage.user("hello")]))

    with pytest.raises(PolicyViolationError):
        response_hook(request, LlmResponse(text="SECRET_ABC should not pass"))
