# Recipe: Policy-Governed LLM + Tool Flow

This recipe shows how to enforce deterministic policy decisions around LLM calls and tool invocations.

## Scenario

You want to:

- Sanitize PII from user prompts.
- Require approval for high-risk tool calls.
- Deny responses containing restricted markers.

## Example

```python
from electripy.ai.policy_gateway import (
    PolicyAction,
    PolicyGateway,
    PolicyRule,
    PolicySeverity,
    PolicyStage,
    after_llm_response,
    authorize_tool_call,
    before_llm_request,
)

gateway = PolicyGateway(
    rules=[
        PolicyRule(
            rule_id="pii-email",
            code="PII_EMAIL",
            description="Mask emails in inbound prompts.",
            stage=PolicyStage.PREFLIGHT,
            pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+",
            action=PolicyAction.SANITIZE,
        ),
        PolicyRule(
            rule_id="tool-delete",
            code="TOOL_DELETE",
            description="Delete operations require approval.",
            stage=PolicyStage.TOOL_CALL,
            pattern=r"drop|delete",
            action=PolicyAction.REQUIRE_APPROVAL,
            severity=PolicySeverity.HIGH,
        ),
        PolicyRule(
            rule_id="secret-leak",
            code="SECRET_LEAK",
            description="Block secret markers in output.",
            stage=PolicyStage.POSTFLIGHT,
            pattern=r"SECRET_[A-Z0-9]+",
            action=PolicyAction.DENY,
        ),
    ]
)

request_decision = before_llm_request(gateway, "Email me at admin@example.com")
if request_decision.action == PolicyAction.SANITIZE:
    prompt = request_decision.sanitized_text or ""
elif request_decision.blocked:
    raise RuntimeError("Prompt blocked by policy")
else:
    prompt = "Email me at admin@example.com"

tool_decision = authorize_tool_call(gateway, "db.execute", {"sql": "drop table users"})
if tool_decision.blocked:
    raise RuntimeError("Tool call blocked or requires approval")

response_decision = after_llm_response(gateway, "ok")
if response_decision.blocked:
    raise RuntimeError("Response blocked")
```

## Notes

- Keep rules versioned and reviewable.
- Start with redaction and explicit deny lists.
- Integrate telemetry for audit trails.
