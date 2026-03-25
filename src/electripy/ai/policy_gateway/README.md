# Policy Gateway

Deterministic in-process policy enforcement for AI request/response flows.

## What it solves

- Standardizes preflight, postflight, streaming, and tool-call policy checks.
- Returns stable decisions (`allow`, `sanitize`, `deny`, `require_approval`).
- Works with existing ElectriPy telemetry and agent tooling.
- Stays library-first (no hosted control plane behavior).

## Quick Start

```python
from electripy.ai.policy_gateway import (
    PolicyAction,
    PolicyGateway,
    PolicyRule,
    PolicyStage,
)

rules = [
    PolicyRule(
        rule_id="pii-email",
        code="PII_EMAIL",
        description="Email addresses must be redacted.",
        stage=PolicyStage.PREFLIGHT,
        pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+",
        action=PolicyAction.SANITIZE,
    )
]

gateway = PolicyGateway(rules=rules)
decision = gateway.evaluate_preflight("Contact me at user@example.com")

if decision.action == PolicyAction.SANITIZE:
    safe_prompt = decision.sanitized_text or ""
```

## Hook Helpers

- `before_llm_request(...)`
- `after_llm_response(...)`
- `on_stream_chunk(...)`
- `authorize_tool_call(...)`

Each helper returns a typed `PolicyDecision` so callers can enforce fail-closed behavior.
