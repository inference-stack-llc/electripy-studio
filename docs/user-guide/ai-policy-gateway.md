# AI Policy Gateway

The AI Policy Gateway provides deterministic in-process guardrails for request/response, stream, and tool-call paths.

## Why it exists

Enterprise AI workflows need stable, auditable policy decisions without forcing teams into hosted control-plane products. This component provides local policy enforcement primitives that you can embed into CLI jobs, APIs, and agent runtimes.

## Decision model

A policy evaluation returns one action:

- `allow`
- `sanitize`
- `deny`
- `require_approval`

These are represented as `PolicyAction` values and are deterministic for the same input and rule set.

## Stages

Rules target explicit stages:

- `preflight`
- `postflight`
- `stream`
- `tool_call`

This makes policy behavior composable and easier to reason about.

## Basic usage

```python
from electripy.ai.policy_gateway import (
    PolicyAction,
    PolicyGateway,
    PolicyRule,
    PolicyStage,
)

gateway = PolicyGateway(
    rules=[
        PolicyRule(
            rule_id="pii-email",
            code="PII_EMAIL",
            description="Mask emails in prompts.",
            stage=PolicyStage.PREFLIGHT,
            pattern=r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+",
            action=PolicyAction.SANITIZE,
        )
    ]
)

decision = gateway.evaluate_preflight("Reach me at admin@example.com")
if decision.action == PolicyAction.SANITIZE:
    prompt = decision.sanitized_text or ""
```

## LLM + Tool hooks

Use helpers for clean wiring:

- `before_llm_request(...)`
- `after_llm_response(...)`
- `on_stream_chunk(...)`
- `authorize_tool_call(...)`

Each helper returns a typed `PolicyDecision` so callers can fail closed.

## LLM Gateway integration hooks

Use `build_llm_policy_hooks(...)` to plug policy checks directly into
`LlmGatewaySettings.request_hook` and `LlmGatewaySettings.response_hook`.

```python
from electripy.ai.llm_gateway import LlmGatewaySettings
from electripy.ai.policy_gateway import PolicyGateway, build_llm_policy_hooks

policy = PolicyGateway(rules=[...])
request_hook, response_hook = build_llm_policy_hooks(policy)

settings = LlmGatewaySettings(
    request_hook=request_hook,
    response_hook=response_hook,
)
```

This gives you deterministic policy enforcement before and after LLM
provider calls with a single reusable bridge.
