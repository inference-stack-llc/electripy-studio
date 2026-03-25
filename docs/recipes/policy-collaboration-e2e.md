# Recipe: Policy + Collaboration End-to-End

This recipe demonstrates a complete local flow that combines:

- LLM Gateway request/response policy hooks
- deterministic policy decisions
- bounded agent collaboration
- telemetry event capture

## Scenario

You want one run that proves policy, orchestration, and observability work together without network dependencies.

## Run the demo script

```bash
python recipes/03_policy_collaboration/run_demo.py
```

## Expected behavior

- inbound prompt content is evaluated in policy preflight
- sensitive prompt fragments can be sanitized by request hooks
- postflight checks run on model output before downstream usage
- collaboration runtime executes bounded handoffs with deterministic results
- policy decisions and outcomes are observable through telemetry

## Key wiring

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
