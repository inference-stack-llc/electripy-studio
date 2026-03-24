# Recipe: AI Telemetry

This recipe shows how to wire the AI Telemetry component into an
application that uses HTTP resilience and the LLM Gateway.

## Scenario

You want to:

- Attach correlation IDs to incoming work.
- Emit structured telemetry for HTTP retries, LLM calls, and policy
decisions.
- Export telemetry either to JSONL or to OpenTelemetry.

## JSONL sink example

```python
from pathlib import Path

from electripy.observability.ai_telemetry import (
    JsonlTelemetrySinkAdapter,
    create_telemetry_context,
    inject_context_headers,
    scoped_telemetry_context,
)
from electripy.observability.ai_telemetry.services import (
    record_http_retry_attempt,
    record_llm_call,
    record_policy_decision,
)

telemetry = JsonlTelemetrySinkAdapter(path=Path("telemetry.jsonl"))
ctx = create_telemetry_context(environment="prod")

with scoped_telemetry_context(ctx):
    # Inject correlation headers into an outbound HTTP request
    headers: dict[str, str] = {}
    inject_context_headers(headers)

    # Record an HTTP retry attempt
    record_http_retry_attempt(
        telemetry,
        attempt=1,
        max_attempts=3,
        url="https://api.example.com/resource",
        status_code=503,
        ctx=ctx,
    )

    # After an LLM call completes
    record_llm_call(
        telemetry,
        provider="openai",
        model="gpt-4.1",
        latency_ms=120.0,
        input_tokens=1000,
        output_tokens=256,
        finish_reason="stop",
        structured_output_valid=True,
        ctx=ctx,
    )

    # Record a policy decision
    record_policy_decision(
        telemetry,
        decision="allow",
        violation_codes=[],
        redactions_applied=False,
        ctx=ctx,
    )
```

All sensitive fields (for example `prompt`, `response`) are hashed by
adapter-level sanitisation, so raw content is never written to disk.

## Using AI Telemetry with the LLM Gateway hook

Instead of calling `record_llm_call` manually after every gateway usage,
you can attach a hook to the LLM Gateway so that every successful call
is automatically exported to your telemetry sink.

```python
from pathlib import Path

from electripy.ai.llm_gateway import (
    LlmGatewaySettings,
    LlmMessage,
    LlmRequest,
    build_llm_sync_client,
)
from electripy.observability.ai_telemetry import JsonlTelemetrySinkAdapter
from electripy.observability.ai_telemetry.services import record_llm_call

telemetry = JsonlTelemetrySinkAdapter(path=Path("telemetry.jsonl"))


def llm_telemetry_hook(request: LlmRequest, response: "LlmResponse", latency_ms: float) -> None:
    # Approximate tokens if the provider does not report usage.
    input_tokens = len(" ".join(m.content for m in request.messages)) // 4
    total = response.usage_total_tokens or 0
    output_tokens = max(total - input_tokens, 0)

    record_llm_call(
        telemetry,
        provider="openai",
        model=response.model or request.model,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=response.finish_reason or "unknown",
        structured_output_valid=bool(response.raw_json),
    )


settings = LlmGatewaySettings(on_llm_call=llm_telemetry_hook)
client = build_llm_sync_client("openai", settings=settings)

response = client.complete(
    LlmRequest(
        model="gpt-4o-mini",
        messages=[LlmMessage.user("Hello with telemetry!")],
    )
)
```

This pattern keeps the gateway decoupled from any specific telemetry
backend while still giving you end-to-end latency and token telemetry
for every call.

## OpenTelemetry example

```python
from electripy.observability.ai_telemetry import OpenTelemetryAdapter
from electripy.observability.ai_telemetry.services import record_llm_call

telemetry = OpenTelemetryAdapter(service_name="example-service")

# Inside a traced operation
record_llm_call(
    telemetry,
    provider="openai",
    model="gpt-4.1",
    latency_ms=80.0,
    input_tokens=800,
    output_tokens=200,
    finish_reason="stop",
    structured_output_valid=True,
)
```

The adapter bridges to the active OpenTelemetry span, allowing you to
export events and metrics to your existing observability backend.

## Cost estimation

To estimate cost per call using a simple rate table:

```python
from electripy.observability.ai_telemetry import TableCostEstimator

estimator = TableCostEstimator(rates={("openai", "gpt-4.1"): (0.01, 0.03)})

cost = estimator.estimate_cost(
    provider="openai",
    model="gpt-4.1",
    input_tokens=1000,
    output_tokens=500,
)

print(cost.estimated_cost_usd)
```

Missing entries return `estimated_cost_usd = None` so that cost
estimation never breaks critical paths.
