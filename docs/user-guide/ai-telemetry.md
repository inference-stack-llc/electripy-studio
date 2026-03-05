# AI Telemetry

The **AI Telemetry** component provides a provider-agnostic, framework-
agnostic way to instrument and observe:

- HTTP resilience behavior (retries, backoff, circuit breaker events)
- LLM gateway calls (latency, tokens, structured-output success/failure)
- Policy gateway decisions (allow/deny/sanitize, violation codes,
  redactions)
- RAG evaluation runs (experiment IDs and metric summaries)

It is **safe by default**: telemetry focuses on correlation identifiers,
metrics, and hashes rather than raw prompts or model responses.

## Key concepts

- `TelemetryContext`: Correlation identifiers (trace ID, span ID,
  request ID, tenant, environment, tags) with helpers to propagate
  headers (`X-Request-Id`, `Traceparent`).
- `TelemetryEvent`: Structured events with severity, attributes, and
  context.
- `TelemetryPort`: Port interface implemented by telemetry adapters
  (JSONL sink, in-memory, optional OpenTelemetry bridge).
- `CostEstimatorPort` / `TableCostEstimator`: Best-effort cost
  estimation based on `(provider, model)` token pricing tables.

All adapters are designed to be backend-agnostic so you can wire them
into your existing logging/observability stack.

## Safe-by-default behavior

- No prompts or responses are logged by default.
- Attributes with keys like `prompt` or `response` are replaced with
  SHA-256 hashes (for example `prompt_hash`).
- Telemetry focuses on counts, durations, status codes, experiment
  identifiers, and policy outcomes.

## Creating and propagating context

Create a root context with `create_telemetry_context` and make it
current for a block of work:

```python
from pathlib import Path

from electripy.observability.ai_telemetry import (
    JsonlTelemetrySinkAdapter,
    create_telemetry_context,
    scoped_telemetry_context,
)

ctx = create_telemetry_context(environment="dev")
sink = JsonlTelemetrySinkAdapter(path=Path("telemetry.jsonl"))

with scoped_telemetry_context(ctx):
    # downstream code can use current_telemetry_context()
    ...
```

To propagate correlation IDs into outbound HTTP calls, use
`inject_context_headers` with a mutable header mapping.

## Instrumenting LLM gateway and policy flows

LLM gateway example:

```python
from electripy.observability.ai_telemetry import (
    InMemoryTelemetryAdapter,
    create_telemetry_context,
)
from electripy.observability.ai_telemetry.services import record_llm_call

telemetry = InMemoryTelemetryAdapter()
ctx = create_telemetry_context(environment="dev")

# After a gateway call completes
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
```

Policy gateway example:

```python
from pathlib import Path

from electripy.observability.ai_telemetry import (
    JsonlTelemetrySinkAdapter,
    create_telemetry_context,
)
from electripy.observability.ai_telemetry.services import record_policy_decision

sink = JsonlTelemetrySinkAdapter(path=Path("telemetry.jsonl"))
ctx = create_telemetry_context(environment="prod")

record_policy_decision(
    sink,
    decision="allow",
    violation_codes=[],
    redactions_applied=False,
    ctx=ctx,
)
```

## RAG evaluation telemetry

The RAG Evaluation Runner can emit high-level events about experiments.
For example, you can call helpers like
`record_rag_experiment_started` and `record_rag_experiment_finished`
from your orchestration layer to track experiment IDs and metric
summaries in the same telemetry stream as LLM and policy events.

See also: the component-level README at
`src/electripy/observability/ai_telemetry/README.md` for additional
examples.
