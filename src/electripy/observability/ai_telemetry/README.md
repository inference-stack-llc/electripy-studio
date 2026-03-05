# AI Telemetry

The AI Telemetry component provides a provider-agnostic, framework-
agnostic way to instrument and observe:

- HTTP resilience behavior (retries, backoff, circuit breaker events)
- LLM gateway calls (latency, tokens, structured-output success/failure)
- Policy gateway decisions (allow/deny/sanitize, violation codes,
  redactions)
- RAG evaluation runs (experiment IDs and metric summaries)

It is **safe by default**: telemetry focuses on correlation identifiers,
metrics, and hashes rather than raw prompts or model responses.

## Why it exists

Enterprise AI systems require consistent, exportable telemetry to:

- Correlate calls across services and CLI tools
- Monitor reliability of downstream providers and HTTP dependencies
- Track usage and cost per provider/model/tenant
- Audit policy gateway decisions without leaking sensitive content

This component offers small, typed primitives and adapters that can be
wired into existing stacks (JSONL, OpenTelemetry) while keeping
ElectriPy components decoupled from any specific telemetry backend.

## Safe-by-default behavior

- No prompts or responses are logged by default.
- Attributes with keys like `prompt` or `response` are replaced with
  SHA-256 hashes (for example `prompt_hash`).
- Telemetry focuses on counts, durations, status codes, and experiment
  identifiers.

## Creating and propagating context

Use `create_telemetry_context` to create a root context and
`scoped_telemetry_context` to bind it for a block of work:

```python
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

## Instrumenting LLM and policy gateways

LLM gateway example:

```python
from electripy.observability.ai_telemetry import (
    InMemoryTelemetryAdapter,
    create_telemetry_context,
    record_llm_call,
)

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
from electripy.observability.ai_telemetry import (
    JsonlTelemetrySinkAdapter,
    create_telemetry_context,
    record_policy_decision,
)

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

## JSONL sink vs OpenTelemetry

- **JSONL sink** (`JsonlTelemetrySinkAdapter`) is ideal for local
  development and simple deployments. It writes one JSON object per
  line, ready to be ingested by log pipelines.
- **OpenTelemetry adapter** (`OpenTelemetryAdapter`) bridges to
  `opentelemetry` if installed via the `electripy[otel]` extra. It
  models events and metrics as OpenTelemetry spans/events.

Both adapters implement the same `TelemetryPort` Protocol, so you can
swap them without changing calling code.

## Usage examples

1. **Basic LLM call with correlation ID**

   ```python
   from pathlib import Path
   from datetime import datetime

   from electripy.observability.ai_telemetry import (
       JsonlTelemetrySinkAdapter,
       Severity,
       TelemetryEvent,
       create_telemetry_context,
   )

   ctx = create_telemetry_context(environment="dev")
   sink = JsonlTelemetrySinkAdapter(path=Path("telemetry.jsonl"))

   event = TelemetryEvent(
       name="llm.request",
       timestamp=datetime.utcnow(),
       context=ctx,
       attributes={"provider": "fake", "model": "demo"},
       severity=Severity.INFO,
   )
   sink.emit_event(event)
   ```

2. **End-to-end flow with spans**

   ```python
   from pathlib import Path

   from electripy.observability.ai_telemetry import (
       JsonlTelemetrySinkAdapter,
       create_telemetry_context,
       record_llm_call,
       record_policy_decision,
   )

   sink = JsonlTelemetrySinkAdapter(path=Path("telemetry.jsonl"))
   ctx = create_telemetry_context(environment="prod")

   with sink.span("policy_and_llm", ctx=ctx):
       record_policy_decision(
           sink,
           decision="allow",
           violation_codes=[],
           redactions_applied=False,
           ctx=ctx,
       )

       # Simulated LLM call
       record_llm_call(
           sink,
           provider="openai",
           model="gpt-4.1",
           latency_ms=150.0,
           input_tokens=1200,
           output_tokens=300,
           finish_reason="stop",
           structured_output_valid=True,
           ctx=ctx,
       )
   ```
