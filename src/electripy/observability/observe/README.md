# Observe — Structured Observability for AI Workloads

The Observe component provides a production-ready tracing and
instrumentation layer for LLM calls, retrieval operations, tool
invocations, workflow steps, policy checks, agent spans, and MCP
interactions.

It is built around OpenTelemetry-aligned concepts while remaining
provider-neutral and consistent with ElectriPy Studio's Ports &
Adapters architecture.

## Structure

- `domain.py` — TraceContext, SpanAttributes, metadata models,
  RedactionPolicy/Rule
- `ports.py` — TracerPort, SpanPort, EventEmitterPort, RedactorPort,
  ContextCarrierPort
- `errors.py` — ObserveError hierarchy
- `redaction.py` — DefaultRedactor implementation
- `adapters.py` — NoOpTracer, InMemoryTracer, OpenTelemetryTracer
- `services.py` — ObservabilityService, observe_span, aobserve_span
- `decorators.py` — @observe_function, @observe_tool

## Quick start

```python
from electripy.observability.observe import (
    InMemoryTracer,
    ObservabilityService,
)

tracer = InMemoryTracer()
svc = ObservabilityService(tracer=tracer)

with svc.start_llm_span(provider="openai", model="gpt-4o") as span:
    span.set_attribute("gen_ai.usage.input_tokens", 120)
    span.set_attribute("gen_ai.usage.output_tokens", 45)
```

See `docs/user-guide/ai-observe.md` for full documentation.
