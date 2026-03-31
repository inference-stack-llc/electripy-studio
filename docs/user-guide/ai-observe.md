# Observe — Structured Observability for AI Workloads

The **Observe** component provides a production-ready tracing and
instrumentation layer for LLM calls, retrieval operations, tool
invocations, workflow steps, policy checks, agent spans, and MCP
interactions.

It is built around OpenTelemetry-aligned concepts while remaining
**provider-neutral** and consistent with ElectriPy Studio's Ports &
Adapters architecture.

## When to use it

Use the Observe package when you want:

- A clean, typed API for instrumenting AI / LLM workloads.
- Automatic span nesting with correct parent-child relationships.
- First-class redaction of sensitive attributes (prompts, secrets, PII).
- The ability to swap tracing backends (no-op, in-memory, OpenTelemetry)
  without changing business logic.
- Safe-by-default behavior in enterprise environments.

## Key concepts

- **`TraceContext`** — correlation identifiers (trace ID, span ID,
  request ID, tenant, environment, baggage) that propagate through
  nested spans.
- **`SpanPort`** — the interface for interacting with an active span
  (set attributes, record events/exceptions, end).
- **`TracerPort`** — the factory interface for creating spans.
- **`ObservabilityService`** — high-level service providing semantic
  span helpers for LLM, tool, retrieval, agent, workflow, policy, and
  MCP operations.
- **`RedactorPort` / `DefaultRedactor`** — attribute redaction before
  export.
- **Adapters** — `NoOpTracer` (zero-cost default), `InMemoryTracer`
  (tests), `OpenTelemetryTracer` (production).

## Architecture

```
┌─────────────────────────────────────────────────────┐
│               ObservabilityService                  │
│  start_llm_span / start_tool_span / ...             │
│  record_policy_decision / annotate_span / ...       │
├─────────────────────────────────────────────────────┤
│                     Ports                           │
│  TracerPort  │  SpanPort  │  RedactorPort           │
├──────────────┼────────────┼─────────────────────────┤
│              Adapters                               │
│  NoOpTracer  │ InMemoryTracer │ OpenTelemetryTracer │
│              │                │                     │
│              │  DefaultRedactor                     │
└──────────────┴────────────────┴─────────────────────┘
```

## Installation

The Observe package is included with ElectriPy Studio:

```bash
pip install electripy-studio
```

For OpenTelemetry integration, install the optional extra:

```bash
pip install opentelemetry-api opentelemetry-sdk
```

## Basic example

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

# Inspect recorded spans
for record in tracer.finished_spans:
    print(record.name, record.attributes)
```

## Advanced example: full AI pipeline

The following example demonstrates instrumenting a realistic LLM +
retrieval + policy pipeline:

```python
from electripy.observability.observe import (
    DefaultRedactor,
    GenAIRequestMetadata,
    GenAIResponseMetadata,
    InMemoryTracer,
    ObservabilityService,
    PolicyDecisionMetadata,
    RedactionPolicy,
    RetrievalMetadata,
)

# Configure tracer with enterprise-safe redaction
redactor = DefaultRedactor(policy=RedactionPolicy.enterprise_default())
tracer = InMemoryTracer(redactor=redactor)
svc = ObservabilityService(tracer=tracer)

# Workflow span wrapping all steps
with svc.start_workflow_span("rag_pipeline"):

    # Step 1: Retrieval
    retrieval_meta = RetrievalMetadata(
        source="pinecone",
        top_k=5,
        results_returned=3,
        latency_ms=45.0,
        score_max=0.95,
    )
    with svc.start_retrieval_span("pinecone", meta=retrieval_meta) as ret_span:
        # ... perform retrieval ...
        pass

    # Step 2: Policy check
    policy_meta = PolicyDecisionMetadata(
        action="allow",
        policy_version="2.1",
        violation_codes=[],
        redactions_applied=False,
    )
    svc.record_policy_decision(policy_meta)

    # Step 3: LLM call
    request_meta = GenAIRequestMetadata(
        provider="openai",
        model="gpt-4o",
        temperature=0.3,
        max_tokens=512,
    )
    with svc.start_llm_span(
        provider="openai",
        model="gpt-4o",
        request_meta=request_meta,
    ) as llm_span:
        # ... invoke LLM ...
        response_meta = GenAIResponseMetadata(
            output_tokens=128,
            input_tokens=400,
            finish_reason="stop",
            latency_ms=350.0,
            cache_hit=False,
        )
        svc.record_llm_response(response_meta)

# All spans are collected with correct nesting
for record in tracer.finished_spans:
    print(f"{record.name}  parent={record.context.parent_span_id}")
```

## Tool instrumentation with decorators

```python
from electripy.observability.observe import (
    InMemoryTracer,
    ObservabilityService,
    observe_function,
    observe_tool,
)

tracer = InMemoryTracer()
svc = ObservabilityService(tracer=tracer)


@observe_tool(svc, tool_name="calculator", tool_version="1.0")
def add(a: int, b: int) -> int:
    return a + b


@observe_function(svc, name="my_pipeline")
def run_pipeline() -> int:
    return add(21, 21)


result = run_pipeline()
# Two spans: "my_pipeline" (parent) → "tool.calculator" (child)
```

## MCP instrumentation

```python
from electripy.observability.observe import (
    InMemoryTracer,
    MCPMetadata,
    ObservabilityService,
)

tracer = InMemoryTracer()
svc = ObservabilityService(tracer=tracer)

meta = MCPMetadata(
    server_name="code-server",
    tool_name="run_code",
    protocol_version="1.0",
)

with svc.start_mcp_span("code-server", meta=meta) as span:
    # ... MCP tool call ...
    span.set_attribute("mcp.status", "success")
```

## Redaction strategy

The redaction subsystem ensures sensitive data never leaves the
application boundary.  Three rule kinds are supported:

| Kind       | How it works                                        |
|------------|-----------------------------------------------------|
| `EXACT`    | Attribute key matches a known-sensitive name.       |
| `PATTERN`  | Attribute key matches a regular expression.         |
| `CALLABLE` | A user-supplied predicate receives `(key, value)`.  |

### Enterprise default

`RedactionPolicy.enterprise_default()` redacts attributes whose keys
match commonly sensitive names:

- `prompt`, `completion`, `response`, `content`
- `api_key`, `secret`, `password`, `token`, `authorization`, `auth_header`
- `ssn`, `credit_card`

### Custom rules

```python
from electripy.observability.observe import (
    DefaultRedactor,
    RedactionPolicy,
    RedactionRule,
)
from electripy.observability.observe.domain import RedactionRuleKind

policy = RedactionPolicy(
    rules=(
        # Exact match
        RedactionRule(kind=RedactionRuleKind.EXACT, match="ssn"),
        # Regex pattern
        RedactionRule(
            kind=RedactionRuleKind.PATTERN,
            match=r"^auth",
            replacement="***",
        ),
        # Custom predicate
        RedactionRule(
            kind=RedactionRuleKind.CALLABLE,
            predicate=lambda k, v: isinstance(v, str) and len(str(v)) > 1000,
            replacement="[LARGE_VALUE_REDACTED]",
        ),
    ),
)

redactor = DefaultRedactor(policy=policy)
clean = redactor.redact({"ssn": "123-45-6789", "model": "gpt-4o"})
assert clean["ssn"] == "[REDACTED]"
assert clean["model"] == "gpt-4o"
```

## Extension points

### Custom tracer adapter

To integrate with a different telemetry backend, implement
`TracerPort`:

```python
from electripy.observability.observe.ports import TracerPort, SpanPort
from electripy.observability.observe.domain import SpanKind, TraceContext

class MyCustomTracer:
    """Custom tracer satisfying TracerPort."""

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        attributes=None,
    ) -> SpanPort:
        # Create and return a span backed by your telemetry system
        ...
```

### Custom redactor

Implement `RedactorPort` for alternative redaction strategies:

```python
from electripy.observability.observe.ports import RedactorPort
from electripy.observability.observe.domain import Attributes

class HashingRedactor:
    """Replace sensitive values with SHA-256 hashes."""

    def redact(self, attributes: Attributes) -> Attributes:
        import hashlib
        result = {}
        for k, v in attributes.items():
            if k in {"prompt", "completion"}:
                result[k] = hashlib.sha256(str(v).encode()).hexdigest()
            else:
                result[k] = v
        return result
```

## No-op behavior

When observability is not needed, `NoOpTracer` (the default) ensures
all span operations are silently discarded with negligible overhead.
No telemetry data is collected, no context variables are modified, and
no exceptions are raised.

```python
from electripy.observability.observe import ObservabilityService

# Default: zero-cost no-op
svc = ObservabilityService()

with svc.start_llm_span(provider="openai", model="gpt-4o") as span:
    span.set_attribute("gen_ai.usage.input_tokens", 120)
    # All operations are silently ignored
```

## Related components

- [AI Telemetry](ai-telemetry.md) — event and metrics-focused
  telemetry primitives for AI workloads.
- [AI Policy Gateway](ai-policy-gateway.md) — policy enforcement
  for LLM requests and responses.
- [AI LLM Gateway](ai-llm-gateway.md) — provider-agnostic LLM
  client with retries and structured output.
