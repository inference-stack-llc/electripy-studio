"""Structured observability for AI, agent, and tool workloads.

The ``observe`` package provides a production-ready tracing and
instrumentation layer for LLM calls, retrieval operations, tool
invocations, workflow steps, policy checks, agent spans, and MCP
interactions.  It is built around OpenTelemetry-aligned concepts while
remaining provider-neutral and consistent with ElectriPy Studio's
Ports & Adapters architecture.

Key design goals:

- **Provider-neutral**: core abstractions never import a specific
  telemetry backend.
- **Safe by default**: a first-class redaction subsystem ensures
  sensitive fields (prompts, completions, secrets, PII) are sanitised
  before leaving the application boundary.
- **No-op when disabled**: the ``NoOpTracer`` adapter is zero-cost and
  is the default when no backend is configured.
- **Composable**: decorators, context managers, and the
  ``ObservabilityService`` can be mixed freely.

Example::

    from electripy.observability.observe import (
        ObservabilityService,
        InMemoryTracer,
    )

    tracer = InMemoryTracer()
    svc = ObservabilityService(tracer=tracer)

    with svc.start_llm_span(provider="openai", model="gpt-4o") as span:
        span.set_attribute("gen_ai.input_tokens", 120)
        span.set_attribute("gen_ai.output_tokens", 45)

    assert tracer.finished_spans
"""

from __future__ import annotations

from .adapters import InMemoryTracer, NoOpTracer, OpenTelemetryTracer
from .decorators import observe_function, observe_tool
from .domain import (
    GenAIRequestMetadata,
    GenAIResponseMetadata,
    MCPMetadata,
    PolicyDecisionMetadata,
    RedactionPolicy,
    RedactionRule,
    RetrievalMetadata,
    SpanAttributes,
    SpanKind,
    SpanStatus,
    SpanStatusCode,
    ToolInvocationMetadata,
    TraceContext,
)
from .errors import ObserveError, RedactionError, SpanError, TracerConfigError
from .ports import (
    ContextCarrierPort,
    EventEmitterPort,
    RedactorPort,
    SpanPort,
    TracerPort,
)
from .redaction import DefaultRedactor
from .services import ObservabilityService, aobserve_span, observe_span

__all__ = [
    # Domain models
    "TraceContext",
    "SpanAttributes",
    "SpanKind",
    "SpanStatus",
    "SpanStatusCode",
    "GenAIRequestMetadata",
    "GenAIResponseMetadata",
    "ToolInvocationMetadata",
    "RetrievalMetadata",
    "PolicyDecisionMetadata",
    "MCPMetadata",
    "RedactionPolicy",
    "RedactionRule",
    # Ports
    "TracerPort",
    "SpanPort",
    "EventEmitterPort",
    "RedactorPort",
    "ContextCarrierPort",
    # Errors
    "ObserveError",
    "SpanError",
    "TracerConfigError",
    "RedactionError",
    # Adapters
    "NoOpTracer",
    "InMemoryTracer",
    "OpenTelemetryTracer",
    # Redaction
    "DefaultRedactor",
    # Services
    "ObservabilityService",
    "observe_span",
    "aobserve_span",
    # Decorators
    "observe_function",
    "observe_tool",
]
