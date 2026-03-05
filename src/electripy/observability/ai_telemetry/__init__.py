"""Public API for the AI telemetry component.

This package provides provider-agnostic, framework-agnostic telemetry
primitives focused on AI workloads such as HTTP resilience behavior,
LLM gateway calls, policy decisions, and RAG evaluation runs.

Example:
    from electripy.observability.ai_telemetry import (
        TelemetryContext,
        TelemetryEvent,
        JsonlTelemetrySinkAdapter,
        InMemoryTelemetryAdapter,
        TelemetryPort,
        create_telemetry_context,
    )

    ctx = create_telemetry_context(environment="dev")
    sink = JsonlTelemetrySinkAdapter(path="telemetry.jsonl")
    sink.emit_event(TelemetryEvent(name="example", timestamp=datetime.utcnow(), context=ctx, attributes={}, severity=Severity.INFO))
"""

from __future__ import annotations

from .adapters import InMemoryTelemetryAdapter, JsonlTelemetrySinkAdapter, OpenTelemetryAdapter
from .domain import (
    CostRecord,
    HistogramObservation,
    Severity,
    TelemetryContext,
    TelemetryEvent,
)
from .ports import CostEstimatorPort, TelemetryPort
from .services import (
    TableCostEstimator,
    create_telemetry_context,
    current_telemetry_context,
    inject_context_headers,
    record_llm_call,
    scoped_telemetry_context,
)

__all__ = [
    "TelemetryPort",
    "CostEstimatorPort",
    "TelemetryContext",
    "TelemetryEvent",
    "Severity",
    "CostRecord",
    "HistogramObservation",
    "JsonlTelemetrySinkAdapter",
    "InMemoryTelemetryAdapter",
    "OpenTelemetryAdapter",
    "TableCostEstimator",
    "create_telemetry_context",
    "current_telemetry_context",
    "scoped_telemetry_context",
    "inject_context_headers",
    "record_llm_call",
]
