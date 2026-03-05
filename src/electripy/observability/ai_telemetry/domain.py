"""Domain models for AI-focused telemetry.

This module defines the core data structures used for telemetry events,
metrics, and cost estimation. It is intentionally free of any concrete
telemetry backends and can be shared across adapters.

The main guarantees are:
- TelemetryContext models correlation identifiers in a backend-agnostic way.
- TelemetryEvent and metric types are small, typed data classes.
- No sensitive prompt/response content is required; callers should only
  provide hashed or redacted data where appropriate.

Example:
    from datetime import datetime

    from electripy.observability.ai_telemetry.domain import (
        Severity,
        TelemetryContext,
        TelemetryEvent,
    )

    ctx = TelemetryContext(trace_id="t1", span_id=None, parent_span_id=None, request_id="req-1", actor_id=None, tenant_id=None, environment="dev", tags={})
    event = TelemetryEvent(
        name="llm_request",
        timestamp=datetime.utcnow(),
        context=ctx,
        attributes={"provider": "openai"},
        severity=Severity.INFO,
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Mapping, MutableMapping, TypeAlias

AttributeValue: TypeAlias = str | int | float | bool | None
Attributes: TypeAlias = MutableMapping[str, AttributeValue]


class Severity(StrEnum):
    """Severity levels for telemetry events.

    These values intentionally mirror common logging and telemetry
    conventions while remaining backend-agnostic.

    Example:
        level = Severity.INFO
    """

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


@dataclass(slots=True)
class TelemetryContext:
    """Correlation context for telemetry operations.

    The context models identifiers that allow events, metrics, and logs
    to be correlated across process boundaries.

    Attributes:
        trace_id: Stable identifier for a logical trace.
        span_id: Identifier for the current span within the trace.
        parent_span_id: Optional parent span identifier.
        request_id: Optional per-request identifier (e.g. HTTP request).
        actor_id: Optional identifier for the end-user or service actor.
        tenant_id: Optional multi-tenant identifier.
        environment: Optional environment label (e.g. "dev", "prod").
        tags: Arbitrary string tags for further classification.

    Example:
        ctx = TelemetryContext(
            trace_id="trace-1",
            span_id=None,
            parent_span_id=None,
            request_id="req-123",
            actor_id=None,
            tenant_id=None,
            environment="dev",
            tags={"service": "api"},
        )
    """

    trace_id: str
    span_id: str | None
    parent_span_id: str | None
    request_id: str | None
    actor_id: str | None
    tenant_id: str | None
    environment: str | None
    tags: dict[str, str] = field(default_factory=dict)

    def child(self, *, span_id: str) -> "TelemetryContext":
        """Return a child context for a nested span.

        Args:
            span_id: Identifier for the child span.

        Returns:
            TelemetryContext: New context with updated span information.
        """

        return TelemetryContext(
            trace_id=self.trace_id,
            span_id=span_id,
            parent_span_id=self.span_id,
            request_id=self.request_id,
            actor_id=self.actor_id,
            tenant_id=self.tenant_id,
            environment=self.environment,
            tags=dict(self.tags),
        )

    def to_headers(self) -> dict[str, str]:
        """Render a minimal set of correlation headers.

        Returns:
            dict[str, str]: Headers such as ``X-Request-Id`` and
            ``Traceparent`` suitable for outbound HTTP calls.
        """

        headers: dict[str, str] = {}
        if self.request_id is not None:
            headers["X-Request-Id"] = self.request_id
        if self.trace_id is not None and self.span_id is not None:
            traceparent = f"00-{self.trace_id}-{self.span_id}-01"
            headers["Traceparent"] = traceparent
        return headers


@dataclass(slots=True)
class TelemetryEvent:
    """A structured telemetry event.

    Attributes:
        name: Event name (for example ``"llm_request"``).
        timestamp: Event timestamp in UTC.
        context: Correlation context.
        attributes: Additional typed attributes.
        severity: Severity level.
    """

    name: str
    timestamp: datetime
    context: TelemetryContext
    attributes: Attributes
    severity: Severity


@dataclass(slots=True)
class CounterIncrement:
    """Represents an increment to a named counter metric.

    Attributes:
        name: Metric name.
        value: Amount to increment by (default 1).
        attributes: Optional metric attributes.
        context: Optional correlation context.
        timestamp: Time of the observation.
    """

    name: str
    value: int = 1
    attributes: Attributes | None = None
    context: TelemetryContext | None = None
    timestamp: datetime | None = None


@dataclass(slots=True)
class HistogramObservation:
    """Represents a single observation for a histogram metric.

    Attributes:
        name: Metric name.
        value: Observed value.
        attributes: Optional metric attributes.
        context: Optional correlation context.
        timestamp: Time of the observation.
    """

    name: str
    value: float
    attributes: Attributes | None = None
    context: TelemetryContext | None = None
    timestamp: datetime | None = None


@dataclass(slots=True)
class CostRecord:
    """Estimated cost information for an AI call.

    Attributes:
        provider: Logical provider name (for example ``"openai"``).
        model: Model identifier.
        input_tokens: Number of input tokens.
        output_tokens: Number of output tokens.
        estimated_cost_usd: Optional estimated cost in USD.
    """

    provider: str
    model: str
    input_tokens: int
    output_tokens: int
    estimated_cost_usd: float | None


__all__ = [
    "AttributeValue",
    "Attributes",
    "Severity",
    "TelemetryContext",
    "TelemetryEvent",
    "CounterIncrement",
    "HistogramObservation",
    "CostRecord",
]
