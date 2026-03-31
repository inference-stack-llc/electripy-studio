"""Tracer adapters for the observe package.

This module provides three concrete tracer implementations:

- :class:`NoOpTracer` — zero-cost stub used when observability is
  disabled.  All operations are silently ignored.
- :class:`InMemoryTracer` — collects finished spans in a list for
  test assertions.
- :class:`OpenTelemetryTracer` — bridges to OpenTelemetry when the
  ``opentelemetry`` package is installed.  Falls back to :class:`NoOpTracer`
  behaviour when the SDK is missing.

All adapters implement :class:`~electripy.observability.observe.ports.TracerPort`
and produce :class:`~electripy.observability.observe.ports.SpanPort` handles.
"""

from __future__ import annotations

import importlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from electripy.core.logging import get_logger

from .domain import (
    Attributes,
    AttributeValue,
    SpanKind,
    SpanStatus,
    SpanStatusCode,
    TraceContext,
)
from .errors import TracerConfigError
from .redaction import DefaultRedactor

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _generate_id(length: int = 32) -> str:
    """Generate a random hexadecimal identifier.

    Args:
        length: Number of hex characters; defaults to 32 (128-bit).

    Returns:
        Random hex string.
    """
    byte_count = (length + 1) // 2
    return os.urandom(byte_count).hex()[:length]


# ---------------------------------------------------------------------------
# NoOpSpan / NoOpTracer
# ---------------------------------------------------------------------------


class NoOpSpan:
    """Span that silently discards all operations.

    This is the span implementation returned by :class:`NoOpTracer`.
    It satisfies the :class:`SpanPort` protocol at near-zero cost.
    """

    __slots__ = ("_context", "_name")

    def __init__(self, name: str, context: TraceContext) -> None:
        self._name = name
        self._context = context

    @property
    def context(self) -> TraceContext:
        """Return the span's trace context."""
        return self._context

    @property
    def name(self) -> str:
        """Return the span name."""
        return self._name

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        """No-op."""

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        """No-op."""

    def add_event(self, name: str, *, attributes: Attributes | None = None) -> None:
        """No-op."""

    def record_exception(
        self,
        exception: BaseException,
        *,
        attributes: Attributes | None = None,
    ) -> None:
        """No-op."""

    def set_status(self, status: SpanStatus) -> None:
        """No-op."""

    def end(self) -> None:
        """No-op."""


@dataclass(slots=True)
class NoOpTracer:
    """Tracer that produces :class:`NoOpSpan` instances.

    Use this tracer when observability is disabled or a telemetry
    backend is not configured.  All span operations are silently
    discarded with negligible overhead.
    """

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> NoOpSpan:
        """Start a no-op span.

        Args:
            name: Span name (ignored).
            kind: Span kind (ignored).
            parent: Parent context (ignored).
            attributes: Initial attributes (ignored).

        Returns:
            A :class:`NoOpSpan` bound to a minimal context.
        """
        ctx = parent or TraceContext(trace_id=_generate_id(), span_id=_generate_id(16))
        child = ctx.child(span_id=_generate_id(16))
        return NoOpSpan(name=name, context=child)


# ---------------------------------------------------------------------------
# InMemorySpan / InMemoryTracer
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class SpanRecord:
    """A completed span record stored by :class:`InMemoryTracer`.

    Attributes:
        name: Span name.
        kind: Span kind.
        context: The span's trace context.
        parent_context: Parent trace context, if any.
        attributes: Final redacted attribute mapping.
        events: List of ``(name, attributes)`` tuples.
        exceptions: List of recorded exceptions.
        status: Terminal span status.
        start_time: Span start timestamp.
        end_time: Span end timestamp.
    """

    name: str
    kind: SpanKind
    context: TraceContext
    parent_context: TraceContext | None
    attributes: Attributes
    events: list[tuple[str, Attributes | None]]
    exceptions: list[BaseException]
    status: SpanStatus
    start_time: datetime
    end_time: datetime | None


class InMemorySpan:
    """Span that collects data in memory for test assertions.

    Satisfies :class:`SpanPort`.  On :meth:`end` the span record is
    appended to the owning tracer's :attr:`InMemoryTracer.finished_spans`.
    """

    __slots__ = (
        "_context",
        "_name",
        "_kind",
        "_parent_context",
        "_attributes",
        "_events",
        "_exceptions",
        "_status",
        "_start_time",
        "_ended",
        "_tracer",
        "_redactor",
    )

    def __init__(
        self,
        *,
        name: str,
        kind: SpanKind,
        context: TraceContext,
        parent_context: TraceContext | None,
        tracer: InMemoryTracer,
        redactor: DefaultRedactor | None,
    ) -> None:
        self._name = name
        self._kind = kind
        self._context = context
        self._parent_context = parent_context
        self._attributes: Attributes = {}
        self._events: list[tuple[str, Attributes | None]] = []
        self._exceptions: list[BaseException] = []
        self._status = SpanStatus()
        self._start_time = datetime.now(tz=UTC)
        self._ended = False
        self._tracer = tracer
        self._redactor = redactor

    @property
    def context(self) -> TraceContext:
        """Return the span's trace context."""
        return self._context

    @property
    def name(self) -> str:
        """Return the span name."""
        return self._name

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        """Set a single span attribute."""
        self._attributes[key] = value

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        """Set multiple span attributes at once."""
        self._attributes.update(attributes)

    def add_event(self, name: str, *, attributes: Attributes | None = None) -> None:
        """Record a named event on the span."""
        self._events.append((name, dict(attributes) if attributes else None))

    def record_exception(
        self,
        exception: BaseException,
        *,
        attributes: Attributes | None = None,
    ) -> None:
        """Record an exception and set the span status to ERROR."""
        self._exceptions.append(exception)
        attrs: Attributes = {
            "exception.type": type(exception).__name__,
            "exception.message": str(exception),
        }
        if attributes:
            attrs.update(attributes)
        self._events.append(("exception", attrs))
        self._status = SpanStatus(
            code=SpanStatusCode.ERROR,
            description=str(exception),
        )

    def set_status(self, status: SpanStatus) -> None:
        """Set the terminal status of the span."""
        self._status = status

    def end(self) -> None:
        """Finish the span and append the record to the tracer."""
        if self._ended:
            return
        self._ended = True

        final_attrs = self._attributes
        if self._redactor is not None:
            final_attrs = self._redactor.redact(final_attrs)

        record = SpanRecord(
            name=self._name,
            kind=self._kind,
            context=self._context,
            parent_context=self._parent_context,
            attributes=final_attrs,
            events=list(self._events),
            exceptions=list(self._exceptions),
            status=self._status,
            start_time=self._start_time,
            end_time=datetime.now(tz=UTC),
        )
        self._tracer.finished_spans.append(record)


@dataclass(slots=True)
class InMemoryTracer:
    """Tracer that stores finished spans in memory for testing.

    Attributes:
        finished_spans: List of completed :class:`SpanRecord` objects.
        redactor: Optional redactor applied to attributes on span end.
    """

    finished_spans: list[SpanRecord] = field(default_factory=list)
    redactor: DefaultRedactor | None = None

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> InMemorySpan:
        """Start a new in-memory span.

        Args:
            name: Span name.
            kind: Semantic span kind.
            parent: Optional parent context for nesting.
            attributes: Initial attributes.

        Returns:
            An :class:`InMemorySpan` instance.
        """
        parent_ctx = parent or TraceContext(
            trace_id=_generate_id(),
            span_id=_generate_id(16),
        )
        child_ctx = parent_ctx.child(span_id=_generate_id(16))

        span = InMemorySpan(
            name=name,
            kind=kind,
            context=child_ctx,
            parent_context=parent_ctx,
            tracer=self,
            redactor=self.redactor,
        )
        if attributes:
            span.set_attributes(attributes)
        return span


# ---------------------------------------------------------------------------
# OpenTelemetry adapter
# ---------------------------------------------------------------------------

# Span-kind mapping from our domain to OTel constants.
_OTEL_SPAN_KIND_MAP: dict[SpanKind, int] = {
    SpanKind.INTERNAL: 0,
    SpanKind.SERVER: 1,
    SpanKind.CLIENT: 2,
    SpanKind.PRODUCER: 3,
    SpanKind.CONSUMER: 4,
    # AI-specific kinds map to INTERNAL in OTel.
    SpanKind.LLM: 0,
    SpanKind.AGENT: 0,
    SpanKind.TOOL: 0,
    SpanKind.RETRIEVAL: 0,
    SpanKind.WORKFLOW: 0,
    SpanKind.POLICY: 0,
    SpanKind.MCP: 0,
}


class OpenTelemetrySpan:
    """Span implementation backed by an OpenTelemetry span.

    Satisfies :class:`SpanPort`.  All attribute values are sanitised
    by the configured redactor before being set on the underlying
    OTel span.
    """

    __slots__ = ("_otel_span", "_context", "_name", "_redactor", "_ended")

    def __init__(
        self,
        *,
        otel_span: Any,
        context: TraceContext,
        name: str,
        redactor: DefaultRedactor | None,
    ) -> None:
        self._otel_span = otel_span
        self._context = context
        self._name = name
        self._redactor = redactor
        self._ended = False

    @property
    def context(self) -> TraceContext:
        """Return the trace context."""
        return self._context

    @property
    def name(self) -> str:
        """Return the span name."""
        return self._name

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        """Set a single attribute, redacting if necessary."""
        safe = self._safe_attrs({key: value})
        for k, v in safe.items():
            self._otel_span.set_attribute(k, v)

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        """Set multiple attributes, redacting if necessary."""
        safe = self._safe_attrs(dict(attributes))
        for k, v in safe.items():
            self._otel_span.set_attribute(k, v)

    def add_event(self, name: str, *, attributes: Attributes | None = None) -> None:
        """Record an event on the OTel span."""
        safe = self._safe_attrs(attributes or {})
        self._otel_span.add_event(name, attributes=safe)

    def record_exception(
        self,
        exception: BaseException,
        *,
        attributes: Attributes | None = None,
    ) -> None:
        """Record an exception on the OTel span."""
        self._otel_span.record_exception(exception, attributes=attributes)
        self._otel_span.set_status(
            _import_otel_status_code("ERROR"),
            description=str(exception),
        )

    def set_status(self, status: SpanStatus) -> None:
        """Set the span status on the OTel span."""
        otel_code = _import_otel_status_code(status.code.value)
        self._otel_span.set_status(otel_code, description=status.description)

    def end(self) -> None:
        """End the OTel span."""
        if self._ended:
            return
        self._ended = True
        self._otel_span.end()

    def _safe_attrs(self, attrs: Attributes) -> Attributes:
        """Apply redaction if a redactor is configured."""
        if self._redactor is not None:
            return self._redactor.redact(attrs)
        return dict(attrs)


def _import_otel_status_code(code: str) -> Any:
    """Dynamically import an OTel StatusCode value.

    Args:
        code: Status code name (``"OK"``, ``"ERROR"``, ``"UNSET"``).

    Returns:
        The corresponding ``opentelemetry.trace.StatusCode`` value.
    """
    mod = importlib.import_module("opentelemetry.trace")
    status_enum = mod.StatusCode
    return getattr(status_enum, code, status_enum.UNSET)


@dataclass(slots=True)
class OpenTelemetryTracer:
    """Tracer adapter that bridges to OpenTelemetry.

    This adapter encapsulates all ``opentelemetry`` imports and API
    calls.  If the ``opentelemetry`` package is not installed, the
    constructor raises :class:`TracerConfigError` with a clear message.

    Attributes:
        service_name: Logical service name used to acquire an OTel
            tracer.
        redactor: Optional redactor applied to span attributes.
    """

    service_name: str = "electripy-observe"
    redactor: DefaultRedactor | None = None
    _otel_trace: Any = field(init=False, repr=False)
    _tracer: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            otel_trace = importlib.import_module("opentelemetry.trace")
        except ImportError as exc:
            raise TracerConfigError(
                "OpenTelemetryTracer requires the `opentelemetry-api` package. "
                "Install with: pip install opentelemetry-api opentelemetry-sdk"
            ) from exc
        self._otel_trace = otel_trace
        self._tracer = otel_trace.get_tracer(self.service_name)

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> OpenTelemetrySpan:
        """Start a new OTel-backed span.

        AI-specific span kinds are mapped to ``INTERNAL`` in
        OpenTelemetry; the semantic kind is recorded as the
        ``observe.span.kind`` attribute for filtering.

        Args:
            name: Span name.
            kind: Semantic span kind.
            parent: Optional parent context.
            attributes: Initial attributes.

        Returns:
            An :class:`OpenTelemetrySpan` wrapping the OTel handle.
        """
        otel_kind_int = _OTEL_SPAN_KIND_MAP.get(kind, 0)
        SpanKindEnum = self._otel_trace.SpanKind
        otel_kind = list(SpanKindEnum)[otel_kind_int]

        otel_span = self._tracer.start_span(name=name, kind=otel_kind)

        # Record the semantic kind as an attribute.
        otel_span.set_attribute("observe.span.kind", kind.value)

        # Build a TraceContext for the caller.
        otel_ctx = otel_span.get_span_context()
        trace_id = format(otel_ctx.trace_id, "032x")
        span_id = format(otel_ctx.span_id, "016x")

        parent_span_id: str | None = None
        if parent is not None:
            parent_span_id = parent.span_id

        ctx = TraceContext(
            trace_id=trace_id,
            span_id=span_id,
            parent_span_id=parent_span_id,
            request_id=parent.request_id if parent else None,
            actor_id=parent.actor_id if parent else None,
            tenant_id=parent.tenant_id if parent else None,
            environment=parent.environment if parent else None,
            baggage=dict(parent.baggage) if parent else {},
        )

        otel_wrapped = OpenTelemetrySpan(
            otel_span=otel_span,
            context=ctx,
            name=name,
            redactor=self.redactor,
        )

        if attributes:
            otel_wrapped.set_attributes(attributes)

        return otel_wrapped


__all__ = [
    "NoOpSpan",
    "NoOpTracer",
    "SpanRecord",
    "InMemorySpan",
    "InMemoryTracer",
    "OpenTelemetrySpan",
    "OpenTelemetryTracer",
]
