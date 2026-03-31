"""Ports (Protocols) for the observe package.

These Protocols define the minimal surface area required by the
service and decorator layers.  Concrete adapters — no-op, in-memory,
OpenTelemetry — implement these interfaces so that the telemetry
backend can be swapped without changing business logic.

All ports are ``@runtime_checkable`` to support structural subtyping
checks at runtime.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
from typing import Protocol, runtime_checkable

from .domain import Attributes, AttributeValue, SpanKind, SpanStatus, TraceContext

# ---------------------------------------------------------------------------
# SpanPort — lifecycle of a single span
# ---------------------------------------------------------------------------


@runtime_checkable
class SpanPort(Protocol):
    """Port for interacting with an active span.

    Implementations wrap a backend-specific span handle and expose a
    uniform interface for setting attributes, recording events and
    exceptions, and finishing the span.
    """

    @property
    def context(self) -> TraceContext:
        """Return the :class:`TraceContext` associated with this span."""
        ...

    @property
    def name(self) -> str:
        """Return the span name."""
        ...

    def set_attribute(self, key: str, value: AttributeValue) -> None:
        """Set a single attribute on the span.

        Args:
            key: Attribute key following dotted-namespace conventions.
            value: Primitive attribute value.
        """
        ...

    def set_attributes(self, attributes: Mapping[str, AttributeValue]) -> None:
        """Set multiple attributes at once.

        Args:
            attributes: Mapping of attribute keys to values.
        """
        ...

    def add_event(
        self,
        name: str,
        *,
        attributes: Attributes | None = None,
    ) -> None:
        """Record a timestamped event on the span.

        Args:
            name: Event name.
            attributes: Optional event attributes.
        """
        ...

    def record_exception(
        self,
        exception: BaseException,
        *,
        attributes: Attributes | None = None,
    ) -> None:
        """Record an exception on the span.

        The span status is automatically set to ``ERROR``.

        Args:
            exception: The exception instance.
            attributes: Optional additional attributes.
        """
        ...

    def set_status(self, status: SpanStatus) -> None:
        """Set the terminal status of the span.

        Args:
            status: Span status to record.
        """
        ...

    def end(self) -> None:
        """Finish the span and record its duration.

        Calling ``end()`` more than once is a no-op.
        """
        ...


# ---------------------------------------------------------------------------
# TracerPort — span factory
# ---------------------------------------------------------------------------


@runtime_checkable
class TracerPort(Protocol):
    """Port for creating spans.

    A tracer is the entry-point for all instrumentation.  It creates
    child spans either as raw objects or as context-manager helpers.
    """

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        parent: TraceContext | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> SpanPort:
        """Start a new span and return a :class:`SpanPort` handle.

        Args:
            name: Human-readable span name.
            kind: Semantic span classification.
            parent: Optional parent context for nesting.
            attributes: Initial attributes.

        Returns:
            An active :class:`SpanPort`.
        """
        ...


# ---------------------------------------------------------------------------
# EventEmitterPort
# ---------------------------------------------------------------------------


@runtime_checkable
class EventEmitterPort(Protocol):
    """Port for emitting standalone events outside of a span.

    This is useful for recording events (e.g. policy decisions) that
    are not naturally associated with a span lifecycle.
    """

    def emit(
        self,
        name: str,
        *,
        attributes: Attributes | None = None,
    ) -> None:
        """Emit a named event with optional attributes.

        Args:
            name: Event name.
            attributes: Optional attributes.
        """
        ...


# ---------------------------------------------------------------------------
# RedactorPort
# ---------------------------------------------------------------------------


@runtime_checkable
class RedactorPort(Protocol):
    """Port for redacting span attributes before export.

    Implementations apply a redaction policy to an attribute mapping
    and return a sanitised copy.
    """

    def redact(self, attributes: Attributes) -> Attributes:
        """Return a redacted copy of *attributes*.

        Args:
            attributes: Raw attribute mapping.

        Returns:
            Sanitised copy; original is never mutated.
        """
        ...


# ---------------------------------------------------------------------------
# ContextCarrierPort — cross-process propagation
# ---------------------------------------------------------------------------


@runtime_checkable
class ContextCarrierPort(Protocol):
    """Port for injecting/extracting trace context into carriers.

    The carrier is typically an HTTP header mapping.  Implementations
    handle serialisation according to W3C Trace-Context or equivalent.
    """

    def inject(
        self,
        headers: MutableMapping[str, str],
        context: TraceContext,
    ) -> None:
        """Inject correlation identifiers into *headers*.

        Args:
            headers: Mutable header mapping to update.
            context: Trace context to serialise.
        """
        ...

    def extract(
        self,
        headers: Mapping[str, str],
    ) -> TraceContext | None:
        """Extract a trace context from *headers*.

        Args:
            headers: Incoming header mapping.

        Returns:
            Extracted :class:`TraceContext` or ``None`` if the headers
            do not carry valid trace information.
        """
        ...


__all__ = [
    "SpanPort",
    "TracerPort",
    "EventEmitterPort",
    "RedactorPort",
    "ContextCarrierPort",
]
