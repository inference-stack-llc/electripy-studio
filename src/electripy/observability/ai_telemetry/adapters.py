"""Telemetry adapters for AI workloads.

This module provides concrete implementations of :class:`TelemetryPort`
for different backends:

- ``JsonlTelemetrySinkAdapter`` writes safe JSON lines to a file for
  local development and lightweight production use.
- ``InMemoryTelemetryAdapter`` keeps events and metrics in memory,
  primarily for tests.
- ``OpenTelemetryAdapter`` offers an optional integration with
  OpenTelemetry when installed.

All adapters are **safe by default**: they avoid emitting raw
prompts/responses and instead rely on hashing and redaction.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

from electripy.core.logging import get_logger

from .domain import (
    Attributes,
    CounterIncrement,
    HistogramObservation,
    Severity,
    TelemetryContext,
    TelemetryEvent,
)
from .ports import SpanContextManager, TelemetryPort

logger = get_logger(__name__)

_SENSITIVE_KEYS = {"prompt", "response", "request_body", "completion", "content"}


def _hash_text(value: str) -> str:
    """Return a stable SHA-256 hex digest for the given text.

    Args:
        value: Text to hash.

    Returns:
        str: Hexadecimal digest.
    """

    digest = sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


def _sanitize_attributes(attrs: dict[str, object] | None) -> Attributes:
    """Sanitise attributes to avoid leaking sensitive text.

    Any attribute whose key suggests prompt/response-like content is
    replaced with a hash. The original value is not emitted.

    Args:
        attrs: Raw attributes.

    Returns:
        Attributes: Sanitised attributes mapping.
    """

    safe: Attributes = {}
    if not attrs:
        return safe

    for key, value in attrs.items():
        lowered = key.lower()
        if lowered in _SENSITIVE_KEYS and isinstance(value, str):
            safe[f"{key}_hash"] = _hash_text(value)
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            safe[key] = value
        else:
            safe[key] = str(value)
    return safe


class _Span(SpanContextManager):
    """Span context manager used by telemetry adapters.

    The span emits a ``span_started`` event on enter and a
    ``span_finished`` event on exit, including a ``duration_ms``
    attribute. It also updates the current context via the provided
    callbacks.
    """

    def __init__(
        self,
        *,
        sink: TelemetryPort,
        name: str,
        ctx: TelemetryContext | None,
        attrs: dict[str, object] | None,
        set_current: Callable[[TelemetryContext | None], None],
        get_current: Callable[[], TelemetryContext | None],
    ) -> None:
        self._sink = sink
        self._name = name
        self._ctx = ctx
        self._attrs = attrs or {}
        self._set_current = set_current
        self._get_current = get_current
        self._start_time: datetime | None = None
        self._previous_ctx: TelemetryContext | None = None

    def __enter__(self) -> TelemetryContext:
        now = datetime.now(tz=UTC)
        base_ctx = self._ctx or self._get_current()
        if base_ctx is None:
            # Minimal context; caller can create richer contexts via
            # services helpers.
            trace_id = _hash_text(f"trace-{now.isoformat()}")
            base_ctx = TelemetryContext(
                trace_id=trace_id,
                span_id=None,
                parent_span_id=None,
                request_id=None,
                actor_id=None,
                tenant_id=None,
                environment=None,
                tags={},
            )
        span_id = _hash_text(f"span-{now.isoformat()}")[:16]
        ctx = base_ctx.child(span_id=span_id)

        self._previous_ctx = self._get_current()
        self._set_current(ctx)
        self._start_time = now

        event = TelemetryEvent(
            name="span_started",
            timestamp=now,
            context=ctx,
            attributes=_sanitize_attributes({"span_name": self._name, **self._attrs}),
            severity=Severity.DEBUG,
        )
        self._sink.emit_event(event)
        return ctx

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        end = datetime.now(tz=UTC)
        ctx = self._get_current()
        if self._start_time is not None and ctx is not None:
            duration_ms = (end - self._start_time).total_seconds() * 1000.0
            attrs: dict[str, object] = {"span_name": self._name, "duration_ms": duration_ms}
            event = TelemetryEvent(
                name="span_finished",
                timestamp=end,
                context=ctx,
                attributes=_sanitize_attributes({**attrs, **self._attrs}),
                severity=Severity.DEBUG,
            )
            self._sink.emit_event(event)

        # Restore previous context.
        self._set_current(self._previous_ctx)
        return None

    async def __aenter__(self) -> TelemetryContext:  # type: ignore[override]
        return self.__enter__()

    async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.__exit__(exc_type, exc, tb)


@dataclass(slots=True)
class JsonlTelemetrySinkAdapter(TelemetryPort):
    """Telemetry adapter that writes JSON lines to a file.

    Each event or metric is serialised as a single JSON object per line
    with a small, stable schema. This is suitable for local development
    and can be shipped into enterprise log pipelines.

    Args:
        path: Path to the JSONL file.
    """

    path: Path

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def _write_record(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")

    def emit_event(self, event: TelemetryEvent) -> None:
        record = {
            "type": "event",
            "name": event.name,
            "timestamp": event.timestamp.astimezone(UTC).isoformat(),
            "severity": event.severity.value,
            "context": asdict(event.context),
            "attributes": _sanitize_attributes(dict(event.attributes)),
        }
        self._write_record(record)

    def increment(
        self,
        name: str,
        value: int = 1,
        *,
        attrs: dict[str, object] | None = None,
        ctx: TelemetryContext | None = None,
    ) -> None:
        now = datetime.now(tz=UTC)
        record = CounterIncrement(
            name=name,
            value=value,
            attributes=_sanitize_attributes(attrs or {}),
            context=ctx,
            timestamp=now,
        )
        serialised = asdict(record)
        serialised["type"] = "counter"
        serialised["timestamp"] = now.isoformat()
        self._write_record(serialised)

    def observe(
        self,
        name: str,
        value: float,
        *,
        attrs: dict[str, object] | None = None,
        ctx: TelemetryContext | None = None,
    ) -> None:
        now = datetime.now(tz=UTC)
        record = HistogramObservation(
            name=name,
            value=value,
            attributes=_sanitize_attributes(attrs or {}),
            context=ctx,
            timestamp=now,
        )
        serialised = asdict(record)
        serialised["type"] = "histogram"
        serialised["timestamp"] = now.isoformat()
        self._write_record(serialised)

    def span(
        self,
        name: str,
        *,
        ctx: TelemetryContext | None = None,
        attrs: dict[str, object] | None = None,
    ) -> SpanContextManager:
        from .services import current_telemetry_context, set_current_telemetry_context

        return _Span(
            sink=self,
            name=name,
            ctx=ctx,
            attrs=attrs,
            set_current=set_current_telemetry_context,
            get_current=current_telemetry_context,
        )


@dataclass(slots=True)
class InMemoryTelemetryAdapter(TelemetryPort):
    """In-memory telemetry adapter for tests.

    Collected events and metrics are stored on the instance for
    inspection within tests. Sensitive attributes are still sanitised by
    default.
    """

    events: list[TelemetryEvent]
    counters: list[CounterIncrement]
    histograms: list[HistogramObservation]

    def __init__(self) -> None:
        self.events = []
        self.counters = []
        self.histograms = []

    def emit_event(self, event: TelemetryEvent) -> None:
        safe_event = TelemetryEvent(
            name=event.name,
            timestamp=event.timestamp,
            context=event.context,
            attributes=_sanitize_attributes(dict(event.attributes)),
            severity=event.severity,
        )
        self.events.append(safe_event)

    def increment(
        self,
        name: str,
        value: int = 1,
        *,
        attrs: dict[str, object] | None = None,
        ctx: TelemetryContext | None = None,
    ) -> None:
        record = CounterIncrement(
            name=name,
            value=value,
            attributes=_sanitize_attributes(attrs or {}),
            context=ctx,
            timestamp=datetime.now(tz=UTC),
        )
        self.counters.append(record)

    def observe(
        self,
        name: str,
        value: float,
        *,
        attrs: dict[str, object] | None = None,
        ctx: TelemetryContext | None = None,
    ) -> None:
        record = HistogramObservation(
            name=name,
            value=value,
            attributes=_sanitize_attributes(attrs or {}),
            context=ctx,
            timestamp=datetime.now(tz=UTC),
        )
        self.histograms.append(record)

    def span(
        self,
        name: str,
        *,
        ctx: TelemetryContext | None = None,
        attrs: dict[str, object] | None = None,
    ) -> SpanContextManager:
        from .services import current_telemetry_context, set_current_telemetry_context

        return _Span(
            sink=self,
            name=name,
            ctx=ctx,
            attrs=attrs,
            set_current=set_current_telemetry_context,
            get_current=current_telemetry_context,
        )


try:  # Optional OpenTelemetry integration
    from opentelemetry import trace as _otel_trace  # type: ignore[import]

    _HAS_OTEL = True
except Exception:  # pragma: no cover - depends on optional extra
    _otel_trace: Any
    _HAS_OTEL = False


@dataclass(slots=True)
class OpenTelemetryAdapter(TelemetryPort):
    """Telemetry adapter that bridges to OpenTelemetry.

    This adapter is only functional when the ``opentelemetry`` package
    is installed (typically via an ``.[otel]`` extra). Importing this
    class never raises; attempting to use it without OpenTelemetry
    installed will result in a runtime error with a clear message.
    """

    service_name: str = "electripy-ai-telemetry"

    def __post_init__(self) -> None:
        if not _HAS_OTEL:  # pragma: no cover - requires optional dependency
            raise RuntimeError(
                "OpenTelemetry is not installed; install electripy[otel] to use OpenTelemetryAdapter",
            )
        self._tracer = _otel_trace.get_tracer(self.service_name)

    def emit_event(self, event: TelemetryEvent) -> None:
        span = _otel_trace.get_current_span()
        attrs = _sanitize_attributes(dict(event.attributes))
        span.add_event(  # type: ignore[union-attr]
            name=event.name,
            attributes={**attrs, "severity": event.severity.value},
            timestamp=int(event.timestamp.timestamp() * 1_000_000_000),
        )

    def increment(
        self,
        name: str,
        value: int = 1,
        *,
        attrs: dict[str, object] | None = None,
        ctx: TelemetryContext | None = None,
    ) -> None:  # pragma: no cover - simple mapping
        # For now, model counters as events; a production deployment can
        # attach these to a metrics exporter.
        now = datetime.now(tz=UTC)
        event = TelemetryEvent(
            name=f"metric.counter.{name}",
            timestamp=now,
            context=ctx
            or TelemetryContext(
                trace_id="",
                span_id=None,
                parent_span_id=None,
                request_id=None,
                actor_id=None,
                tenant_id=None,
                environment=None,
                tags={},
            ),
            attributes=_sanitize_attributes({"value": value, **(attrs or {})}),
            severity=Severity.DEBUG,
        )
        self.emit_event(event)

    def observe(
        self,
        name: str,
        value: float,
        *,
        attrs: dict[str, object] | None = None,
        ctx: TelemetryContext | None = None,
    ) -> None:  # pragma: no cover - simple mapping
        now = datetime.now(tz=UTC)
        event = TelemetryEvent(
            name=f"metric.histogram.{name}",
            timestamp=now,
            context=ctx
            or TelemetryContext(
                trace_id="",
                span_id=None,
                parent_span_id=None,
                request_id=None,
                actor_id=None,
                tenant_id=None,
                environment=None,
                tags={},
            ),
            attributes=_sanitize_attributes({"value": value, **(attrs or {})}),
            severity=Severity.DEBUG,
        )
        self.emit_event(event)

    def span(
        self,
        name: str,
        *,
        ctx: TelemetryContext | None = None,
        attrs: dict[str, object] | None = None,
    ) -> SpanContextManager:
        # For OpenTelemetry, defer to the tracer's context manager and
        # rely on upstream context propagation. The TelemetryContext is
        # not automatically mapped; callers can still emit explicit
        # events if they need a bridge.
        class _OtelSpan(SpanContextManager):
            def __init__(self, adapter: OpenTelemetryAdapter) -> None:
                self._adapter = adapter
                self._cm = adapter._tracer.start_as_current_span(name)  # type: ignore[union-attr]

            def __enter__(self) -> TelemetryContext:
                self._cm.__enter__()
                return ctx or TelemetryContext(
                    trace_id="",
                    span_id=None,
                    parent_span_id=None,
                    request_id=None,
                    actor_id=None,
                    tenant_id=None,
                    environment=None,
                    tags={},
                )

            def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
                self._cm.__exit__(exc_type, exc, tb)
                return None

            async def __aenter__(self) -> TelemetryContext:  # type: ignore[override]
                self.__enter__()
                return ctx or TelemetryContext(
                    trace_id="",
                    span_id=None,
                    parent_span_id=None,
                    request_id=None,
                    actor_id=None,
                    tenant_id=None,
                    environment=None,
                    tags={},
                )

            async def __aexit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
                self.__exit__(exc_type, exc, tb)

        return _OtelSpan(self)


__all__ = [
    "JsonlTelemetrySinkAdapter",
    "InMemoryTelemetryAdapter",
    "OpenTelemetryAdapter",
]
