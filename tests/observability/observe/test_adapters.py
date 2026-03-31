"""Tests for the observe package adapters (NoOp, InMemory)."""

from __future__ import annotations

import pytest

from electripy.observability.observe.adapters import (
    InMemorySpan,
    InMemoryTracer,
    NoOpSpan,
    NoOpTracer,
    SpanRecord,
)
from electripy.observability.observe.domain import (
    RedactionPolicy,
    RedactionRule,
    RedactionRuleKind,
    SpanKind,
    SpanStatus,
    SpanStatusCode,
    TraceContext,
)
from electripy.observability.observe.errors import TracerConfigError
from electripy.observability.observe.redaction import DefaultRedactor


class TestNoOpTracer:
    """NoOpTracer and NoOpSpan are zero-cost stubs."""

    def test_start_span_returns_noop(self) -> None:
        """start_span returns a NoOpSpan."""
        tracer = NoOpTracer()
        span = tracer.start_span("test")
        assert isinstance(span, NoOpSpan)

    def test_noop_span_has_context(self) -> None:
        """NoOpSpan still carries a valid TraceContext."""
        tracer = NoOpTracer()
        span = tracer.start_span("test")
        assert span.context.trace_id
        assert span.context.span_id

    def test_noop_operations_do_not_raise(self) -> None:
        """All NoOpSpan methods are silently ignored."""
        tracer = NoOpTracer()
        span = tracer.start_span("test")

        span.set_attribute("k", "v")
        span.set_attributes({"a": 1, "b": 2})
        span.add_event("evt", attributes={"x": "y"})
        span.record_exception(ValueError("boom"))
        span.set_status(SpanStatus(code=SpanStatusCode.OK))
        span.end()
        span.end()  # double-end is also a no-op

    def test_parent_context_is_respected(self) -> None:
        """When a parent context is given, the child inherits it."""
        tracer = NoOpTracer()
        parent = TraceContext(trace_id="t1", span_id="s1")
        span = tracer.start_span("child", parent=parent)

        assert span.context.trace_id == "t1"
        assert span.context.parent_span_id == "s1"


class TestInMemoryTracer:
    """InMemoryTracer collects finished spans for assertions."""

    def test_start_and_end_records_span(self) -> None:
        """A started-then-ended span appears in finished_spans."""
        tracer = InMemoryTracer()
        span = tracer.start_span("work", kind=SpanKind.LLM)
        span.set_attribute("gen_ai.model", "gpt-4o")
        span.end()

        assert len(tracer.finished_spans) == 1
        record = tracer.finished_spans[0]
        assert record.name == "work"
        assert record.kind == SpanKind.LLM
        assert record.attributes["gen_ai.model"] == "gpt-4o"
        assert record.end_time is not None

    def test_double_end_is_idempotent(self) -> None:
        """Calling end() twice does not produce duplicate records."""
        tracer = InMemoryTracer()
        span = tracer.start_span("work")
        span.end()
        span.end()

        assert len(tracer.finished_spans) == 1

    def test_record_exception_sets_error_status(self) -> None:
        """record_exception sets the span status to ERROR."""
        tracer = InMemoryTracer()
        span = tracer.start_span("work")
        span.record_exception(ValueError("bad input"))
        span.end()

        record = tracer.finished_spans[0]
        assert record.status.code == SpanStatusCode.ERROR
        assert "bad input" in (record.status.description or "")
        assert len(record.exceptions) == 1
        assert isinstance(record.exceptions[0], ValueError)

    def test_events_are_recorded(self) -> None:
        """add_event produces event records."""
        tracer = InMemoryTracer()
        span = tracer.start_span("work")
        span.add_event("policy.check", attributes={"action": "allow"})
        span.end()

        events = tracer.finished_spans[0].events
        assert len(events) == 1
        assert events[0][0] == "policy.check"
        assert events[0][1] == {"action": "allow"}

    def test_nested_spans_have_correct_parent(self) -> None:
        """Child spans reference the parent's span_id."""
        tracer = InMemoryTracer()

        parent = tracer.start_span("parent")
        child = tracer.start_span("child", parent=parent.context)
        child.end()
        parent.end()

        assert len(tracer.finished_spans) == 2
        child_record = tracer.finished_spans[0]
        parent_record = tracer.finished_spans[1]

        assert child_record.context.parent_span_id == parent_record.context.span_id

    def test_initial_attributes_are_set(self) -> None:
        """Attributes passed to start_span are recorded."""
        tracer = InMemoryTracer()
        span = tracer.start_span("work", attributes={"k": "v"})
        span.end()

        assert tracer.finished_spans[0].attributes["k"] == "v"

    def test_set_status(self) -> None:
        """set_status overrides the default status."""
        tracer = InMemoryTracer()
        span = tracer.start_span("work")
        span.set_status(SpanStatus(code=SpanStatusCode.OK, description="all good"))
        span.end()

        assert tracer.finished_spans[0].status.code == SpanStatusCode.OK

    def test_redaction_on_end(self) -> None:
        """When a redactor is configured, attributes are redacted on end."""
        policy = RedactionPolicy(
            rules=(
                RedactionRule(kind=RedactionRuleKind.EXACT, match="secret"),
            ),
        )
        redactor = DefaultRedactor(policy=policy)
        tracer = InMemoryTracer(redactor=redactor)

        span = tracer.start_span("work")
        span.set_attribute("secret", "s3cr3t")
        span.set_attribute("model", "gpt-4o")
        span.end()

        record = tracer.finished_spans[0]
        assert record.attributes["secret"] == "[REDACTED]"
        assert record.attributes["model"] == "gpt-4o"


class TestOpenTelemetryTracer:
    """OpenTelemetryTracer graceful degradation."""

    def test_missing_otel_raises_config_error(self) -> None:
        """Constructing without opentelemetry installed raises TracerConfigError."""
        # This test verifies graceful degradation.  If opentelemetry
        # happens to be installed in the test environment the adapter
        # will succeed — which is also acceptable.
        try:
            import opentelemetry  # noqa: F401

            pytest.skip("opentelemetry is installed; cannot test missing-dep path")
        except ImportError:
            pass

        from electripy.observability.observe.adapters import OpenTelemetryTracer

        with pytest.raises(TracerConfigError, match="opentelemetry"):
            OpenTelemetryTracer()
