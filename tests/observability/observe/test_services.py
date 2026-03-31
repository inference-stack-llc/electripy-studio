"""Tests for the observe package ObservabilityService and context managers."""

from __future__ import annotations

import pytest

from electripy.observability.observe.adapters import InMemoryTracer, NoOpTracer
from electripy.observability.observe.domain import (
    GenAIRequestMetadata,
    GenAIResponseMetadata,
    MCPMetadata,
    PolicyDecisionMetadata,
    RetrievalMetadata,
    SpanKind,
    SpanStatusCode,
    ToolInvocationMetadata,
)
from electripy.observability.observe.services import (
    ObservabilityService,
    aobserve_span,
    current_span,
    observe_span,
)


class TestObservabilityServiceNoOp:
    """ObservabilityService with default NoOpTracer."""

    def test_default_tracer_is_noop(self) -> None:
        """The default service uses NoOpTracer."""
        svc = ObservabilityService()
        assert isinstance(svc.tracer, NoOpTracer)

    def test_noop_span_does_not_raise(self) -> None:
        """Using the service with a NoOp tracer is silent."""
        svc = ObservabilityService()
        with svc.start_llm_span(provider="openai", model="gpt-4o") as span:
            span.set_attribute("k", "v")

    def test_record_policy_without_span_is_silent(self) -> None:
        """record_policy_decision with no active span does not raise."""
        svc = ObservabilityService()
        meta = PolicyDecisionMetadata(action="allow")
        svc.record_policy_decision(meta)

    def test_record_exception_without_span_is_silent(self) -> None:
        """record_exception with no active span does not raise."""
        svc = ObservabilityService()
        svc.record_exception(ValueError("boom"))


class TestObservabilityServiceInMemory:
    """ObservabilityService with InMemoryTracer for full verification."""

    def test_start_llm_span(self) -> None:
        """start_llm_span produces a span with gen_ai attributes."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        with svc.start_llm_span(provider="openai", model="gpt-4o") as span:
            span.set_attribute("gen_ai.usage.input_tokens", 120)

        assert len(tracer.finished_spans) == 1
        record = tracer.finished_spans[0]
        assert record.kind == SpanKind.LLM
        assert record.attributes["gen_ai.system"] == "openai"
        assert record.attributes["gen_ai.request.model"] == "gpt-4o"
        assert record.attributes["gen_ai.usage.input_tokens"] == 120

    def test_start_llm_span_with_request_meta(self) -> None:
        """start_llm_span accepts GenAIRequestMetadata."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)
        meta = GenAIRequestMetadata(
            provider="anthropic",
            model="claude-3",
            temperature=0.5,
        )

        with svc.start_llm_span(
            provider="anthropic", model="claude-3", request_meta=meta
        ):
            pass

        attrs = tracer.finished_spans[0].attributes
        assert attrs["gen_ai.request.temperature"] == 0.5

    def test_start_tool_span(self) -> None:
        """start_tool_span produces a TOOL span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        with svc.start_tool_span("calculator", tool_version="1.0"):
            pass

        record = tracer.finished_spans[0]
        assert record.kind == SpanKind.TOOL
        assert record.attributes["tool.name"] == "calculator"
        assert record.attributes["tool.version"] == "1.0"

    def test_start_retrieval_span(self) -> None:
        """start_retrieval_span produces a RETRIEVAL span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        with svc.start_retrieval_span("pinecone"):
            pass

        record = tracer.finished_spans[0]
        assert record.kind == SpanKind.RETRIEVAL
        assert record.attributes["retrieval.source"] == "pinecone"

    def test_start_retrieval_span_with_meta(self) -> None:
        """start_retrieval_span accepts RetrievalMetadata."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)
        meta = RetrievalMetadata(source="pinecone", top_k=5, results_returned=3)

        with svc.start_retrieval_span("pinecone", meta=meta):
            pass

        attrs = tracer.finished_spans[0].attributes
        assert attrs["retrieval.top_k"] == 5
        assert attrs["retrieval.results_returned"] == 3

    def test_start_agent_span(self) -> None:
        """start_agent_span produces an AGENT span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        with svc.start_agent_span("planner", agent_id="agent-001"):
            pass

        record = tracer.finished_spans[0]
        assert record.kind == SpanKind.AGENT
        assert record.attributes["agent.id"] == "agent-001"

    def test_start_workflow_span(self) -> None:
        """start_workflow_span produces a WORKFLOW span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        with svc.start_workflow_span("ingest_pipeline"):
            pass

        record = tracer.finished_spans[0]
        assert record.kind == SpanKind.WORKFLOW

    def test_start_mcp_span(self) -> None:
        """start_mcp_span produces an MCP span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        with svc.start_mcp_span("code-server"):
            pass

        record = tracer.finished_spans[0]
        assert record.kind == SpanKind.MCP
        assert record.attributes["mcp.server_name"] == "code-server"

    def test_start_mcp_span_with_meta(self) -> None:
        """start_mcp_span accepts MCPMetadata."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)
        meta = MCPMetadata(
            server_name="code-server",
            tool_name="run_code",
            protocol_version="1.0",
        )

        with svc.start_mcp_span("code-server", meta=meta):
            pass

        attrs = tracer.finished_spans[0].attributes
        assert attrs["mcp.tool_name"] == "run_code"

    def test_nested_spans(self) -> None:
        """Nested spans are parented correctly via ContextVar."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        with svc.start_workflow_span("pipeline") as outer:
            with svc.start_llm_span(provider="openai", model="gpt-4o") as inner:
                pass

        assert len(tracer.finished_spans) == 2
        inner_record = tracer.finished_spans[0]
        outer_record = tracer.finished_spans[1]

        assert inner_record.context.parent_span_id == outer_record.context.span_id

    def test_deeply_nested_spans(self) -> None:
        """Three levels of nesting maintain correct parentage."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        with svc.start_workflow_span("pipeline") as w:
            with svc.start_agent_span("agent") as a:
                with svc.start_tool_span("calc") as t:
                    pass

        assert len(tracer.finished_spans) == 3
        tool_rec = tracer.finished_spans[0]
        agent_rec = tracer.finished_spans[1]
        workflow_rec = tracer.finished_spans[2]

        assert tool_rec.context.parent_span_id == agent_rec.context.span_id
        assert agent_rec.context.parent_span_id == workflow_rec.context.span_id

    def test_exception_is_recorded_on_span(self) -> None:
        """An exception propagating through a span is recorded."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        with pytest.raises(ValueError, match="bad"):
            with svc.start_llm_span(provider="openai", model="gpt-4o"):
                raise ValueError("bad")

        record = tracer.finished_spans[0]
        assert record.status.code == SpanStatusCode.ERROR
        assert len(record.exceptions) == 1

    def test_record_policy_decision_on_active_span(self) -> None:
        """record_policy_decision annotates the active span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)
        meta = PolicyDecisionMetadata(
            action="deny",
            violation_codes=["PII"],
            redactions_applied=True,
        )

        with svc.start_workflow_span("check"):
            svc.record_policy_decision(meta)

        events = tracer.finished_spans[0].events
        assert any(e[0] == "policy.decision" for e in events)

    def test_record_exception_on_active_span(self) -> None:
        """record_exception annotates the active span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        with svc.start_workflow_span("work") as span:
            svc.record_exception(RuntimeError("oops"))

        record = tracer.finished_spans[0]
        assert record.status.code == SpanStatusCode.ERROR

    def test_annotate_span(self) -> None:
        """annotate_span adds attributes to the active span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        with svc.start_workflow_span("work"):
            svc.annotate_span({"custom.key": "value"})

        assert tracer.finished_spans[0].attributes["custom.key"] == "value"

    def test_record_llm_response(self) -> None:
        """record_llm_response annotates the active span with response metadata."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)
        meta = GenAIResponseMetadata(output_tokens=50, finish_reason="stop")

        with svc.start_llm_span(provider="openai", model="gpt-4o"):
            svc.record_llm_response(meta)

        attrs = tracer.finished_spans[0].attributes
        assert attrs["gen_ai.usage.output_tokens"] == 50

    def test_record_tool_result(self) -> None:
        """record_tool_result annotates the active span with tool metadata."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)
        meta = ToolInvocationMetadata(tool_name="calc", status="success", latency_ms=5.0)

        with svc.start_tool_span("calc"):
            svc.record_tool_result(meta)

        attrs = tracer.finished_spans[0].attributes
        assert attrs["tool.status"] == "success"

    def test_current_span_is_none_outside_context(self) -> None:
        """current_span returns None when no span is active."""
        assert current_span() is None

    def test_current_span_restores_after_exit(self) -> None:
        """After a span context exits, the previous span is restored."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        assert current_span() is None

        with svc.start_workflow_span("outer") as outer:
            assert current_span() is outer
            with svc.start_llm_span(provider="o", model="m") as inner:
                assert current_span() is inner
            assert current_span() is outer

        assert current_span() is None


class TestObserveSpanFunctions:
    """Module-level observe_span / aobserve_span helpers."""

    def test_observe_span_sync(self) -> None:
        """observe_span creates and ends a span synchronously."""
        tracer = InMemoryTracer()

        with observe_span(tracer, "work", kind=SpanKind.WORKFLOW) as span:
            span.set_attribute("step", "first")

        assert len(tracer.finished_spans) == 1
        assert tracer.finished_spans[0].attributes["step"] == "first"

    def test_observe_span_records_exception(self) -> None:
        """observe_span records exceptions on the span."""
        tracer = InMemoryTracer()

        with pytest.raises(RuntimeError, match="fail"):
            with observe_span(tracer, "work") as span:
                raise RuntimeError("fail")

        assert tracer.finished_spans[0].status.code == SpanStatusCode.ERROR

    async def test_aobserve_span_async(self) -> None:
        """aobserve_span creates and ends a span asynchronously."""
        tracer = InMemoryTracer()

        async with aobserve_span(tracer, "async_work", kind=SpanKind.LLM) as span:
            span.set_attribute("gen_ai.model", "gpt-4o")

        assert len(tracer.finished_spans) == 1
        assert tracer.finished_spans[0].attributes["gen_ai.model"] == "gpt-4o"

    async def test_aobserve_span_records_exception(self) -> None:
        """aobserve_span records exceptions on async spans."""
        tracer = InMemoryTracer()

        with pytest.raises(ValueError, match="async_fail"):
            async with aobserve_span(tracer, "work"):
                raise ValueError("async_fail")

        assert tracer.finished_spans[0].status.code == SpanStatusCode.ERROR

    def test_observe_span_nested(self) -> None:
        """Nested observe_span calls parent correctly."""
        tracer = InMemoryTracer()

        with observe_span(tracer, "outer") as outer:
            with observe_span(tracer, "inner") as inner:
                pass

        assert len(tracer.finished_spans) == 2
        inner_rec = tracer.finished_spans[0]
        outer_rec = tracer.finished_spans[1]
        assert inner_rec.context.parent_span_id == outer_rec.context.span_id
