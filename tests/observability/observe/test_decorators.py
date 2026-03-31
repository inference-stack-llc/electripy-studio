"""Tests for the observe package decorators."""

from __future__ import annotations

import pytest

from electripy.observability.observe.adapters import InMemoryTracer
from electripy.observability.observe.decorators import observe_function, observe_tool
from electripy.observability.observe.domain import SpanKind, SpanStatusCode
from electripy.observability.observe.services import ObservabilityService


class TestObserveFunction:
    """@observe_function decorator for sync and async functions."""

    def test_sync_function_creates_span(self) -> None:
        """A decorated sync function produces a finished span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        @observe_function(svc, name="my_step")
        def do_work(x: int) -> int:
            return x * 2

        result = do_work(21)

        assert result == 42
        assert len(tracer.finished_spans) == 1
        record = tracer.finished_spans[0]
        assert record.name == "my_step"
        assert "do_work" in record.attributes["code.function"]
        assert record.status.code == SpanStatusCode.OK
        assert "code.latency_ms" in record.attributes

    def test_sync_function_default_name(self) -> None:
        """Without explicit name, the function qualname is used."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        @observe_function(svc)
        def helper() -> str:
            return "ok"

        helper()
        assert "helper" in tracer.finished_spans[0].name

    def test_sync_function_records_exception(self) -> None:
        """Exceptions propagate and are recorded on the span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        @observe_function(svc)
        def fail() -> None:
            raise RuntimeError("boom")

        with pytest.raises(RuntimeError, match="boom"):
            fail()

        assert len(tracer.finished_spans) == 1
        record = tracer.finished_spans[0]
        assert record.status.code == SpanStatusCode.ERROR
        assert len(record.exceptions) == 1

    def test_sync_function_with_kind(self) -> None:
        """The kind parameter is forwarded to the span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        @observe_function(svc, kind=SpanKind.WORKFLOW)
        def pipeline() -> None:
            pass

        pipeline()
        assert tracer.finished_spans[0].kind == SpanKind.WORKFLOW

    async def test_async_function_creates_span(self) -> None:
        """A decorated async function produces a finished span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        @observe_function(svc, name="async_step")
        async def do_async(x: int) -> int:
            return x + 1

        result = await do_async(10)

        assert result == 11
        assert len(tracer.finished_spans) == 1
        record = tracer.finished_spans[0]
        assert record.name == "async_step"
        assert record.status.code == SpanStatusCode.OK

    async def test_async_function_records_exception(self) -> None:
        """Async exceptions propagate and are recorded on the span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        @observe_function(svc)
        async def afail() -> None:
            raise ValueError("async boom")

        with pytest.raises(ValueError, match="async boom"):
            await afail()

        assert tracer.finished_spans[0].status.code == SpanStatusCode.ERROR


class TestObserveTool:
    """@observe_tool decorator for sync and async tool functions."""

    def test_sync_tool_creates_span(self) -> None:
        """A decorated sync tool produces a TOOL span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        @observe_tool(svc, tool_name="calculator", tool_version="1.0")
        def add(a: int, b: int) -> int:
            return a + b

        result = add(3, 4)

        assert result == 7
        assert len(tracer.finished_spans) == 1
        record = tracer.finished_spans[0]
        assert record.kind == SpanKind.TOOL
        assert record.attributes["tool.name"] == "calculator"
        assert record.attributes["tool.version"] == "1.0"
        assert record.attributes["tool.status"] == "success"
        assert "tool.latency_ms" in record.attributes

    def test_sync_tool_default_name(self) -> None:
        """Without explicit tool_name, the function name is used."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        @observe_tool(svc)
        def search(query: str) -> str:
            return f"results for {query}"

        search("test")
        assert tracer.finished_spans[0].attributes["tool.name"] == "search"

    def test_sync_tool_records_error(self) -> None:
        """Tool errors are recorded with status and error_type."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        @observe_tool(svc, tool_name="broken")
        def broken_tool() -> None:
            raise OSError("disk full")

        with pytest.raises(IOError, match="disk full"):
            broken_tool()

        record = tracer.finished_spans[0]
        assert record.attributes["tool.status"] == "error"
        assert record.attributes["tool.error_type"] == "OSError"
        assert record.status.code == SpanStatusCode.ERROR

    async def test_async_tool_creates_span(self) -> None:
        """A decorated async tool produces a TOOL span."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        @observe_tool(svc, tool_name="async_calc")
        async def multiply(a: int, b: int) -> int:
            return a * b

        result = await multiply(6, 7)

        assert result == 42
        record = tracer.finished_spans[0]
        assert record.attributes["tool.name"] == "async_calc"
        assert record.attributes["tool.status"] == "success"

    async def test_async_tool_records_error(self) -> None:
        """Async tool errors are recorded."""
        tracer = InMemoryTracer()
        svc = ObservabilityService(tracer=tracer)

        @observe_tool(svc, tool_name="async_broken")
        async def broken() -> None:
            raise TypeError("wrong type")

        with pytest.raises(TypeError, match="wrong type"):
            await broken()

        record = tracer.finished_spans[0]
        assert record.attributes["tool.status"] == "error"
        assert record.attributes["tool.error_type"] == "TypeError"
