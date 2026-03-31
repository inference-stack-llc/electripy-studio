"""Ergonomic decorators for the observe package.

Thin, deterministic decorators for instrumenting functions and tool
callables.  They create spans automatically, record return status, and
attach metadata without hidden magic.

Usage::

    from electripy.observability.observe import (
        InMemoryTracer,
        ObservabilityService,
        observe_function,
        observe_tool,
    )

    tracer = InMemoryTracer()
    svc = ObservabilityService(tracer=tracer)

    @observe_function(svc, name="my_step")
    def do_work(x: int) -> int:
        return x * 2

    result = do_work(21)
    assert result == 42
"""

from __future__ import annotations

import asyncio
import functools
import time
from collections.abc import Callable, Mapping
from typing import ParamSpec, TypeVar

from .domain import AttributeValue, SpanKind, SpanStatus, SpanStatusCode
from .services import ObservabilityService

P = ParamSpec("P")
R = TypeVar("R")


# ---------------------------------------------------------------------------
# @observe_function
# ---------------------------------------------------------------------------


def observe_function(
    service: ObservabilityService,
    *,
    name: str | None = None,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that wraps a sync or async function in a traced span.

    The span is started before the function executes and ended after it
    returns (or raises).  If the function is a coroutine, the decorator
    produces an async wrapper automatically.

    Args:
        service: Observability service providing the tracer.
        name: Span name; defaults to the function's qualified name.
        kind: Semantic span kind.
        attributes: Initial span attributes.

    Returns:
        Decorator that preserves the original function's signature.

    Example::

        @observe_function(svc, kind=SpanKind.WORKFLOW)
        def run_pipeline(data: list[str]) -> int:
            return len(data)
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        span_name = name or fn.__qualname__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                cm = service.start_span(
                    span_name,
                    kind=kind,
                    attributes=dict(attributes) if attributes else {},
                )
                async with cm as span:
                    span.set_attribute("code.function", fn.__qualname__)
                    t0 = time.monotonic()
                    try:
                        result = await fn(*args, **kwargs)
                        elapsed_ms = (time.monotonic() - t0) * 1000.0
                        span.set_attribute("code.latency_ms", elapsed_ms)
                        span.set_status(SpanStatus(code=SpanStatusCode.OK))
                        return result  # type: ignore[no-any-return]
                    except BaseException:
                        elapsed_ms = (time.monotonic() - t0) * 1000.0
                        span.set_attribute("code.latency_ms", elapsed_ms)
                        raise

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            cm = service.start_span(
                span_name,
                kind=kind,
                attributes=dict(attributes) if attributes else {},
            )
            with cm as span:
                span.set_attribute("code.function", fn.__qualname__)
                t0 = time.monotonic()
                try:
                    result = fn(*args, **kwargs)
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    span.set_attribute("code.latency_ms", elapsed_ms)
                    span.set_status(SpanStatus(code=SpanStatusCode.OK))
                    return result
                except BaseException:
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    span.set_attribute("code.latency_ms", elapsed_ms)
                    raise

        return sync_wrapper

    return decorator


# ---------------------------------------------------------------------------
# @observe_tool
# ---------------------------------------------------------------------------


def observe_tool(
    service: ObservabilityService,
    *,
    tool_name: str | None = None,
    tool_version: str | None = None,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator that wraps a tool/function-call implementation in a
    :attr:`SpanKind.TOOL` span.

    The decorator records the tool name, version, execution latency,
    and outcome status (success or error type) as span attributes.

    Args:
        service: Observability service providing the tracer.
        tool_name: Logical tool name; defaults to the function name.
        tool_version: Optional tool version string.
        attributes: Extra initial span attributes.

    Returns:
        Decorator preserving the original function's signature.

    Example::

        @observe_tool(svc, tool_name="calculator")
        def add(a: int, b: int) -> int:
            return a + b
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        resolved_name = tool_name or fn.__name__

        if asyncio.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def async_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
                extra: dict[str, AttributeValue] = {"tool.name": resolved_name}
                if tool_version is not None:
                    extra["tool.version"] = tool_version
                if attributes:
                    extra.update(attributes)

                cm = service.start_span(
                    f"tool.{resolved_name}",
                    kind=SpanKind.TOOL,
                    attributes=extra,
                )
                async with cm as span:
                    t0 = time.monotonic()
                    try:
                        result = await fn(*args, **kwargs)
                        elapsed_ms = (time.monotonic() - t0) * 1000.0
                        span.set_attribute("tool.latency_ms", elapsed_ms)
                        span.set_attribute("tool.status", "success")
                        span.set_status(SpanStatus(code=SpanStatusCode.OK))
                        return result  # type: ignore[no-any-return]
                    except BaseException as exc:
                        elapsed_ms = (time.monotonic() - t0) * 1000.0
                        span.set_attribute("tool.latency_ms", elapsed_ms)
                        span.set_attribute("tool.status", "error")
                        span.set_attribute("tool.error_type", type(exc).__name__)
                        raise

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(fn)
        def sync_wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            extra: dict[str, AttributeValue] = {"tool.name": resolved_name}
            if tool_version is not None:
                extra["tool.version"] = tool_version
            if attributes:
                extra.update(attributes)

            cm = service.start_span(
                f"tool.{resolved_name}",
                kind=SpanKind.TOOL,
                attributes=extra,
            )
            with cm as span:
                t0 = time.monotonic()
                try:
                    result = fn(*args, **kwargs)
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    span.set_attribute("tool.latency_ms", elapsed_ms)
                    span.set_attribute("tool.status", "success")
                    span.set_status(SpanStatus(code=SpanStatusCode.OK))
                    return result
                except BaseException as exc:
                    elapsed_ms = (time.monotonic() - t0) * 1000.0
                    span.set_attribute("tool.latency_ms", elapsed_ms)
                    span.set_attribute("tool.status", "error")
                    span.set_attribute("tool.error_type", type(exc).__name__)
                    raise

        return sync_wrapper

    return decorator


__all__ = ["observe_function", "observe_tool"]
