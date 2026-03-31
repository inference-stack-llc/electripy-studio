"""Service / orchestration layer for structured observability.

This module provides :class:`ObservabilityService` — the recommended
entry-point for instrumenting AI, agent, tool, retrieval, policy, and
MCP workloads.  It also exposes ``observe_span`` and ``aobserve_span``
context managers for lightweight instrumentation.

The service manages a ``ContextVar``-based span stack so that nested
spans are automatically parented correctly.

Usage::

    from electripy.observability.observe import (
        ObservabilityService,
        InMemoryTracer,
    )

    tracer = InMemoryTracer()
    svc = ObservabilityService(tracer=tracer)

    with svc.start_llm_span(provider="openai", model="gpt-4o") as span:
        span.set_attribute("gen_ai.usage.input_tokens", 120)
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncIterator, Iterator, Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any

from electripy.core.logging import get_logger

from .adapters import NoOpTracer
from .domain import (
    Attributes,
    AttributeValue,
    GenAIRequestMetadata,
    GenAIResponseMetadata,
    MCPMetadata,
    PolicyDecisionMetadata,
    RetrievalMetadata,
    SpanKind,
    ToolInvocationMetadata,
)
from .ports import SpanPort, TracerPort

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# ContextVar for current span propagation
# ---------------------------------------------------------------------------

_current_span: ContextVar[SpanPort | None] = ContextVar("observe_current_span", default=None)


def current_span() -> SpanPort | None:
    """Return the current active span, if any.

    Returns:
        The active :class:`SpanPort` or ``None``.
    """
    return _current_span.get()


# ---------------------------------------------------------------------------
# ObservabilityService
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ObservabilityService:
    """High-level instrumentation service for AI workloads.

    ``ObservabilityService`` wraps a :class:`TracerPort` and provides
    semantic helpers that create correctly-typed, correctly-parented
    spans for common AI operations.

    By default a :class:`NoOpTracer` is used, meaning instrumentation
    is zero-cost when no backend is configured.

    Attributes:
        tracer: Underlying tracer implementation.
    """

    tracer: TracerPort = field(default_factory=NoOpTracer)

    # -- generic span helpers ----------------------------------------------

    def start_span(
        self,
        name: str,
        *,
        kind: SpanKind = SpanKind.INTERNAL,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> _SpanContextManager:
        """Start a span and return a sync/async context manager.

        The returned context manager automatically parents the span
        under the current active span and restores the previous span
        on exit.

        Args:
            name: Span name.
            kind: Semantic span kind.
            attributes: Initial attributes.

        Returns:
            A context manager yielding the :class:`SpanPort`.
        """
        return _SpanContextManager(
            tracer=self.tracer,
            name=name,
            kind=kind,
            attributes=dict(attributes) if attributes else {},
        )

    # -- semantic span helpers ---------------------------------------------

    def start_workflow_span(
        self,
        name: str,
        *,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> _SpanContextManager:
        """Start a workflow span.

        Args:
            name: Workflow name.
            attributes: Optional attributes.

        Returns:
            Context manager yielding the span.
        """
        attrs = dict(attributes) if attributes else {}
        attrs.setdefault("observe.span.semantic", "workflow")
        return self.start_span(name, kind=SpanKind.WORKFLOW, attributes=attrs)

    def start_agent_span(
        self,
        name: str,
        *,
        agent_id: str | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> _SpanContextManager:
        """Start a span representing an agent step.

        Args:
            name: Agent span name.
            agent_id: Logical agent identifier.
            attributes: Optional attributes.

        Returns:
            Context manager yielding the span.
        """
        attrs = dict(attributes) if attributes else {}
        attrs.setdefault("observe.span.semantic", "agent")
        if agent_id is not None:
            attrs["agent.id"] = agent_id
        return self.start_span(name, kind=SpanKind.AGENT, attributes=attrs)

    def start_llm_span(
        self,
        *,
        provider: str,
        model: str,
        request_meta: GenAIRequestMetadata | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> _SpanContextManager:
        """Start a span for an LLM call.

        If *request_meta* is provided its attributes are merged into
        the span.  Otherwise minimal ``gen_ai.system`` and
        ``gen_ai.request.model`` attributes are set.

        Args:
            provider: Logical provider name.
            model: Model identifier.
            request_meta: Optional rich request metadata.
            attributes: Optional extra attributes.

        Returns:
            Context manager yielding the span.
        """
        attrs: dict[str, AttributeValue] = {}
        if request_meta is not None:
            attrs.update(request_meta.to_attributes())
        else:
            attrs["gen_ai.system"] = provider
            attrs["gen_ai.request.model"] = model
        if attributes:
            attrs.update(attributes)
        attrs.setdefault("observe.span.semantic", "llm")
        return self.start_span(f"llm.{provider}.{model}", kind=SpanKind.LLM, attributes=attrs)

    def start_tool_span(
        self,
        tool_name: str,
        *,
        tool_version: str | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> _SpanContextManager:
        """Start a span for a tool invocation.

        Args:
            tool_name: Tool name.
            tool_version: Optional tool version.
            attributes: Optional extra attributes.

        Returns:
            Context manager yielding the span.
        """
        attrs: dict[str, AttributeValue] = {"tool.name": tool_name}
        if tool_version is not None:
            attrs["tool.version"] = tool_version
        if attributes:
            attrs.update(attributes)
        attrs.setdefault("observe.span.semantic", "tool")
        return self.start_span(f"tool.{tool_name}", kind=SpanKind.TOOL, attributes=attrs)

    def start_retrieval_span(
        self,
        source: str,
        *,
        meta: RetrievalMetadata | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> _SpanContextManager:
        """Start a span for a retrieval / RAG operation.

        Args:
            source: Data source name.
            meta: Optional rich retrieval metadata.
            attributes: Optional extra attributes.

        Returns:
            Context manager yielding the span.
        """
        attrs: dict[str, AttributeValue] = {}
        if meta is not None:
            attrs.update(meta.to_attributes())
        else:
            attrs["retrieval.source"] = source
        if attributes:
            attrs.update(attributes)
        attrs.setdefault("observe.span.semantic", "retrieval")
        return self.start_span(f"retrieval.{source}", kind=SpanKind.RETRIEVAL, attributes=attrs)

    def start_mcp_span(
        self,
        server_name: str,
        *,
        meta: MCPMetadata | None = None,
        attributes: Mapping[str, AttributeValue] | None = None,
    ) -> _SpanContextManager:
        """Start a span for an MCP server/tool interaction.

        Args:
            server_name: MCP server name.
            meta: Optional rich MCP metadata.
            attributes: Optional extra attributes.

        Returns:
            Context manager yielding the span.
        """
        attrs: dict[str, AttributeValue] = {}
        if meta is not None:
            attrs.update(meta.to_attributes())
        else:
            attrs["mcp.server_name"] = server_name
        if attributes:
            attrs.update(attributes)
        attrs.setdefault("observe.span.semantic", "mcp")
        return self.start_span(f"mcp.{server_name}", kind=SpanKind.MCP, attributes=attrs)

    # -- event helpers (no span lifecycle) ---------------------------------

    def record_policy_decision(
        self,
        meta: PolicyDecisionMetadata,
        *,
        span: SpanPort | None = None,
    ) -> None:
        """Annotate the current (or given) span with a policy decision.

        If no span is active, the event is silently discarded.

        Args:
            meta: Policy decision metadata.
            span: Explicit span to annotate; defaults to the current span.
        """
        target = span or current_span()
        if target is None:
            return
        target.add_event("policy.decision", attributes=meta.to_attributes())

    def record_exception(
        self,
        exception: BaseException,
        *,
        span: SpanPort | None = None,
        attributes: Attributes | None = None,
    ) -> None:
        """Record an exception on the current (or given) span.

        Args:
            exception: The exception to record.
            span: Explicit span; defaults to the current span.
            attributes: Optional extra attributes.
        """
        target = span or current_span()
        if target is None:
            return
        target.record_exception(exception, attributes=attributes)

    def annotate_span(
        self,
        attributes: Mapping[str, AttributeValue],
        *,
        span: SpanPort | None = None,
    ) -> None:
        """Set additional attributes on the current (or given) span.

        Args:
            attributes: Attributes to set.
            span: Explicit span; defaults to the current span.
        """
        target = span or current_span()
        if target is None:
            return
        target.set_attributes(attributes)

    def record_llm_response(
        self,
        meta: GenAIResponseMetadata,
        *,
        span: SpanPort | None = None,
    ) -> None:
        """Annotate a span with LLM response metadata.

        Args:
            meta: Response metadata.
            span: Explicit span; defaults to the current span.
        """
        target = span or current_span()
        if target is None:
            return
        target.set_attributes(meta.to_attributes())

    def record_tool_result(
        self,
        meta: ToolInvocationMetadata,
        *,
        span: SpanPort | None = None,
    ) -> None:
        """Annotate a span with tool invocation result metadata.

        Args:
            meta: Tool invocation metadata.
            span: Explicit span; defaults to the current span.
        """
        target = span or current_span()
        if target is None:
            return
        target.set_attributes(meta.to_attributes())


# ---------------------------------------------------------------------------
# _SpanContextManager — sync and async
# ---------------------------------------------------------------------------


class _SpanContextManager:
    """Combined sync/async context manager for a traced span.

    On entry, a new span is started and set as the current span.
    On exit, the span is ended and the previous span is restored.
    If an exception propagates, it is recorded on the span.
    """

    __slots__ = ("_tracer", "_name", "_kind", "_attributes", "_span", "_token")

    def __init__(
        self,
        *,
        tracer: TracerPort,
        name: str,
        kind: SpanKind,
        attributes: dict[str, AttributeValue],
    ) -> None:
        self._tracer = tracer
        self._name = name
        self._kind = kind
        self._attributes = attributes
        self._span: SpanPort | None = None
        self._token: Any = None

    def __enter__(self) -> SpanPort:
        parent = current_span()
        parent_ctx = parent.context if parent is not None else None
        self._span = self._tracer.start_span(
            self._name,
            kind=self._kind,
            parent=parent_ctx,
            attributes=self._attributes,
        )
        self._token = _current_span.set(self._span)
        return self._span

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        span = self._span
        if span is not None:
            if exc_val is not None:
                span.record_exception(exc_val)
            elif span is not None:
                # Only set OK if no exception and status was not already set.
                pass
            span.end()
        if self._token is not None:
            _current_span.reset(self._token)

    async def __aenter__(self) -> SpanPort:
        return self.__enter__()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: Any,
    ) -> None:
        self.__exit__(exc_type, exc_val, exc_tb)


# ---------------------------------------------------------------------------
# Module-level context-manager helpers
# ---------------------------------------------------------------------------


@contextlib.contextmanager
def observe_span(
    tracer: TracerPort,
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> Iterator[SpanPort]:
    """Synchronous context manager for a traced span.

    This is a convenience wrapper around :class:`ObservabilityService`
    for callers who want lightweight, standalone instrumentation.

    Args:
        tracer: Tracer implementation.
        name: Span name.
        kind: Semantic span kind.
        attributes: Initial attributes.

    Yields:
        The active :class:`SpanPort`.
    """
    parent = current_span()
    parent_ctx = parent.context if parent is not None else None
    span = tracer.start_span(
        name,
        kind=kind,
        parent=parent_ctx,
        attributes=dict(attributes or {}),
    )
    token = _current_span.set(span)
    try:
        yield span
    except BaseException as exc:
        span.record_exception(exc)
        raise
    finally:
        span.end()
        _current_span.reset(token)


@contextlib.asynccontextmanager
async def aobserve_span(
    tracer: TracerPort,
    name: str,
    *,
    kind: SpanKind = SpanKind.INTERNAL,
    attributes: Mapping[str, AttributeValue] | None = None,
) -> AsyncIterator[SpanPort]:
    """Asynchronous context manager for a traced span.

    Async counterpart of :func:`observe_span`.

    Args:
        tracer: Tracer implementation.
        name: Span name.
        kind: Semantic span kind.
        attributes: Initial attributes.

    Yields:
        The active :class:`SpanPort`.
    """
    parent = current_span()
    parent_ctx = parent.context if parent is not None else None
    span = tracer.start_span(
        name,
        kind=kind,
        parent=parent_ctx,
        attributes=dict(attributes or {}),
    )
    token = _current_span.set(span)
    try:
        yield span
    except BaseException as exc:
        span.record_exception(exc)
        raise
    finally:
        span.end()
        _current_span.reset(token)


__all__ = [
    "current_span",
    "ObservabilityService",
    "observe_span",
    "aobserve_span",
]
