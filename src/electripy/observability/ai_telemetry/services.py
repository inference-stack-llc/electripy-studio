"""Service and helper functions for AI telemetry.

This module provides utilities for:

- Creating and managing :class:`TelemetryContext` instances.
- Propagating correlation identifiers into outbound HTTP headers.
- Defining thin, standardised helpers for AI-specific events such as
  HTTP resilience, LLM gateway calls, policy decisions, and RAG
  evaluation runs.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator, Mapping, MutableMapping
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import UTC, datetime

from electripy.core.logging import get_logger

from .domain import CostRecord, Severity, TelemetryContext, TelemetryEvent
from .ports import CostEstimatorPort, TelemetryPort

logger = get_logger(__name__)

_current_ctx: ContextVar[TelemetryContext | None] = ContextVar("ai_telemetry_context", default=None)


def _generate_id(prefix: str) -> str:
    """Generate a stable-looking hexadecimal identifier.

    Args:
        prefix: Logical prefix used in the hash input.

    Returns:
        str: Hex string identifier.
    """

    raw = f"{prefix}-{os.urandom(16).hex()}"
    from hashlib import sha256

    return sha256(raw.encode("utf-8")).hexdigest()


def create_telemetry_context(
    *,
    environment: str | None = None,
    request_id: str | None = None,
    actor_id: str | None = None,
    tenant_id: str | None = None,
    tags: Mapping[str, str] | None = None,
) -> TelemetryContext:
    """Create a new root :class:`TelemetryContext`.

    Args:
        environment: Optional environment label.
        request_id: Optional request identifier; generated if omitted.
        actor_id: Optional identifier for the end-user or service actor.
        tenant_id: Optional tenant identifier.
        tags: Optional additional tags.

    Returns:
        TelemetryContext: New root context instance.
    """

    trace_id = _generate_id("trace")
    req_id = request_id or _generate_id("request")
    return TelemetryContext(
        trace_id=trace_id,
        span_id=None,
        parent_span_id=None,
        request_id=req_id,
        actor_id=actor_id,
        tenant_id=tenant_id,
        environment=environment,
        tags=dict(tags or {}),
    )


def set_current_telemetry_context(ctx: TelemetryContext | None) -> None:
    """Set the current telemetry context in a context variable.

    Args:
        ctx: Context to set as current, or ``None`` to clear.
    """

    _current_ctx.set(ctx)


def current_telemetry_context() -> TelemetryContext | None:
    """Return the current telemetry context, if any.

    Returns:
        TelemetryContext | None: Current context or ``None``.
    """

    return _current_ctx.get()


@contextlib.contextmanager
def scoped_telemetry_context(ctx: TelemetryContext) -> Iterator[TelemetryContext]:
    """Context manager that temporarily sets the current context.

    Args:
        ctx: Context to make current within the scope.

    Yields:
        TelemetryContext: The provided context.
    """

    token = _current_ctx.set(ctx)
    try:
        yield ctx
    finally:
        _current_ctx.reset(token)


def inject_context_headers(
    headers: MutableMapping[str, str],
    ctx: TelemetryContext | None = None,
) -> None:
    """Inject correlation headers into an outbound HTTP header mapping.

    Args:
        headers: Mutable header mapping to update in place.
        ctx: Optional context; if omitted, the current context is used.
    """

    context = ctx or current_telemetry_context()
    if context is None:
        return
    for key, value in context.to_headers().items():
        headers[key] = value


@dataclass(slots=True)
class TableCostEstimator(CostEstimatorPort):
    """Table-driven cost estimator.

    The estimator uses a simple mapping of ``(provider, model)`` to
    per-1K token rates. It is intentionally conservative and should be
    configured explicitly by the user.

    Example:
        estimator = TableCostEstimator(rates={("openai", "gpt-4.1"): (0.01, 0.03)})
    """

    rates: Mapping[tuple[str, str], tuple[float, float]]

    def estimate_cost(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostRecord:
        """Estimate the cost of an AI call.

        If no rate is configured for the given provider/model,
        ``estimated_cost_usd`` will be ``None``.
        """

        rate = self.rates.get((provider, model))
        if rate is None:
            return CostRecord(
                provider=provider,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                estimated_cost_usd=None,
            )

        input_rate, output_rate = rate
        cost = (input_tokens / 1000.0) * input_rate + (output_tokens / 1000.0) * output_rate
        return CostRecord(
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            estimated_cost_usd=cost,
        )


# AI-specific instrumentation helpers -------------------------------------------------


def record_http_retry_attempt(
    telemetry: TelemetryPort,
    *,
    attempt: int,
    max_attempts: int,
    url: str,
    status_code: int | None,
    ctx: TelemetryContext | None = None,
) -> None:
    """Record an HTTP retry attempt event.

    Args:
        telemetry: Telemetry port implementation.
        attempt: Attempt index (1-based).
        max_attempts: Maximum attempts configured.
        url: Target URL.
        status_code: Optional last seen status code.
        ctx: Optional correlation context.
    """

    event = TelemetryEvent(
        name="http.retry_attempt",
        timestamp=datetime.now(tz=UTC),
        context=ctx or current_telemetry_context() or create_telemetry_context(),
        attributes={
            "attempt": attempt,
            "max_attempts": max_attempts,
            "url": url,
            "status_code": status_code,
        },
        severity=Severity.INFO,
    )
    telemetry.emit_event(event)


def record_http_circuit_opened(
    telemetry: TelemetryPort,
    *,
    url: str,
    ctx: TelemetryContext | None = None,
) -> None:
    """Record that a circuit breaker has opened for an HTTP dependency."""

    event = TelemetryEvent(
        name="http.breaker_opened",
        timestamp=datetime.now(tz=UTC),
        context=ctx or current_telemetry_context() or create_telemetry_context(),
        attributes={"url": url},
        severity=Severity.WARNING,
    )
    telemetry.emit_event(event)


def record_llm_call(
    telemetry: TelemetryPort,
    *,
    provider: str,
    model: str,
    latency_ms: float,
    input_tokens: int,
    output_tokens: int,
    finish_reason: str,
    structured_output_valid: bool,
    ctx: TelemetryContext | None = None,
) -> None:
    """Record a summary event for an LLM call."""

    context = ctx or current_telemetry_context() or create_telemetry_context()
    telemetry.observe(
        "llm.latency_ms",
        latency_ms,
        attrs={"provider": provider, "model": model},
        ctx=context,
    )
    telemetry.increment(
        "llm.calls",
        attrs={"provider": provider, "model": model},
        ctx=context,
    )
    event = TelemetryEvent(
        name="llm.call",
        timestamp=datetime.now(tz=UTC),
        context=context,
        attributes={
            "provider": provider,
            "model": model,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "finish_reason": finish_reason,
            "structured_output_valid": structured_output_valid,
        },
        severity=Severity.INFO,
    )
    telemetry.emit_event(event)


def record_policy_decision(
    telemetry: TelemetryPort,
    *,
    decision: str,
    violation_codes: list[str] | None,
    redactions_applied: bool,
    ctx: TelemetryContext | None = None,
) -> None:
    """Record a policy gateway decision event."""

    event = TelemetryEvent(
        name="policy.decision",
        timestamp=datetime.now(tz=UTC),
        context=ctx or current_telemetry_context() or create_telemetry_context(),
        attributes={
            "decision": decision,
            "violation_codes": ",".join(violation_codes or []),
            "redactions_applied": redactions_applied,
        },
        severity=Severity.INFO,
    )
    telemetry.emit_event(event)


def record_rag_experiment_started(
    telemetry: TelemetryPort,
    *,
    experiment_id: str,
    ctx: TelemetryContext | None = None,
) -> None:
    """Record the start of a RAG evaluation experiment."""

    event = TelemetryEvent(
        name="rag_eval.experiment_started",
        timestamp=datetime.now(tz=UTC),
        context=ctx or current_telemetry_context() or create_telemetry_context(),
        attributes={"experiment_id": experiment_id},
        severity=Severity.INFO,
    )
    telemetry.emit_event(event)


def record_rag_experiment_finished(
    telemetry: TelemetryPort,
    *,
    experiment_id: str,
    metrics_summary: Mapping[str, float],
    ctx: TelemetryContext | None = None,
) -> None:
    """Record the completion of a RAG evaluation experiment."""

    event = TelemetryEvent(
        name="rag_eval.experiment_finished",
        timestamp=datetime.now(tz=UTC),
        context=ctx or current_telemetry_context() or create_telemetry_context(),
        attributes={"experiment_id": experiment_id, **dict(metrics_summary)},
        severity=Severity.INFO,
    )
    telemetry.emit_event(event)


__all__ = [
    "create_telemetry_context",
    "set_current_telemetry_context",
    "current_telemetry_context",
    "scoped_telemetry_context",
    "inject_context_headers",
    "TableCostEstimator",
    "record_http_retry_attempt",
    "record_http_circuit_opened",
    "record_llm_call",
    "record_policy_decision",
    "record_rag_experiment_started",
    "record_rag_experiment_finished",
]
