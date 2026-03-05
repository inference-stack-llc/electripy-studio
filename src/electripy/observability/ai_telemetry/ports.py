"""Ports (Protocols) for telemetry sinks and cost estimators.

These Protocols define the minimal surface area required by the
orchestration layer. Concrete adapters (for example, JSONL sinks or
OpenTelemetry integrations) must implement these interfaces so that the
underlying telemetry backend can be swapped without changing business
logic.

Example:
    from electripy.observability.ai_telemetry.domain import TelemetryEvent
    from electripy.observability.ai_telemetry.ports import TelemetryPort

    class MyTelemetrySink(TelemetryPort):
        def emit_event(self, event: TelemetryEvent) -> None:
            ...
"""

from __future__ import annotations

from contextlib import AbstractAsyncContextManager, AbstractContextManager
from typing import Protocol, runtime_checkable

from .domain import CostRecord, TelemetryContext, TelemetryEvent


class SpanContextManager(
    AbstractContextManager[TelemetryContext], AbstractAsyncContextManager[TelemetryContext]
):
    """Combined sync/async context manager for telemetry spans.

    Implementations typically record span start and end events and
    update the active :class:`TelemetryContext`.
    """

    # Protocol-level base; concrete adapters implement the actual
    # context manager methods.


@runtime_checkable
class TelemetryPort(Protocol):
    """Port for emitting telemetry events and metrics.

    Implementations must be safe-by-default and avoid logging raw
    prompts or responses. They should rely on redaction or hashing at
    the adapter layer.
    """

    def emit_event(self, event: TelemetryEvent) -> None:
        """Emit a single structured telemetry event.

        Args:
            event: Event to emit.
        """

    def increment(
        self,
        name: str,
        value: int = 1,
        *,
        attrs: dict[str, object] | None = None,
        ctx: TelemetryContext | None = None,
    ) -> None:
        """Increment a counter metric.

        Args:
            name: Metric name.
            value: Amount to increment by.
            attrs: Optional metric attributes.
            ctx: Optional correlation context.
        """

    def observe(
        self,
        name: str,
        value: float,
        *,
        attrs: dict[str, object] | None = None,
        ctx: TelemetryContext | None = None,
    ) -> None:
        """Record a histogram/summary observation.

        Args:
            name: Metric name.
            value: Observed value.
            attrs: Optional metric attributes.
            ctx: Optional correlation context.
        """

    def span(
        self,
        name: str,
        *,
        ctx: TelemetryContext | None = None,
        attrs: dict[str, object] | None = None,
    ) -> SpanContextManager:
        """Return a context manager for a telemetry span.

        Args:
            name: Span name.
            ctx: Optional existing context to base the span on.
            attrs: Optional span attributes.

        Returns:
            SpanContextManager: Sync/async context manager.
        """
        raise NotImplementedError


@runtime_checkable
class CostEstimatorPort(Protocol):
    """Port for estimating AI call costs.

    Implementations convert token counts into approximate cost
    information. Failure to estimate cost must not break core
    functionality; callers should treat missing values as best-effort.
    """

    def estimate_cost(
        self,
        *,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
    ) -> CostRecord:
        """Estimate the cost of an AI call.

        Args:
            provider: Logical provider name.
            model: Model identifier.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            CostRecord: Estimated cost information.
        """
        raise NotImplementedError


__all__ = ["TelemetryPort", "CostEstimatorPort", "SpanContextManager"]
