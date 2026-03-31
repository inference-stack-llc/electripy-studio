"""Adapters for the workload routing engine.

Provides concrete implementations of the ports defined in
:mod:`electripy.ai.workload_router.ports`:

- **InMemoryCatalog** — static list of candidate models.
- **StaticHealthAdapter** — marks all candidates as healthy.
- **NoOpTelemetryAdapter** — silently discards telemetry events.
- **LoggingTelemetryAdapter** — logs routing decisions.

Example:
    from electripy.ai.workload_router.adapters import InMemoryCatalog

    catalog = InMemoryCatalog(candidates=[model_a, model_b])
"""

from __future__ import annotations

from dataclasses import dataclass, field

from electripy.core.logging import get_logger

from .domain import CandidateModel, RoutingDecision

__all__ = [
    "InMemoryCatalog",
    "LoggingTelemetryAdapter",
    "NoOpTelemetryAdapter",
    "StaticHealthAdapter",
]

logger = get_logger(__name__)


@dataclass(slots=True)
class InMemoryCatalog:
    """Static in-memory model catalog.

    Attributes:
        candidates: The list of candidate models.
    """

    candidates: list[CandidateModel] = field(default_factory=list)

    def list_candidates(self) -> list[CandidateModel]:
        """Return the stored candidate list."""
        return list(self.candidates)


@dataclass(frozen=True, slots=True)
class StaticHealthAdapter:
    """Health adapter that reports all models as healthy."""

    def is_healthy(self, model_id: str, provider: str) -> bool:
        """Always returns ``True``."""
        return True


@dataclass(frozen=True, slots=True)
class NoOpTelemetryAdapter:
    """Telemetry adapter that silently discards events."""

    def on_routing_decision(self, decision: RoutingDecision) -> None:
        """Do nothing."""


@dataclass(frozen=True, slots=True)
class LoggingTelemetryAdapter:
    """Telemetry adapter that logs routing decisions."""

    def on_routing_decision(self, decision: RoutingDecision) -> None:
        """Log the routing decision at INFO level."""
        logger.info(
            "Routing decision",
            extra={
                "selected_model": decision.selected.model_id,
                "selected_provider": decision.selected.provider,
                "fallback_depth": decision.fallback_plan.depth,
                "disqualified_count": len(decision.explanation.disqualified),
                "workload_type": decision.request.workload_type,
            },
        )
