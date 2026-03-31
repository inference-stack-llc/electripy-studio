"""Ports (Protocol interfaces) for the workload routing engine.

These runtime-checkable protocols define the pluggable boundaries:

- **ModelCatalogPort** — supplies the list of candidate models.
- **HealthStatusPort** — reports live health for each model.
- **TelemetryHookPort** — receives routing telemetry events.

All concrete implementations live in :mod:`electripy.ai.workload_router.adapters`.

Example:
    from electripy.ai.workload_router.ports import ModelCatalogPort

    class MyDynamicCatalog(ModelCatalogPort):
        def list_candidates(self) -> list[CandidateModel]:
            return fetch_from_registry()
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .domain import CandidateModel, RoutingDecision

__all__ = [
    "HealthStatusPort",
    "ModelCatalogPort",
    "TelemetryHookPort",
]


@runtime_checkable
class ModelCatalogPort(Protocol):
    """Supplier of candidate models for routing.

    Implementations may read from a static list, a configuration file,
    a database, or a live vendor registry.
    """

    def list_candidates(self) -> list[CandidateModel]:
        """Return all available candidate models.

        Returns:
            List of candidate model profiles.
        """
        ...


@runtime_checkable
class HealthStatusPort(Protocol):
    """Provider of live health status for candidate models.

    Implementations may query a monitoring system, a heartbeat
    endpoint, or simply return a static healthy state.
    """

    def is_healthy(self, model_id: str, provider: str) -> bool:
        """Return whether the given model is currently healthy.

        Args:
            model_id: Model identifier.
            provider: Provider name.

        Returns:
            ``True`` if the model is healthy; ``False`` otherwise.
        """
        ...


@runtime_checkable
class TelemetryHookPort(Protocol):
    """Optional telemetry hook for routing events.

    Called after each routing decision to allow logging, metrics,
    or tracing integration.
    """

    def on_routing_decision(self, decision: RoutingDecision) -> None:
        """Record a routing decision.

        Args:
            decision: The complete routing result.
        """
        ...
