"""Enterprise workload routing engine for model and provider selection.

Purpose:
  - Route LLM and AI workloads to the right model/provider based on
    policy, cost, latency, capability requirements, and fallback rules.
  - Replace bespoke if/else routing logic scattered across services.

Guarantees:
  - Routing decisions are deterministic from stable inputs.
  - All policy enforcement is transparent and explainable.
  - The engine is provider-neutral — no vendor lock-in.

Usage:
  Basic example::

    from electripy.ai.workload_router import (
        CandidateModel,
        InMemoryCatalog,
        RoutingPolicy,
        RoutingRequest,
        RoutingService,
        WorkloadType,
    )

    catalog = InMemoryCatalog(candidates=[model_a, model_b])
    service = RoutingService(catalog=catalog)
    decision = service.route(
        RoutingRequest(workload_type=WorkloadType.CHAT),
    )
    print(decision.selected.model_id)
"""

from __future__ import annotations

from .adapters import (
    InMemoryCatalog,
    LoggingTelemetryAdapter,
    NoOpTelemetryAdapter,
    StaticHealthAdapter,
)
from .domain import (
    CandidateModel,
    CandidateScore,
    CapabilityProfile,
    ContextWindowProfile,
    CostProfile,
    DisqualifiedCandidate,
    FallbackPlan,
    LatencyProfile,
    ReliabilityProfile,
    RouteExplanation,
    RoutingConstraint,
    RoutingDecision,
    RoutingPolicy,
    RoutingRequest,
    ScoringWeights,
    WorkloadType,
)
from .errors import (
    BudgetExceededError,
    ConstraintViolationError,
    NoCandidateError,
    WorkloadRouterError,
)
from .ports import (
    HealthStatusPort,
    ModelCatalogPort,
    TelemetryHookPort,
)
from .services import (
    RoutingService,
)

__all__ = [
    # Domain
    "CandidateModel",
    "CandidateScore",
    "CapabilityProfile",
    "ContextWindowProfile",
    "CostProfile",
    "DisqualifiedCandidate",
    "FallbackPlan",
    "LatencyProfile",
    "ReliabilityProfile",
    "RouteExplanation",
    "RoutingConstraint",
    "RoutingDecision",
    "RoutingPolicy",
    "RoutingRequest",
    "ScoringWeights",
    "WorkloadType",
    # Errors
    "BudgetExceededError",
    "ConstraintViolationError",
    "NoCandidateError",
    "WorkloadRouterError",
    # Ports
    "HealthStatusPort",
    "ModelCatalogPort",
    "TelemetryHookPort",
    # Adapters
    "InMemoryCatalog",
    "LoggingTelemetryAdapter",
    "NoOpTelemetryAdapter",
    "StaticHealthAdapter",
    # Services
    "RoutingService",
]
