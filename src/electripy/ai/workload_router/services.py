"""Routing service — the core decision engine.

This module provides :class:`RoutingService`, the main orchestration
layer that evaluates candidates against a routing policy, scores them,
builds a fallback plan, and produces a fully explainable decision.

Example:
    from electripy.ai.workload_router import (
        RoutingService,
        InMemoryCatalog,
        RoutingRequest,
        RoutingPolicy,
        WorkloadType,
    )

    catalog = InMemoryCatalog(candidates=[model_a, model_b])
    service = RoutingService(catalog=catalog)
    decision = service.route(RoutingRequest(workload_type=WorkloadType.CHAT))
"""

from __future__ import annotations

from dataclasses import dataclass, field

from electripy.core.logging import get_logger

from .adapters import NoOpTelemetryAdapter, StaticHealthAdapter
from .domain import (
    CandidateModel,
    CandidateScore,
    DisqualifiedCandidate,
    FallbackPlan,
    RouteExplanation,
    RoutingDecision,
    RoutingPolicy,
    RoutingRequest,
    ScoringWeights,
    WorkloadType,
)
from .errors import BudgetExceededError, NoCandidateError
from .ports import HealthStatusPort, ModelCatalogPort, TelemetryHookPort

__all__ = [
    "RoutingService",
]

logger = get_logger(__name__)


# ── Scoring helpers ──────────────────────────────────────────────────

# Normalisation ranges — used to map raw values to a 0–1 score.
_MAX_COST = 0.1  # $0.10 / 1 k tokens as upper bound
_MAX_LATENCY_MS = 10_000.0  # 10 s as upper bound
_MAX_CONTEXT = 1_000_000  # 1 M tokens as upper bound


def _normalise(value: float, maximum: float) -> float:
    """Map *value* into [0, 1] where 0 is worst (at *maximum*) and 1 is best (at 0)."""
    if maximum <= 0:
        return 1.0
    return max(0.0, 1.0 - value / maximum)


def _score_candidate(
    candidate: CandidateModel,
    weights: ScoringWeights,
    preferred_providers: frozenset[str],
) -> CandidateScore:
    """Compute a multi-dimensional score for a single candidate."""
    cost_score = _normalise(candidate.cost.blended_cost_per_1k, _MAX_COST) * weights.cost
    latency_score = _normalise(candidate.latency.median_ms, _MAX_LATENCY_MS) * weights.latency
    reliability_score = candidate.reliability.availability * weights.reliability
    context_score = (
        min(candidate.context_window.max_input_tokens / _MAX_CONTEXT, 1.0) * weights.context_window
    )

    bonus = 0.05 if candidate.provider in preferred_providers else 0.0

    total = cost_score + latency_score + reliability_score + context_score + bonus

    return CandidateScore(
        candidate=candidate,
        total_score=total,
        cost_score=cost_score,
        latency_score=latency_score,
        reliability_score=reliability_score,
        context_score=context_score,
    )


# ── Built-in constraint builders ────────────────────────────────────

_CAPABILITY_FIELDS = {
    "structured_output",
    "tool_use",
    "streaming",
    "vision",
    "reasoning",
    "code_execution",
}


def _build_policy_summary(policy: RoutingPolicy) -> str:
    """Generate a human-readable summary of the active policy."""
    parts: list[str] = []
    if policy.allowed_providers:
        parts.append(f"allowed_providers={sorted(policy.allowed_providers)}")
    if policy.denied_providers:
        parts.append(f"denied_providers={sorted(policy.denied_providers)}")
    if policy.denied_model_families:
        parts.append(f"denied_families={sorted(policy.denied_model_families)}")
    if policy.max_cost_per_1k is not None:
        parts.append(f"max_cost=${policy.max_cost_per_1k}/1k")
    if policy.max_latency_ms is not None:
        parts.append(f"max_latency={policy.max_latency_ms}ms")
    if policy.min_context_tokens is not None:
        parts.append(f"min_context={policy.min_context_tokens}")
    if policy.required_capabilities:
        parts.append(f"required_caps={sorted(policy.required_capabilities)}")
    if policy.constraints:
        parts.append(f"custom_constraints={[c.name for c in policy.constraints]}")
    parts.append(
        f"weights(cost={policy.weights.cost}, lat={policy.weights.latency}, "
        f"rel={policy.weights.reliability}, ctx={policy.weights.context_window})"
    )
    return "; ".join(parts) if parts else "default policy"


# ── RoutingService ───────────────────────────────────────────────────


@dataclass(slots=True)
class RoutingService:
    """Enterprise workload routing engine.

    Evaluates candidate models from a catalog against a routing policy,
    applies hard constraints, scores remaining candidates, selects the
    best one, and produces a fallback plan with full explanations.

    Args:
        catalog: Source of candidate models.
        health: Optional live health provider.
        telemetry: Optional telemetry hook.
    """

    catalog: ModelCatalogPort
    health: HealthStatusPort = field(default_factory=StaticHealthAdapter)
    telemetry: TelemetryHookPort = field(default_factory=NoOpTelemetryAdapter)

    # ── Primary API ──────────────────────────────────────────────────

    def route(self, request: RoutingRequest) -> RoutingDecision:
        """Choose the best model for a routing request.

        Applies the full constraint → score → rank → fallback pipeline
        and returns a decision with explanation.

        Args:
            request: The routing request containing workload type and
                policy.

        Returns:
            A complete routing decision.

        Raises:
            NoCandidateError: If no candidate satisfies the policy.
            BudgetExceededError: If all candidates exceed the budget.
        """
        candidates = self.catalog.list_candidates()
        if not candidates:
            raise NoCandidateError("Model catalog is empty")

        policy = request.policy
        eligible, disqualified = self._apply_constraints(candidates, policy, request)

        if not eligible:
            self._raise_no_candidate(disqualified)

        scored = self._score_and_rank(eligible, policy)
        selected = scored[0].candidate
        fallbacks = tuple(s.candidate for s in scored[1:])

        if policy.min_fallbacks > 0 and len(fallbacks) < policy.min_fallbacks:
            raise NoCandidateError(
                f"Policy requires at least {policy.min_fallbacks} fallback(s) "
                f"but only {len(fallbacks)} candidate(s) survived filtering"
            )

        explanation = RouteExplanation(
            selected_model_id=selected.model_id,
            scores=tuple(scored),
            disqualified=tuple(disqualified),
            policy_summary=_build_policy_summary(policy),
        )

        decision = RoutingDecision(
            selected=selected,
            fallback_plan=FallbackPlan(candidates=fallbacks),
            explanation=explanation,
            request=request,
        )

        self.telemetry.on_routing_decision(decision)
        logger.debug(
            "Routing decision",
            extra={
                "selected": selected.model_id,
                "fallback_depth": len(fallbacks),
                "disqualified": len(disqualified),
            },
        )
        return decision

    def route_by_workload(
        self,
        workload_type: WorkloadType,
        *,
        policy: RoutingPolicy | None = None,
    ) -> RoutingDecision:
        """Route for a specific workload type.

        Convenience wrapper that builds a :class:`RoutingRequest` from
        the workload type and optional policy.

        Args:
            workload_type: The type of workload to route.
            policy: Optional routing policy; defaults to :class:`RoutingPolicy`.

        Returns:
            A routing decision.
        """
        return self.route(
            RoutingRequest(workload_type=workload_type, policy=policy or RoutingPolicy())
        )

    def route_by_budget(
        self,
        max_cost_per_1k: float,
        *,
        workload_type: WorkloadType = WorkloadType.CHAT,
    ) -> RoutingDecision:
        """Route with a budget ceiling.

        Args:
            max_cost_per_1k: Maximum blended cost per 1 000 tokens.
            workload_type: Workload type; defaults to ``CHAT``.

        Returns:
            A routing decision.

        Raises:
            BudgetExceededError: If all candidates exceed the budget.
        """
        return self.route(
            RoutingRequest(
                workload_type=workload_type,
                policy=RoutingPolicy(max_cost_per_1k=max_cost_per_1k),
            )
        )

    def route_by_latency(
        self,
        max_latency_ms: float,
        *,
        workload_type: WorkloadType = WorkloadType.CHAT,
    ) -> RoutingDecision:
        """Route with a latency SLO.

        Args:
            max_latency_ms: Maximum acceptable median latency in ms.
            workload_type: Workload type; defaults to ``CHAT``.

        Returns:
            A routing decision.
        """
        return self.route(
            RoutingRequest(
                workload_type=workload_type,
                policy=RoutingPolicy(max_latency_ms=max_latency_ms),
            )
        )

    def explain(self, request: RoutingRequest) -> RouteExplanation:
        """Return only the explanation for a routing request.

        This is useful for debugging and policy tuning without using
        the selected model.

        Args:
            request: The routing request.

        Returns:
            Detailed route explanation.
        """
        decision = self.route(request)
        return decision.explanation

    # ── Constraint pipeline ──────────────────────────────────────────

    def _apply_constraints(
        self,
        candidates: list[CandidateModel],
        policy: RoutingPolicy,
        request: RoutingRequest,
    ) -> tuple[list[CandidateModel], list[DisqualifiedCandidate]]:
        """Filter candidates through all hard constraints.

        Returns (eligible, disqualified).
        """
        eligible: list[CandidateModel] = []
        disqualified: list[DisqualifiedCandidate] = []

        for c in candidates:
            reason = self._check_candidate(c, policy, request)
            if reason is None:
                eligible.append(c)
            else:
                disqualified.append(DisqualifiedCandidate(candidate=c, reason=reason))

        return eligible, disqualified

    def _check_candidate(
        self,
        candidate: CandidateModel,
        policy: RoutingPolicy,
        request: RoutingRequest,
    ) -> str | None:
        """Return a disqualification reason, or None if eligible."""
        # Health check
        if not self.health.is_healthy(candidate.model_id, candidate.provider):
            return "unhealthy"

        # Reliability profile
        if not candidate.reliability.healthy:
            return "marked unhealthy in reliability profile"

        # Provider allow/deny lists
        if policy.allowed_providers and candidate.provider not in policy.allowed_providers:
            return f"provider {candidate.provider!r} not in allowed list"

        if candidate.provider in policy.denied_providers:
            return f"provider {candidate.provider!r} is denied"

        # Model family deny
        for family in policy.denied_model_families:
            if candidate.model_id.startswith(family):
                return f"model family {family!r} is denied"

        # Budget ceiling
        if policy.max_cost_per_1k is not None:
            if candidate.cost.blended_cost_per_1k > policy.max_cost_per_1k:
                return (
                    f"cost ${candidate.cost.blended_cost_per_1k:.4f}/1k "
                    f"exceeds budget ${policy.max_cost_per_1k:.4f}/1k"
                )

        # Latency SLO
        if policy.max_latency_ms is not None:
            if candidate.latency.median_ms > policy.max_latency_ms:
                return (
                    f"median latency {candidate.latency.median_ms}ms "
                    f"exceeds SLO {policy.max_latency_ms}ms"
                )

        # Minimum context window
        if policy.min_context_tokens is not None:
            if candidate.context_window.max_input_tokens < policy.min_context_tokens:
                return (
                    f"context window {candidate.context_window.max_input_tokens} "
                    f"< required {policy.min_context_tokens}"
                )

        # Required capabilities
        for cap_name in policy.required_capabilities:
            if cap_name in _CAPABILITY_FIELDS:
                if not getattr(candidate.capabilities, cap_name, False):
                    return f"missing required capability: {cap_name}"

        # Custom constraints
        for constraint in policy.constraints:
            if not constraint.predicate(candidate):
                return f"failed constraint: {constraint.name}"

        return None

    # ── Scoring ──────────────────────────────────────────────────────

    def _score_and_rank(
        self,
        candidates: list[CandidateModel],
        policy: RoutingPolicy,
    ) -> list[CandidateScore]:
        """Score candidates and return them sorted best → worst."""
        scored = [
            _score_candidate(c, policy.weights, policy.preferred_providers)
            for c in candidates
        ]
        scored.sort(key=lambda s: s.total_score, reverse=True)
        return scored

    # ── Error helpers ────────────────────────────────────────────────

    @staticmethod
    def _raise_no_candidate(disqualified: list[DisqualifiedCandidate]) -> None:
        """Raise an appropriate error based on disqualification reasons."""
        budget_reasons = [d for d in disqualified if "exceeds budget" in d.reason]
        if budget_reasons and len(budget_reasons) == len(disqualified):
            raise BudgetExceededError(
                f"All {len(disqualified)} candidates exceed the budget ceiling"
            )
        reasons = "; ".join(f"{d.candidate.model_id}: {d.reason}" for d in disqualified[:5])
        raise NoCandidateError(f"No candidate satisfies the routing policy. Reasons: {reasons}")
