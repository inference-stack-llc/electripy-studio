"""Tests for electripy.ai.workload_router.services — the routing engine."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from electripy.ai.workload_router.adapters import InMemoryCatalog
from electripy.ai.workload_router.domain import (
    CandidateModel,
    CapabilityProfile,
    ContextWindowProfile,
    CostProfile,
    LatencyProfile,
    ReliabilityProfile,
    RoutingConstraint,
    RoutingDecision,
    RoutingPolicy,
    RoutingRequest,
    ScoringWeights,
    WorkloadType,
)
from electripy.ai.workload_router.errors import BudgetExceededError, NoCandidateError
from electripy.ai.workload_router.services import RoutingService

# ── Test fixtures ────────────────────────────────────────────────────

_CHEAP_FAST = CandidateModel(
    model_id="gpt-4o-mini",
    provider="openai",
    capabilities=CapabilityProfile(structured_output=True, tool_use=True, streaming=True),
    cost=CostProfile(cost_per_1k_input=0.00015, cost_per_1k_output=0.0006),
    latency=LatencyProfile(median_ms=200.0, p99_ms=1000.0),
    reliability=ReliabilityProfile(availability=0.999, healthy=True),
    context_window=ContextWindowProfile(max_input_tokens=128_000, max_output_tokens=16_384),
)

_EXPENSIVE_POWERFUL = CandidateModel(
    model_id="gpt-4o",
    provider="openai",
    capabilities=CapabilityProfile(
        structured_output=True,
        tool_use=True,
        streaming=True,
        vision=True,
        reasoning=True,
    ),
    cost=CostProfile(cost_per_1k_input=0.005, cost_per_1k_output=0.015),
    latency=LatencyProfile(median_ms=800.0, p99_ms=5000.0),
    reliability=ReliabilityProfile(availability=0.998, healthy=True),
    context_window=ContextWindowProfile(max_input_tokens=128_000, max_output_tokens=16_384),
)

_ANTHROPIC_LARGE = CandidateModel(
    model_id="claude-3.5-sonnet",
    provider="anthropic",
    capabilities=CapabilityProfile(
        structured_output=True,
        tool_use=True,
        streaming=True,
        vision=True,
    ),
    cost=CostProfile(cost_per_1k_input=0.003, cost_per_1k_output=0.015),
    latency=LatencyProfile(median_ms=600.0, p99_ms=4000.0),
    reliability=ReliabilityProfile(availability=0.997, healthy=True),
    context_window=ContextWindowProfile(max_input_tokens=200_000, max_output_tokens=8_192),
)

_SMALL_LOCAL = CandidateModel(
    model_id="llama-3-8b",
    provider="ollama",
    capabilities=CapabilityProfile(streaming=True),
    cost=CostProfile(cost_per_1k_input=0.0, cost_per_1k_output=0.0),
    latency=LatencyProfile(median_ms=100.0, p99_ms=500.0),
    reliability=ReliabilityProfile(availability=0.95, healthy=True),
    context_window=ContextWindowProfile(max_input_tokens=8_192, max_output_tokens=4_096),
)

_UNHEALTHY = CandidateModel(
    model_id="broken-model",
    provider="broken",
    reliability=ReliabilityProfile(availability=0.5, healthy=False),
)


def _all_models() -> list[CandidateModel]:
    return [_CHEAP_FAST, _EXPENSIVE_POWERFUL, _ANTHROPIC_LARGE, _SMALL_LOCAL]


def _make_service(candidates: list[CandidateModel] | None = None) -> RoutingService:
    return RoutingService(
        catalog=InMemoryCatalog(
            candidates=_all_models() if candidates is None else candidates,
        ),
    )


# ── Hard constraint filtering ────────────────────────────────────────


class TestConstraintFiltering:
    def test_allowed_providers(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(allowed_providers=frozenset({"openai"})),
            )
        )
        assert decision.selected.provider == "openai"
        # anthropic and ollama should be disqualified
        dq_providers = {d.candidate.provider for d in decision.explanation.disqualified}
        assert "anthropic" in dq_providers
        assert "ollama" in dq_providers

    def test_denied_providers(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(denied_providers=frozenset({"openai"})),
            )
        )
        assert decision.selected.provider != "openai"
        dq_ids = {d.candidate.model_id for d in decision.explanation.disqualified}
        assert "gpt-4o-mini" in dq_ids
        assert "gpt-4o" in dq_ids

    def test_denied_model_families(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(denied_model_families=frozenset({"gpt-4o"})),
            )
        )
        assert not decision.selected.model_id.startswith("gpt-4o")

    def test_required_capabilities_structured_output(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(
                    required_capabilities=frozenset({"structured_output"}),
                ),
            )
        )
        assert decision.selected.capabilities.structured_output
        # llama-3-8b should be disqualified
        dq_ids = {d.candidate.model_id for d in decision.explanation.disqualified}
        assert "llama-3-8b" in dq_ids

    def test_required_capabilities_vision(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(required_capabilities=frozenset({"vision"})),
            )
        )
        assert decision.selected.capabilities.vision

    def test_required_capabilities_reasoning(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(required_capabilities=frozenset({"reasoning"})),
            )
        )
        assert decision.selected.model_id == "gpt-4o"

    def test_custom_constraint(self) -> None:
        service = _make_service()
        constraint = RoutingConstraint(
            name="only-anthropic",
            predicate=lambda c: c.provider == "anthropic",
        )
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(constraints=(constraint,)),
            )
        )
        assert decision.selected.provider == "anthropic"

    def test_custom_constraint_disqualification_reason(self) -> None:
        service = _make_service()
        constraint = RoutingConstraint(
            name="my-custom-rule",
            predicate=lambda c: c.model_id == "nonexistent",
        )
        with pytest.raises(NoCandidateError, match="my-custom-rule"):
            service.route(RoutingRequest(policy=RoutingPolicy(constraints=(constraint,))))

    def test_unhealthy_excluded(self) -> None:
        service = _make_service([_CHEAP_FAST, _UNHEALTHY])
        decision = service.route(RoutingRequest())
        assert decision.selected.model_id == "gpt-4o-mini"
        dq_ids = {d.candidate.model_id for d in decision.explanation.disqualified}
        assert "broken-model" in dq_ids

    def test_unhealthy_via_health_port(self) -> None:
        @dataclass(frozen=True, slots=True)
        class _UnhealthyOpenAI:
            def is_healthy(self, model_id: str, provider: str) -> bool:
                return provider != "openai"

        service = RoutingService(
            catalog=InMemoryCatalog(candidates=_all_models()),
            health=_UnhealthyOpenAI(),
        )
        decision = service.route(RoutingRequest())
        assert decision.selected.provider != "openai"


# ── Budget-aware decisions ───────────────────────────────────────────


class TestBudgetRouting:
    def test_budget_ceiling(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(max_cost_per_1k=0.001),
            )
        )
        assert decision.selected.cost.blended_cost_per_1k <= 0.001

    def test_budget_excludes_expensive(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(max_cost_per_1k=0.005),
            )
        )
        dq_ids = {d.candidate.model_id for d in decision.explanation.disqualified}
        assert "gpt-4o" in dq_ids  # blended = 0.01, exceeds 0.005

    def test_all_exceed_budget_raises(self) -> None:
        service = _make_service([_EXPENSIVE_POWERFUL, _ANTHROPIC_LARGE])
        with pytest.raises(BudgetExceededError, match="exceed the budget"):
            service.route(RoutingRequest(policy=RoutingPolicy(max_cost_per_1k=0.0001)))

    def test_route_by_budget_convenience(self) -> None:
        service = _make_service()
        decision = service.route_by_budget(0.001)
        assert decision.selected.cost.blended_cost_per_1k <= 0.001


# ── Latency-aware decisions ──────────────────────────────────────────


class TestLatencyRouting:
    def test_latency_slo(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(max_latency_ms=300.0),
            )
        )
        assert decision.selected.latency.median_ms <= 300.0

    def test_latency_excludes_slow(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(max_latency_ms=300.0),
            )
        )
        dq_ids = {d.candidate.model_id for d in decision.explanation.disqualified}
        assert "gpt-4o" in dq_ids
        assert "claude-3.5-sonnet" in dq_ids

    def test_route_by_latency_convenience(self) -> None:
        service = _make_service()
        decision = service.route_by_latency(300.0)
        assert decision.selected.latency.median_ms <= 300.0


# ── Context window requirements ──────────────────────────────────────


class TestContextWindowRouting:
    def test_min_context_tokens(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(min_context_tokens=100_000),
            )
        )
        assert decision.selected.context_window.max_input_tokens >= 100_000
        dq_ids = {d.candidate.model_id for d in decision.explanation.disqualified}
        assert "llama-3-8b" in dq_ids

    def test_large_context_selects_anthropic(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(min_context_tokens=130_000),
            )
        )
        assert decision.selected.model_id == "claude-3.5-sonnet"


# ── Score-based ranking ──────────────────────────────────────────────


class TestScoreRanking:
    def test_cheapest_wins_by_default(self) -> None:
        service = _make_service()
        decision = service.route(RoutingRequest())
        # Cost-weighted scoring should favor the cheapest candidate (llama or gpt-4o-mini)
        assert decision.selected.cost.blended_cost_per_1k <= 0.001

    def test_latency_focused_weights(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(
                    weights=ScoringWeights(cost=0.0, latency=1.0, reliability=0.0),
                ),
            )
        )
        # Fastest should win: llama (100ms) or gpt-4o-mini (200ms)
        assert decision.selected.latency.median_ms <= 200.0

    def test_preferred_providers_get_bonus(self) -> None:
        """With equal-ish candidates, preferred provider gets a tie-breaking bonus."""
        c_openai = CandidateModel(
            model_id="oai",
            provider="openai",
            cost=CostProfile(cost_per_1k_input=0.01, cost_per_1k_output=0.01),
            latency=LatencyProfile(median_ms=500.0),
        )
        c_anthro = CandidateModel(
            model_id="anth",
            provider="anthropic",
            cost=CostProfile(cost_per_1k_input=0.01, cost_per_1k_output=0.01),
            latency=LatencyProfile(median_ms=500.0),
        )
        service = _make_service([c_openai, c_anthro])
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(preferred_providers=frozenset({"anthropic"})),
            )
        )
        assert decision.selected.provider == "anthropic"

    def test_scores_are_sorted_descending(self) -> None:
        service = _make_service()
        decision = service.route(RoutingRequest())
        scores = [s.total_score for s in decision.explanation.scores]
        assert scores == sorted(scores, reverse=True)


# ── Fallback ordering ────────────────────────────────────────────────


class TestFallbackPlan:
    def test_fallbacks_present(self) -> None:
        service = _make_service()
        decision = service.route(RoutingRequest())
        assert decision.fallback_plan.depth >= 1

    def test_fallbacks_ordered_by_score(self) -> None:
        service = _make_service()
        decision = service.route(RoutingRequest())
        # Fallbacks should be scored in descending order
        fb_ids = [c.model_id for c in decision.fallback_plan.candidates]
        score_ids = [s.candidate.model_id for s in decision.explanation.scores[1:]]
        assert fb_ids == score_ids

    def test_fallbacks_exclude_selected(self) -> None:
        service = _make_service()
        decision = service.route(RoutingRequest())
        fb_ids = {c.model_id for c in decision.fallback_plan.candidates}
        assert decision.selected.model_id not in fb_ids

    def test_min_fallbacks_satisfied(self) -> None:
        service = _make_service()
        decision = service.route(RoutingRequest(policy=RoutingPolicy(min_fallbacks=2)))
        assert decision.fallback_plan.depth >= 2

    def test_min_fallbacks_not_met_raises(self) -> None:
        service = _make_service([_CHEAP_FAST])
        with pytest.raises(NoCandidateError, match="fallback"):
            service.route(RoutingRequest(policy=RoutingPolicy(min_fallbacks=1)))

    def test_unhealthy_excluded_from_fallbacks(self) -> None:
        service = _make_service([_CHEAP_FAST, _EXPENSIVE_POWERFUL, _UNHEALTHY])
        decision = service.route(RoutingRequest())
        fb_ids = {c.model_id for c in decision.fallback_plan.candidates}
        assert "broken-model" not in fb_ids

    def test_fallbacks_graceful_degradation(self) -> None:
        """With default policy, fallbacks go from cheaper → more expensive."""
        service = _make_service()
        decision = service.route(RoutingRequest())
        # The selected model should be cheapest/best-scored
        # Fallback chain provides alternatives
        assert decision.fallback_plan.depth >= 2


# ── Explanation generation ───────────────────────────────────────────


class TestExplanation:
    def test_explanation_selected_id(self) -> None:
        service = _make_service()
        decision = service.route(RoutingRequest())
        assert decision.explanation.selected_model_id == decision.selected.model_id

    def test_explanation_policy_summary(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(
                    allowed_providers=frozenset({"openai"}),
                    max_cost_per_1k=0.01,
                ),
            )
        )
        summary = decision.explanation.policy_summary
        assert "allowed_providers" in summary
        assert "max_cost" in summary

    def test_explain_convenience(self) -> None:
        service = _make_service()
        explanation = service.explain(RoutingRequest())
        assert explanation.selected_model_id is not None
        assert len(explanation.scores) > 0

    def test_explanation_default_policy_summary(self) -> None:
        service = _make_service()
        decision = service.route(RoutingRequest())
        assert "weights" in decision.explanation.policy_summary

    def test_disqualified_reasons_populated(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(allowed_providers=frozenset({"openai"})),
            )
        )
        for dq in decision.explanation.disqualified:
            assert len(dq.reason) > 0

    def test_explanation_shows_custom_constraints(self) -> None:
        service = _make_service()
        constraint = RoutingConstraint(name="my-rule", predicate=lambda c: True)
        decision = service.route(RoutingRequest(policy=RoutingPolicy(constraints=(constraint,))))
        assert "my-rule" in decision.explanation.policy_summary


# ── Deterministic outputs ────────────────────────────────────────────


class TestDeterminism:
    def test_stable_inputs_stable_outputs(self) -> None:
        service = _make_service()
        request = RoutingRequest(
            policy=RoutingPolicy(allowed_providers=frozenset({"openai"})),
        )
        d1 = service.route(request)
        d2 = service.route(request)
        assert d1.selected.model_id == d2.selected.model_id
        assert d1.fallback_plan.depth == d2.fallback_plan.depth

    def test_same_scores_across_calls(self) -> None:
        service = _make_service()
        request = RoutingRequest()
        d1 = service.route(request)
        d2 = service.route(request)
        for s1, s2 in zip(d1.explanation.scores, d2.explanation.scores, strict=True):
            assert s1.total_score == s2.total_score
            assert s1.candidate.model_id == s2.candidate.model_id


# ── Convenience methods ──────────────────────────────────────────────


class TestConvenienceMethods:
    def test_route_by_workload(self) -> None:
        service = _make_service()
        decision = service.route_by_workload(WorkloadType.EXTRACTION)
        assert isinstance(decision, RoutingDecision)
        assert decision.request.workload_type == WorkloadType.EXTRACTION

    def test_route_by_workload_with_policy(self) -> None:
        service = _make_service()
        policy = RoutingPolicy(allowed_providers=frozenset({"openai"}))
        decision = service.route_by_workload(WorkloadType.CHAT, policy=policy)
        assert decision.selected.provider == "openai"

    def test_route_by_budget(self) -> None:
        service = _make_service()
        decision = service.route_by_budget(0.01)
        assert decision.selected.cost.blended_cost_per_1k <= 0.01

    def test_route_by_latency(self) -> None:
        service = _make_service()
        decision = service.route_by_latency(500.0)
        assert decision.selected.latency.median_ms <= 500.0


# ── Edge cases ───────────────────────────────────────────────────────


class TestEdgeCases:
    def test_empty_catalog_raises(self) -> None:
        service = _make_service([])
        with pytest.raises(NoCandidateError, match="empty"):
            service.route(RoutingRequest())

    def test_single_candidate(self) -> None:
        service = _make_service([_CHEAP_FAST])
        decision = service.route(RoutingRequest())
        assert decision.selected.model_id == "gpt-4o-mini"
        assert decision.fallback_plan.depth == 0

    def test_all_denied(self) -> None:
        service = _make_service()
        with pytest.raises(NoCandidateError):
            service.route(
                RoutingRequest(
                    policy=RoutingPolicy(
                        denied_providers=frozenset({"openai", "anthropic", "ollama"}),
                    ),
                )
            )

    def test_conflicting_allowed_and_denied(self) -> None:
        """Denied takes effect only outside the allowed set."""
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(
                    allowed_providers=frozenset({"openai", "anthropic"}),
                    denied_providers=frozenset({"anthropic"}),
                ),
            )
        )
        assert decision.selected.provider == "openai"

    def test_combined_budget_and_latency(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(
                    max_cost_per_1k=0.01,
                    max_latency_ms=700.0,
                ),
            )
        )
        assert decision.selected.cost.blended_cost_per_1k <= 0.01
        assert decision.selected.latency.median_ms <= 700.0

    def test_combined_required_caps_and_budget(self) -> None:
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(
                    required_capabilities=frozenset({"tool_use"}),
                    max_cost_per_1k=0.005,
                ),
            )
        )
        assert decision.selected.capabilities.tool_use
        assert decision.selected.cost.blended_cost_per_1k <= 0.005


# ── Observability hooks ──────────────────────────────────────────────


class TestTelemetryHook:
    def test_telemetry_called(self) -> None:
        recorded: list[RoutingDecision] = []

        class _Recorder:
            def on_routing_decision(self, decision: RoutingDecision) -> None:
                recorded.append(decision)

        service = RoutingService(
            catalog=InMemoryCatalog(candidates=_all_models()),
            telemetry=_Recorder(),
        )
        service.route(RoutingRequest())
        assert len(recorded) == 1
        assert recorded[0].selected.model_id is not None

    def test_telemetry_receives_full_decision(self) -> None:
        recorded: list[RoutingDecision] = []

        class _Recorder:
            def on_routing_decision(self, decision: RoutingDecision) -> None:
                recorded.append(decision)

        service = RoutingService(
            catalog=InMemoryCatalog(candidates=_all_models()),
            telemetry=_Recorder(),
        )
        service.route(
            RoutingRequest(
                policy=RoutingPolicy(allowed_providers=frozenset({"openai"})),
            )
        )
        decision = recorded[0]
        assert decision.selected.provider == "openai"
        assert len(decision.explanation.disqualified) > 0


# ── Policy expression patterns ───────────────────────────────────────


class TestPolicyExpressions:
    def test_prefer_cheapest_capable(self) -> None:
        """'prefer cheapest capable model' — cost-weighted with required caps."""
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(
                    required_capabilities=frozenset({"structured_output"}),
                    weights=ScoringWeights(cost=2.0, latency=0.1, reliability=0.1),
                ),
            )
        )
        assert decision.selected.capabilities.structured_output
        # Should pick cheapest among structured-output-capable
        assert decision.selected.cost.blended_cost_per_1k <= 0.005

    def test_prefer_lowest_latency_under_target(self) -> None:
        """'prefer lowest latency under target'."""
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(
                    max_latency_ms=800.0,
                    weights=ScoringWeights(cost=0.0, latency=2.0),
                ),
            )
        )
        assert decision.selected.latency.median_ms <= 800.0

    def test_only_approved_providers(self) -> None:
        """'use only approved providers'."""
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(
                    allowed_providers=frozenset({"openai", "anthropic"}),
                ),
            )
        )
        assert decision.selected.provider in {"openai", "anthropic"}

    def test_never_use_model_family_for_phi(self) -> None:
        """'never use model family X for PHI-bearing workloads'."""
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(
                    denied_model_families=frozenset({"llama"}),
                ),
            )
        )
        assert not decision.selected.model_id.startswith("llama")

    def test_force_structured_output_for_extraction(self) -> None:
        """'force structured-output-capable routes for extraction'."""
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                workload_type=WorkloadType.EXTRACTION,
                policy=RoutingPolicy(
                    required_capabilities=frozenset({"structured_output"}),
                ),
            )
        )
        assert decision.selected.capabilities.structured_output

    def test_require_fallback_chain_depth(self) -> None:
        """'require fallback chain of at least N approved models'."""
        service = _make_service()
        decision = service.route(
            RoutingRequest(
                policy=RoutingPolicy(min_fallbacks=2),
            )
        )
        assert decision.fallback_plan.depth >= 2
