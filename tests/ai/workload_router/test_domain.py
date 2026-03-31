"""Tests for electripy.ai.workload_router.domain."""

from __future__ import annotations

import pytest

from electripy.ai.workload_router.domain import (
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


class TestWorkloadType:
    def test_values(self) -> None:
        assert WorkloadType.CHAT == "chat"
        assert WorkloadType.EXTRACTION == "extraction"
        assert WorkloadType.TOOL_USE == "tool_use"
        assert WorkloadType.EMBEDDING == "embedding"
        assert WorkloadType.LONG_CONTEXT == "long_context"
        assert WorkloadType.STRUCTURED_OUTPUT == "structured_output"

    def test_is_str(self) -> None:
        assert isinstance(WorkloadType.CHAT, str)

    def test_all_values_are_unique(self) -> None:
        values = [wt.value for wt in WorkloadType]
        assert len(values) == len(set(values))


class TestCapabilityProfile:
    def test_defaults(self) -> None:
        cap = CapabilityProfile()
        assert not cap.structured_output
        assert not cap.tool_use
        assert cap.streaming
        assert not cap.vision
        assert not cap.reasoning
        assert cap.workload_types == frozenset()

    def test_custom(self) -> None:
        cap = CapabilityProfile(
            structured_output=True,
            tool_use=True,
            workload_types=frozenset({WorkloadType.EXTRACTION}),
        )
        assert cap.structured_output
        assert cap.tool_use
        assert WorkloadType.EXTRACTION in cap.workload_types

    def test_frozen(self) -> None:
        cap = CapabilityProfile()
        with pytest.raises(AttributeError):
            cap.streaming = False  # type: ignore[misc]


class TestCostProfile:
    def test_defaults(self) -> None:
        cost = CostProfile()
        assert cost.cost_per_1k_input == 0.0
        assert cost.cost_per_1k_output == 0.0

    def test_blended_cost(self) -> None:
        cost = CostProfile(cost_per_1k_input=0.01, cost_per_1k_output=0.03)
        assert cost.blended_cost_per_1k == pytest.approx(0.02)

    def test_blended_cost_zero(self) -> None:
        cost = CostProfile()
        assert cost.blended_cost_per_1k == 0.0


class TestLatencyProfile:
    def test_defaults(self) -> None:
        lat = LatencyProfile()
        assert lat.median_ms == 500.0
        assert lat.p99_ms == 5000.0


class TestReliabilityProfile:
    def test_defaults(self) -> None:
        rel = ReliabilityProfile()
        assert rel.availability == 0.99
        assert rel.healthy


class TestContextWindowProfile:
    def test_defaults(self) -> None:
        ctx = ContextWindowProfile()
        assert ctx.max_input_tokens == 4096
        assert ctx.max_output_tokens == 4096


class TestCandidateModel:
    def test_basic(self) -> None:
        c = CandidateModel(model_id="gpt-4o-mini", provider="openai")
        assert c.model_id == "gpt-4o-mini"
        assert c.provider == "openai"

    def test_defaults(self) -> None:
        c = CandidateModel(model_id="x", provider="y")
        assert isinstance(c.capabilities, CapabilityProfile)
        assert isinstance(c.cost, CostProfile)
        assert isinstance(c.latency, LatencyProfile)
        assert isinstance(c.reliability, ReliabilityProfile)
        assert isinstance(c.context_window, ContextWindowProfile)
        assert c.tags == frozenset()

    def test_frozen(self) -> None:
        c = CandidateModel(model_id="x", provider="y")
        with pytest.raises(AttributeError):
            c.model_id = "z"  # type: ignore[misc]


class TestScoringWeights:
    def test_defaults(self) -> None:
        w = ScoringWeights()
        assert w.cost == 1.0
        assert w.latency == 0.5
        assert w.reliability == 0.3
        assert w.context_window == 0.1

    def test_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="cost"):
            ScoringWeights(cost=-1.0)

    def test_negative_latency_raises(self) -> None:
        with pytest.raises(ValueError, match="latency"):
            ScoringWeights(latency=-0.1)


class TestRoutingPolicy:
    def test_defaults(self) -> None:
        p = RoutingPolicy()
        assert p.constraints == ()
        assert isinstance(p.weights, ScoringWeights)
        assert p.allowed_providers == frozenset()
        assert p.denied_providers == frozenset()
        assert p.max_cost_per_1k is None
        assert p.max_latency_ms is None
        assert p.min_context_tokens is None
        assert p.min_fallbacks == 0

    def test_custom(self) -> None:
        p = RoutingPolicy(
            allowed_providers=frozenset({"openai"}),
            max_cost_per_1k=0.01,
            max_latency_ms=2000.0,
            min_context_tokens=8000,
            required_capabilities=frozenset({"tool_use"}),
        )
        assert "openai" in p.allowed_providers
        assert p.max_cost_per_1k == 0.01


class TestRoutingRequest:
    def test_defaults(self) -> None:
        r = RoutingRequest()
        assert r.workload_type == WorkloadType.CHAT
        assert r.estimated_input_tokens is None
        assert r.metadata == {}

    def test_custom(self) -> None:
        r = RoutingRequest(
            workload_type=WorkloadType.EXTRACTION,
            estimated_input_tokens=5000,
        )
        assert r.workload_type == WorkloadType.EXTRACTION
        assert r.estimated_input_tokens == 5000


class TestFallbackPlan:
    def test_empty(self) -> None:
        fp = FallbackPlan()
        assert fp.depth == 0
        assert fp.candidates == ()

    def test_with_candidates(self) -> None:
        c1 = CandidateModel(model_id="a", provider="p")
        c2 = CandidateModel(model_id="b", provider="p")
        fp = FallbackPlan(candidates=(c1, c2))
        assert fp.depth == 2


class TestRoutingConstraint:
    def test_basic(self) -> None:
        rc = RoutingConstraint(
            name="cheap",
            predicate=lambda c: c.cost.blended_cost_per_1k < 0.01,
        )
        assert rc.name == "cheap"
        cheap = CandidateModel(
            model_id="x",
            provider="y",
            cost=CostProfile(cost_per_1k_input=0.001, cost_per_1k_output=0.001),
        )
        assert rc.predicate(cheap)

    def test_predicate_false(self) -> None:
        rc = RoutingConstraint(
            name="cheap",
            predicate=lambda c: c.cost.blended_cost_per_1k < 0.001,
        )
        expensive = CandidateModel(
            model_id="x",
            provider="y",
            cost=CostProfile(cost_per_1k_input=0.05, cost_per_1k_output=0.05),
        )
        assert not rc.predicate(expensive)


class TestDisqualifiedCandidate:
    def test_basic(self) -> None:
        c = CandidateModel(model_id="x", provider="y")
        dq = DisqualifiedCandidate(candidate=c, reason="too expensive")
        assert dq.reason == "too expensive"
        assert dq.candidate.model_id == "x"


class TestCandidateScore:
    def test_basic(self) -> None:
        c = CandidateModel(model_id="x", provider="y")
        score = CandidateScore(candidate=c, total_score=0.85)
        assert score.total_score == 0.85
        assert score.cost_score == 0.0


class TestRouteExplanation:
    def test_basic(self) -> None:
        c = CandidateModel(model_id="x", provider="y")
        score = CandidateScore(candidate=c, total_score=0.85)
        exp = RouteExplanation(
            selected_model_id="x",
            scores=(score,),
            disqualified=(),
            policy_summary="default",
        )
        assert exp.selected_model_id == "x"
        assert len(exp.scores) == 1
        assert exp.disqualified == ()


class TestRoutingDecision:
    def test_basic(self) -> None:
        c = CandidateModel(model_id="x", provider="y")
        score = CandidateScore(candidate=c, total_score=0.85)
        exp = RouteExplanation(
            selected_model_id="x",
            scores=(score,),
            disqualified=(),
            policy_summary="test",
        )
        decision = RoutingDecision(
            selected=c,
            fallback_plan=FallbackPlan(),
            explanation=exp,
            request=RoutingRequest(),
        )
        assert decision.selected.model_id == "x"
        assert decision.fallback_plan.depth == 0
