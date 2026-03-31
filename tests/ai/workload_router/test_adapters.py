"""Tests for electripy.ai.workload_router.adapters."""

from __future__ import annotations

from electripy.ai.workload_router.adapters import (
    InMemoryCatalog,
    LoggingTelemetryAdapter,
    NoOpTelemetryAdapter,
    StaticHealthAdapter,
)
from electripy.ai.workload_router.domain import (
    CandidateModel,
    CandidateScore,
    FallbackPlan,
    RouteExplanation,
    RoutingDecision,
    RoutingRequest,
)
from electripy.ai.workload_router.ports import (
    HealthStatusPort,
    ModelCatalogPort,
    TelemetryHookPort,
)


def _dummy_decision() -> RoutingDecision:
    c = CandidateModel(model_id="test", provider="p")
    return RoutingDecision(
        selected=c,
        fallback_plan=FallbackPlan(),
        explanation=RouteExplanation(
            selected_model_id="test",
            scores=(CandidateScore(candidate=c, total_score=1.0),),
            disqualified=(),
            policy_summary="test",
        ),
        request=RoutingRequest(),
    )


class TestInMemoryCatalog:
    def test_empty(self) -> None:
        catalog = InMemoryCatalog()
        assert catalog.list_candidates() == []

    def test_with_candidates(self) -> None:
        c1 = CandidateModel(model_id="a", provider="p")
        c2 = CandidateModel(model_id="b", provider="p")
        catalog = InMemoryCatalog(candidates=[c1, c2])
        result = catalog.list_candidates()
        assert len(result) == 2
        assert result[0].model_id == "a"

    def test_returns_copy(self) -> None:
        c = CandidateModel(model_id="a", provider="p")
        catalog = InMemoryCatalog(candidates=[c])
        result = catalog.list_candidates()
        result.clear()
        assert len(catalog.list_candidates()) == 1

    def test_satisfies_port(self) -> None:
        assert isinstance(InMemoryCatalog(), ModelCatalogPort)


class TestStaticHealthAdapter:
    def test_always_healthy(self) -> None:
        adapter = StaticHealthAdapter()
        assert adapter.is_healthy("gpt-4o", "openai")
        assert adapter.is_healthy("claude-3", "anthropic")

    def test_satisfies_port(self) -> None:
        assert isinstance(StaticHealthAdapter(), HealthStatusPort)


class TestNoOpTelemetryAdapter:
    def test_does_not_raise(self) -> None:
        adapter = NoOpTelemetryAdapter()
        adapter.on_routing_decision(_dummy_decision())

    def test_satisfies_port(self) -> None:
        assert isinstance(NoOpTelemetryAdapter(), TelemetryHookPort)


class TestLoggingTelemetryAdapter:
    def test_does_not_raise(self) -> None:
        adapter = LoggingTelemetryAdapter()
        adapter.on_routing_decision(_dummy_decision())

    def test_satisfies_port(self) -> None:
        assert isinstance(LoggingTelemetryAdapter(), TelemetryHookPort)
