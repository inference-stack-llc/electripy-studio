from __future__ import annotations

import pytest

from electripy.ai.model_router import (
    CostTier,
    ModelProfile,
    ModelRouter,
    NoMatchingModelError,
    RoutingRule,
)


def _make_models() -> list[ModelProfile]:
    return [
        ModelProfile(
            model_id="gpt-4o-mini",
            provider="openai",
            cost_tier=CostTier.LOW,
            max_context_tokens=128_000,
            supports_structured_output=True,
        ),
        ModelProfile(
            model_id="gpt-4o",
            provider="openai",
            cost_tier=CostTier.HIGH,
            max_context_tokens=128_000,
            supports_structured_output=True,
            supports_vision=True,
        ),
        ModelProfile(
            model_id="claude-3-haiku",
            provider="anthropic",
            cost_tier=CostTier.LOW,
            max_context_tokens=200_000,
        ),
    ]


class TestModelRouter:
    def test_route_cheapest_by_default(self) -> None:
        router = ModelRouter(models=_make_models())
        decision = router.route([])
        assert decision.selected.cost_tier == CostTier.LOW

    def test_route_with_rule(self) -> None:
        router = ModelRouter(models=_make_models())
        decision = router.route(
            [
                RoutingRule(
                    name="needs-vision",
                    predicate=lambda m: m.supports_vision,
                ),
            ]
        )
        assert decision.selected.model_id == "gpt-4o"
        assert "needs-vision" in decision.matched_rules

    def test_route_multiple_rules(self) -> None:
        router = ModelRouter(models=_make_models())
        decision = router.route(
            [
                RoutingRule(
                    name="structured",
                    predicate=lambda m: m.supports_structured_output,
                ),
                RoutingRule(
                    name="cheap",
                    predicate=lambda m: m.cost_tier == CostTier.LOW,
                ),
            ]
        )
        assert decision.selected.model_id == "gpt-4o-mini"

    def test_no_match_raises(self) -> None:
        router = ModelRouter(models=_make_models())
        with pytest.raises(NoMatchingModelError):
            router.route(
                [
                    RoutingRule(
                        name="impossible",
                        predicate=lambda m: m.provider == "nonexistent",
                    ),
                ]
            )

    def test_cheapest(self) -> None:
        router = ModelRouter(models=_make_models())
        assert router.cheapest().cost_tier == CostTier.LOW

    def test_by_id(self) -> None:
        router = ModelRouter(models=_make_models())
        model = router.by_id("gpt-4o")
        assert model.provider == "openai"

    def test_by_id_not_found(self) -> None:
        router = ModelRouter(models=_make_models())
        with pytest.raises(NoMatchingModelError, match="nonexistent"):
            router.by_id("nonexistent")

    def test_empty_models_raises(self) -> None:
        with pytest.raises(ValueError, match="At least one"):
            ModelRouter(models=[])

    def test_candidates_considered(self) -> None:
        router = ModelRouter(models=_make_models())
        decision = router.route([])
        assert decision.candidates_considered == 3
