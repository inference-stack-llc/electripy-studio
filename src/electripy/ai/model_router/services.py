"""Services for rule-based model selection."""

from __future__ import annotations

from collections.abc import Sequence

from .domain import CostTier, ModelProfile, RoutingDecision, RoutingRule
from .errors import NoMatchingModelError

_COST_ORDER = {
    CostTier.FREE: 0,
    CostTier.LOW: 1,
    CostTier.MEDIUM: 2,
    CostTier.HIGH: 3,
    CostTier.PREMIUM: 4,
}


class ModelRouter:
    """Route LLM requests to models using composable rules.

    The router evaluates all registered rules against a registry of model
    profiles and selects the cheapest model that satisfies all rules.

    Args:
        models: Available model profiles.

    Example::

        router = ModelRouter(models=[
            ModelProfile(model_id="gpt-4o-mini", provider="openai", cost_tier=CostTier.LOW),
            ModelProfile(model_id="gpt-4o", provider="openai", cost_tier=CostTier.HIGH),
        ])
        decision = router.route([
            RoutingRule(name="needs-json", predicate=lambda m: m.supports_structured_output),
        ])
    """

    def __init__(self, models: Sequence[ModelProfile]) -> None:
        if not models:
            raise ValueError("At least one model profile is required")
        self._models = list(models)

    def route(
        self,
        rules: Sequence[RoutingRule],
        *,
        prefer_cheapest: bool = True,
    ) -> RoutingDecision:
        """Select a model that satisfies all the given rules.

        Args:
            rules: Rules that models must satisfy.
            prefer_cheapest: If True, pick the cheapest matching model.

        Returns:
            A RoutingDecision with the selected model.

        Raises:
            NoMatchingModelError: If no model satisfies all rules.
        """
        candidates = self._models
        matched_names: list[str] = []

        for rule in rules:
            matched = [m for m in candidates if rule.predicate(m)]
            if not matched:
                raise NoMatchingModelError(f"No model satisfies rule {rule.name!r}")
            candidates = matched
            matched_names.append(rule.name)

        if prefer_cheapest:
            candidates.sort(key=lambda m: _COST_ORDER.get(m.cost_tier, 99))

        return RoutingDecision(
            selected=candidates[0],
            matched_rules=matched_names,
            candidates_considered=len(self._models),
        )

    def cheapest(self) -> ModelProfile:
        """Return the cheapest registered model.

        Returns:
            The model profile with the lowest cost tier.
        """
        return min(self._models, key=lambda m: _COST_ORDER.get(m.cost_tier, 99))

    def by_id(self, model_id: str) -> ModelProfile:
        """Look up a model profile by its model_id.

        Args:
            model_id: The model identifier to look up.

        Returns:
            The matching model profile.

        Raises:
            NoMatchingModelError: If no model has the given ID.
        """
        for model in self._models:
            if model.model_id == model_id:
                return model
        raise NoMatchingModelError(f"No model with id {model_id!r}")
