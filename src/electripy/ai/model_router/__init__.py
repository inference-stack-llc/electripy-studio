"""Rule-based model routing for cost, latency, and capability optimization.

Purpose:
  - Route LLM requests to the most appropriate model based on configurable rules.
  - Enable tiered model strategies (e.g. cheap model for simple tasks, powerful model for complex ones).

Guarantees:
  - Routing is deterministic and fully offline (no network calls).
  - Rules are composable and inspectable.
"""

from __future__ import annotations

from .domain import CostTier, ModelProfile, RoutingDecision, RoutingRule
from .errors import ModelRouterError, NoMatchingModelError
from .services import ModelRouter

__all__ = [
    "CostTier",
    "ModelProfile",
    "RoutingDecision",
    "RoutingRule",
    "ModelRouterError",
    "NoMatchingModelError",
    "ModelRouter",
]
