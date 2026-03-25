"""Domain models for model routing."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import Enum


class CostTier(Enum):
    """Cost tier classification for models."""

    FREE = "free"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    PREMIUM = "premium"


@dataclass(slots=True)
class ModelProfile:
    """Profile describing a model's capabilities and cost characteristics.

    Attributes:
        model_id: Unique model identifier (e.g. "gpt-4o-mini").
        provider: Provider name (e.g. "openai", "anthropic").
        cost_tier: Cost classification.
        max_context_tokens: Maximum context window size.
        supports_structured_output: Whether the model supports JSON mode.
        supports_vision: Whether the model supports image inputs.
        tags: Arbitrary tags for custom routing rules.
    """

    model_id: str
    provider: str
    cost_tier: CostTier = CostTier.MEDIUM
    max_context_tokens: int = 4096
    supports_structured_output: bool = False
    supports_vision: bool = False
    tags: frozenset[str] = field(default_factory=frozenset)


@dataclass(slots=True)
class RoutingRule:
    """A named predicate for filtering candidate models.

    Attributes:
        name: Human-readable rule name.
        predicate: Function that returns True if a model profile matches.
    """

    name: str
    predicate: Callable[[ModelProfile], bool]


@dataclass(slots=True)
class RoutingDecision:
    """Result of a routing decision.

    Attributes:
        selected: The selected model profile.
        matched_rules: Names of rules that participated in selection.
        candidates_considered: Number of models evaluated.
    """

    selected: ModelProfile
    matched_rules: list[str]
    candidates_considered: int
