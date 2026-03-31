"""Domain models for the workload routing engine.

This module defines the core data structures for intelligent model and
workload routing: candidate profiles, routing policies, constraints,
requests, decisions, explanations, and fallback plans.

All models are provider-neutral and free of third-party dependencies.

Example:
    from electripy.ai.workload_router.domain import (
        CandidateModel,
        RoutingPolicy,
        RoutingRequest,
        WorkloadType,
    )

    candidate = CandidateModel(
        model_id="gpt-4o-mini",
        provider="openai",
        capabilities=CapabilityProfile(structured_output=True),
        cost=CostProfile(cost_per_1k_input=0.00015),
    )
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from enum import StrEnum

__all__ = [
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
]


# ── Enums ────────────────────────────────────────────────────────────


class WorkloadType(StrEnum):
    """Classification of the task being routed."""

    CHAT = "chat"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    RERANKING = "reranking"
    SUMMARIZATION = "summarization"
    TOOL_USE = "tool_use"
    REALTIME = "realtime"
    EMBEDDING = "embedding"
    LONG_CONTEXT = "long_context"
    STRUCTURED_OUTPUT = "structured_output"
    CODE_GENERATION = "code_generation"


# ── Profiles ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    """Feature capabilities of a candidate model.

    Attributes:
        structured_output: Supports structured / JSON mode.
        tool_use: Supports function calling / tool use.
        streaming: Supports streaming responses.
        vision: Supports image / multimodal inputs.
        reasoning: Supports extended reasoning / chain-of-thought.
        code_execution: Supports code execution sandbox.
        workload_types: Workload types this model is suitable for.
    """

    structured_output: bool = False
    tool_use: bool = False
    streaming: bool = True
    vision: bool = False
    reasoning: bool = False
    code_execution: bool = False
    workload_types: frozenset[WorkloadType] = field(default_factory=frozenset)


@dataclass(frozen=True, slots=True)
class CostProfile:
    """Pricing information for a candidate model.

    Costs are expressed in US dollars per 1 000 tokens.

    Attributes:
        cost_per_1k_input: Dollar cost per 1 000 input tokens.
        cost_per_1k_output: Dollar cost per 1 000 output tokens.
    """

    cost_per_1k_input: float = 0.0
    cost_per_1k_output: float = 0.0

    @property
    def blended_cost_per_1k(self) -> float:
        """Average of input and output cost (rough estimator)."""
        return (self.cost_per_1k_input + self.cost_per_1k_output) / 2.0


@dataclass(frozen=True, slots=True)
class LatencyProfile:
    """Latency characteristics of a candidate model.

    Attributes:
        median_ms: Typical (p50) response latency in milliseconds.
        p99_ms: 99th-percentile response latency in milliseconds.
    """

    median_ms: float = 500.0
    p99_ms: float = 5000.0


@dataclass(frozen=True, slots=True)
class ReliabilityProfile:
    """Reliability characteristics of a candidate model.

    Attributes:
        availability: Availability fraction (0.0–1.0).
        healthy: Whether the model is currently healthy.
    """

    availability: float = 0.99
    healthy: bool = True


@dataclass(frozen=True, slots=True)
class ContextWindowProfile:
    """Context window limits for a candidate model.

    Attributes:
        max_input_tokens: Maximum input tokens the model accepts.
        max_output_tokens: Maximum output tokens the model can produce.
    """

    max_input_tokens: int = 4096
    max_output_tokens: int = 4096


# ── Candidate ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class CandidateModel:
    """Full profile of a routeable model.

    Attributes:
        model_id: Unique identifier (e.g. ``"gpt-4o-mini"``).
        provider: Provider name (e.g. ``"openai"``, ``"anthropic"``).
        capabilities: Feature capabilities.
        cost: Pricing information.
        latency: Latency characteristics.
        reliability: Reliability information.
        context_window: Context window limits.
        tags: Arbitrary tags for custom constraints.
    """

    model_id: str
    provider: str
    capabilities: CapabilityProfile = field(default_factory=CapabilityProfile)
    cost: CostProfile = field(default_factory=CostProfile)
    latency: LatencyProfile = field(default_factory=LatencyProfile)
    reliability: ReliabilityProfile = field(default_factory=ReliabilityProfile)
    context_window: ContextWindowProfile = field(default_factory=ContextWindowProfile)
    tags: frozenset[str] = field(default_factory=frozenset)


# ── Constraints and Policy ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RoutingConstraint:
    """A named predicate that must hold for a candidate to be eligible.

    Attributes:
        name: Human-readable constraint name (used in explanations).
        predicate: Returns ``True`` if the candidate satisfies this
            constraint.
    """

    name: str
    predicate: Callable[[CandidateModel], bool]


@dataclass(frozen=True, slots=True)
class ScoringWeights:
    """Relative weights for the scoring dimensions.

    Higher weight means that dimension matters more in ranking.  All
    weights must be >= 0.  They do not need to sum to 1.

    Attributes:
        cost: Weight for cost (lower cost → higher score).
        latency: Weight for latency (lower latency → higher score).
        reliability: Weight for reliability (higher → higher score).
        context_window: Weight for context window size (larger → higher score).
    """

    cost: float = 1.0
    latency: float = 0.5
    reliability: float = 0.3
    context_window: float = 0.1

    def __post_init__(self) -> None:
        for fname in ("cost", "latency", "reliability", "context_window"):
            if getattr(self, fname) < 0:
                raise ValueError(f"Weight {fname!r} must be >= 0")


@dataclass(frozen=True, slots=True)
class RoutingPolicy:
    """Configurable routing policy.

    Combines hard constraints, scoring weights, provider/model
    allow/deny lists, budget limits, and latency SLOs into a single
    declarative policy.

    Attributes:
        constraints: Hard constraints that candidates must satisfy.
        weights: Scoring weights for ranking candidates.
        allowed_providers: If non-empty, only these providers are eligible.
        denied_providers: Providers that are never eligible.
        denied_model_families: Model-ID prefixes that are excluded.
        max_cost_per_1k: Budget ceiling — blended cost per 1 000 tokens.
        max_latency_ms: Maximum acceptable median latency in milliseconds.
        min_context_tokens: Minimum required input context window.
        required_capabilities: Capability flags that must be ``True``.
        preferred_providers: Providers scored with a small bonus.
        min_fallbacks: Minimum number of fallback candidates required.
    """

    constraints: tuple[RoutingConstraint, ...] = ()
    weights: ScoringWeights = field(default_factory=ScoringWeights)
    allowed_providers: frozenset[str] = field(default_factory=frozenset)
    denied_providers: frozenset[str] = field(default_factory=frozenset)
    denied_model_families: frozenset[str] = field(default_factory=frozenset)
    max_cost_per_1k: float | None = None
    max_latency_ms: float | None = None
    min_context_tokens: int | None = None
    required_capabilities: frozenset[str] = field(default_factory=frozenset)
    preferred_providers: frozenset[str] = field(default_factory=frozenset)
    min_fallbacks: int = 0


# ── Routing request / decision ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class RoutingRequest:
    """Input to the routing engine.

    Attributes:
        workload_type: Classification of the task.
        policy: Routing policy to apply.
        estimated_input_tokens: Optional token estimate used for budget
            and context-window checks.
        metadata: Arbitrary caller-provided metadata.
    """

    workload_type: WorkloadType = WorkloadType.CHAT
    policy: RoutingPolicy = field(default_factory=RoutingPolicy)
    estimated_input_tokens: int | None = None
    metadata: dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class DisqualifiedCandidate:
    """A candidate that was excluded during routing, with a reason.

    Attributes:
        candidate: The excluded model.
        reason: Human-readable explanation of exclusion.
    """

    candidate: CandidateModel
    reason: str


@dataclass(frozen=True, slots=True)
class CandidateScore:
    """Computed score for a candidate.

    Attributes:
        candidate: The scored model.
        total_score: Aggregated score (higher is better).
        cost_score: Cost dimension score.
        latency_score: Latency dimension score.
        reliability_score: Reliability dimension score.
        context_score: Context-window dimension score.
    """

    candidate: CandidateModel
    total_score: float
    cost_score: float = 0.0
    latency_score: float = 0.0
    reliability_score: float = 0.0
    context_score: float = 0.0


@dataclass(frozen=True, slots=True)
class RouteExplanation:
    """Detailed explanation of a routing decision.

    Attributes:
        selected_model_id: ID of the chosen model.
        scores: All scored candidates, ranked best to worst.
        disqualified: Candidates that failed hard constraints.
        policy_summary: Human-readable summary of the active policy.
    """

    selected_model_id: str
    scores: tuple[CandidateScore, ...]
    disqualified: tuple[DisqualifiedCandidate, ...]
    policy_summary: str


@dataclass(frozen=True, slots=True)
class FallbackPlan:
    """Ordered fallback candidates following the primary selection.

    Attributes:
        candidates: Ordered list of fallback models (best → worst).
    """

    candidates: tuple[CandidateModel, ...] = ()

    @property
    def depth(self) -> int:
        """Number of fallback candidates available."""
        return len(self.candidates)


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Complete result of a routing evaluation.

    Attributes:
        selected: The chosen model.
        fallback_plan: Ordered fallback candidates.
        explanation: Full decision explanation.
        request: The original routing request.
    """

    selected: CandidateModel
    fallback_plan: FallbackPlan
    explanation: RouteExplanation
    request: RoutingRequest
