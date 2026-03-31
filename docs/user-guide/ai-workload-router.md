# Workload Router

The **Workload Router** is ElectriPy's production-ready engine for
choosing the right LLM model and provider for a given workload. It
replaces bespoke if/else routing logic with a policy-driven,
cost-aware, latency-aware, capability-driven pipeline that produces
deterministic, explainable decisions with built-in fallback
orchestration.

## When to use it

Use the Workload Router when you want:

- **Policy-based selection** – enforce provider allow/deny lists, model
  family restrictions, budget caps, and latency SLOs declaratively.
- **Multi-dimensional scoring** – rank candidates across cost, latency,
  reliability, and context window size with configurable weights.
- **Fallback planning** – every decision comes with a ranked fallback
  chain so callers can retry on a secondary model automatically.
- **Explainability** – each decision includes an explanation with scores,
  disqualification reasons, and a policy summary for auditing.
- **Pluggable catalogs & health checks** – bring your own model registry
  and health probes via clean protocol ports.

## Core concepts

- **Domain models**:
  - `WorkloadType` – enum of workload categories: `CHAT`,
    `EXTRACTION`, `CLASSIFICATION`, `SUMMARIZATION`, `TOOL_USE`,
    `REALTIME`, `EMBEDDING`, `LONG_CONTEXT`, `STRUCTURED_OUTPUT`,
    `CODE_GENERATION`, `RERANKING`.
  - `CandidateModel` – a model/provider pair with capability, cost,
    latency, reliability, and context window profiles.
  - `RoutingPolicy` – a declarative constraint set: allowed/denied
    providers, denied model families, budget ceiling, latency SLO,
    required capabilities, preferred providers, minimum fallbacks, and
    custom `RoutingConstraint` predicates.
  - `RoutingRequest` – carries the workload type, policy, token
    estimate, and arbitrary metadata.
  - `RoutingDecision` – the selected model, a `FallbackPlan` of ranked
    alternatives, and a full `RouteExplanation`.
  - `ScoringWeights` – tune the relative importance of cost, latency,
    reliability, and context window in the ranking step.
- **Ports** (protocol interfaces):
  - `ModelCatalogPort` – list available candidate models.
  - `HealthStatusPort` – report whether a model/provider is healthy.
  - `TelemetryHookPort` – receive routing decisions for observability.
- **Adapters** (built-in implementations):
  - `InMemoryCatalog` – static list of candidates.
  - `StaticHealthAdapter` – treats all candidates as healthy.
  - `NoOpTelemetryAdapter` / `LoggingTelemetryAdapter`.
- **Service**:
  - `RoutingService` – orchestrates the full constraint → score → rank
    → fallback pipeline.

## Basic example

```python
from electripy.ai.workload_router import (
    CandidateModel,
    CapabilityProfile,
    CostProfile,
    InMemoryCatalog,
    LatencyProfile,
    RoutingRequest,
    RoutingService,
    WorkloadType,
)

gpt_mini = CandidateModel(
    model_id="gpt-4o-mini",
    provider="openai",
    capabilities=CapabilityProfile(),
    cost=CostProfile(cost_per_1k_input=0.15, cost_per_1k_output=0.60),
    latency=LatencyProfile(median_ms=200, p99_ms=800),
)

claude_sonnet = CandidateModel(
    model_id="claude-3.5-sonnet",
    provider="anthropic",
    capabilities=CapabilityProfile(structured_output=True, reasoning=True),
    cost=CostProfile(cost_per_1k_input=3.0, cost_per_1k_output=15.0),
    latency=LatencyProfile(median_ms=600, p99_ms=2000),
)

service = RoutingService(
    catalog=InMemoryCatalog(candidates=[gpt_mini, claude_sonnet]),
)

decision = service.route(RoutingRequest(workload_type=WorkloadType.CHAT))
print(decision.selected.model_id)   # "gpt-4o-mini" (cheapest by default)
print(decision.fallback_plan.depth)  # 1
```

## Policy-driven routing

Declare constraints in a `RoutingPolicy`; the router enforces them as
hard filters before scoring.

```python
from electripy.ai.workload_router import RoutingPolicy, RoutingRequest

# Only route to approved providers, never exceed $1/1k tokens
policy = RoutingPolicy(
    allowed_providers=frozenset({"openai", "anthropic"}),
    denied_model_families=frozenset({"llama"}),
    max_cost_per_1k=1.0,
    max_latency_ms=1000,
    required_capabilities=frozenset({"structured_output"}),
)

decision = service.route(RoutingRequest(policy=policy))
```

## Budget and latency convenience methods

```python
# Route under a strict budget ceiling
decision = service.route_by_budget(max_cost_per_1k=0.50)

# Route with a latency SLO
decision = service.route_by_latency(max_latency_ms=500)
```

## Custom constraints

Add arbitrary predicates that must pass for a candidate to be eligible:

```python
from electripy.ai.workload_router import RoutingConstraint

no_preview = RoutingConstraint(
    name="no-preview-models",
    predicate=lambda c: "preview" not in c.model_id,
)

policy = RoutingPolicy(constraints=(no_preview,))
decision = service.route(RoutingRequest(policy=policy))
```

## Tuning scoring weights

Adjust how dimensions are weighted during ranking:

```python
from electripy.ai.workload_router import ScoringWeights

# Favour low latency over cost
weights = ScoringWeights(cost=0.3, latency=1.0, reliability=0.5)
policy = RoutingPolicy(weights=weights)
decision = service.route(RoutingRequest(policy=policy))
```

## Preferred providers

Give a scoring bonus to certain providers without hard-filtering others:

```python
policy = RoutingPolicy(preferred_providers=frozenset({"anthropic"}))
decision = service.route(RoutingRequest(policy=policy))
# Anthropic candidates receive a +0.05 score bonus
```

## Fallback orchestration

Every `RoutingDecision` includes a `FallbackPlan` with the remaining
candidates ranked by score. Enforce a minimum depth:

```python
policy = RoutingPolicy(min_fallbacks=2)
decision = service.route(RoutingRequest(policy=policy))

for model in decision.fallback_plan.candidates:
    print(f"  fallback: {model.model_id}")
```

If fewer candidates survive filtering than required, the router raises
`NoCandidateError`.

## Explanation and debugging

Inspect why a model was selected or disqualified:

```python
explanation = service.explain(RoutingRequest(policy=policy))
print(explanation.selected_model_id)
print(explanation.policy_summary)

for score in explanation.scores:
    print(f"  {score.candidate.model_id}: {score.total:.4f}")

for dq in explanation.disqualified:
    print(f"  {dq.candidate.model_id}: {dq.reason}")
```

## Telemetry hook

Plug in observability by implementing `TelemetryHookPort`:

```python
from electripy.ai.workload_router import (
    LoggingTelemetryAdapter,
    RoutingService,
)

service = RoutingService(
    catalog=catalog,
    telemetry=LoggingTelemetryAdapter(),
)
```

Or implement your own:

```python
from electripy.ai.workload_router import RoutingDecision

class MetricsTelemetry:
    def on_routing_decision(self, decision: RoutingDecision) -> None:
        emit_metric("router.selected", decision.selected.model_id)
```

## Custom catalog and health adapters

Implement `ModelCatalogPort` and `HealthStatusPort` protocols to pull
model metadata from a database, config service, or live health endpoint:

```python
from electripy.ai.workload_router import CandidateModel

class DbCatalog:
    def list_candidates(self) -> list[CandidateModel]:
        return fetch_models_from_db()

class LiveHealth:
    def is_healthy(self, model_id: str, provider: str) -> bool:
        return ping_provider(provider)
```

## Error handling

- `NoCandidateError` – raised when no candidate survives filtering.
- `BudgetExceededError` – specialised subclass when all candidates
  exceed `max_cost_per_1k`.
- `ConstraintViolationError` – raised for policy constraint violations.

All errors extend `WorkloadRouterError`, which extends
`ElectriPyError(Exception)`.
