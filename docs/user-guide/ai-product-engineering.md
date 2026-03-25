# AI Product Engineering Utilities

ElectriPy Studio includes lightweight, composable Python components for advanced AI product development.

## What this adds

- Streaming chat primitives for sync and async token/delta handling.
- Deterministic agent runtime primitives for ordered tool execution.
- RAG quality metrics and retrieval drift comparison helpers.
- Hallucination-risk reduction helpers through grounding/citation checks.
- Response robustness helpers for JSON extraction, repair, and strict field validation.
- Prompt templating with variable injection and few-shot example management.
- Token budget tracking, budget checking, and multi-strategy truncation.
- Priority-based context window assembly with automatic low-priority block dropping.
- Rule-based model routing for cost/capability optimization.
- Sliding-window conversation memory with token-budget-aware trimming.
- Declarative tool registry with automatic JSON schema generation and OpenAI export.
- Deterministic policy gateway for pre/post/stream/tool safety decisions.
- Bounded agent collaboration runtime for specialist agent handoffs.

## Component map

- `electripy.ai.streaming_chat`
- `electripy.ai.agent_runtime`
- `electripy.ai.rag_quality`
- `electripy.ai.hallucination_guard`
- `electripy.ai.response_robustness`
- `electripy.ai.prompt_engine`
- `electripy.ai.token_budget`
- `electripy.ai.context_assembly`
- `electripy.ai.model_router`
- `electripy.ai.conversation_memory`
- `electripy.ai.tool_registry`
- `electripy.ai.policy_gateway`
- `electripy.ai.agent_collaboration`

## Quick examples

### Streaming chat collection

```python
from electripy.ai.streaming_chat import StreamChunk, collect_text

chunks = [
    StreamChunk(index=0, delta_text="Hello"),
    StreamChunk(index=1, delta_text=" world", done=True),
]

text = collect_text(chunks)
assert text == "Hello world"
```

### Agent runtime plan execution

```python
from electripy.ai.agent_runtime import AgentExecutor, ToolInvocation

class ToolRunner:
    def execute(self, name: str, args: dict[str, object]) -> str:
        return f"{name}:{args}"

executor = AgentExecutor(tool_port=ToolRunner())
result = executor.run([ToolInvocation(name="search", args={"q": "latency"})])
assert result.all_successful
```

### RAG metric and drift checks

```python
from electripy.ai.rag_quality import hit_rate_at_k, retrieval_drift, RetrievalSnapshot

score = hit_rate_at_k(["a", "b", "c"], ["b"], 3)
assert score == 1.0

drift = retrieval_drift(
    RetrievalSnapshot(query_id="q1", retrieved_ids=["a", "b"]),
    RetrievalSnapshot(query_id="q1", retrieved_ids=["b", "c"]),
    k=2,
)
assert drift.overlap_ratio == 0.5
```

### Grounding check for generated responses

```python
from electripy.ai.hallucination_guard import evaluate_grounding

result = evaluate_grounding(
    response_text="Paris is in France [cite:doc-1]",
    evidence_texts=["Paris is the capital of France."],
)

assert result.grounded
```

### Robust JSON parsing and validation

```python
from electripy.ai.response_robustness import parse_json_with_repair, require_fields

parsed = parse_json_with_repair("```json\n{\"answer\": \"ok\",}\n```")
require_fields(parsed.value, ["answer"])
```

### Prompt templating and composition

```python
from electripy.ai.prompt_engine import compose_messages, FewShotExample

prompt = compose_messages(
    system="You are a {{persona}}.",
    few_shot=[FewShotExample(user="2+2?", assistant="4")],
    user="Summarize: {{text}}",
    variables={"persona": "helpful assistant", "text": "ElectriPy is great"},
)

# Ready for any LLM API
messages = prompt.to_dicts()
```

### Token budget management

```python
from electripy.ai.token_budget import (
    CharEstimatorTokenizer,
    fits_budget,
    truncate_to_budget,
    TruncationStrategy,
)

tokenizer = CharEstimatorTokenizer()

assert fits_budget("short text", budget=100, tokenizer=tokenizer)

result = truncate_to_budget(
    "A very long document that exceeds the budget...",
    budget=5,
    tokenizer=tokenizer,
    strategy=TruncationStrategy.TAIL,
)
assert result.was_truncated
```

### Priority-based context assembly

```python
from electripy.ai.context_assembly import (
    ContextBlock,
    ContextPriority,
    assemble_context,
)
from electripy.ai.token_budget import CharEstimatorTokenizer

blocks = [
    ContextBlock(label="system", content="You are helpful.", priority=ContextPriority.CRITICAL),
    ContextBlock(label="docs", content="Long reference document...", priority=ContextPriority.LOW),
    ContextBlock(label="query", content="What is X?", priority=ContextPriority.HIGH),
]

result = assemble_context(blocks, budget=50, tokenizer=CharEstimatorTokenizer())
# Low-priority blocks are dropped first when budget is exceeded
print(result.dropped_labels)
```

### Rule-based model routing

```python
from electripy.ai.model_router import (
    CostTier,
    ModelProfile,
    ModelRouter,
    RoutingRule,
)

router = ModelRouter(models=[
    ModelProfile(model_id="gpt-4o-mini", provider="openai", cost_tier=CostTier.LOW, supports_structured_output=True),
    ModelProfile(model_id="gpt-4o", provider="openai", cost_tier=CostTier.HIGH, supports_vision=True),
])

decision = router.route([
    RoutingRule(name="needs-vision", predicate=lambda m: m.supports_vision),
])
assert decision.selected.model_id == "gpt-4o"
```

### Conversation memory with token budgets

```python
from electripy.ai.conversation_memory import (
    ConversationWindow,
    TurnRole,
    append_turn,
    trim_to_budget,
)
from electripy.ai.token_budget import CharEstimatorTokenizer

tokenizer = CharEstimatorTokenizer()
window = ConversationWindow()
window = append_turn(window, TurnRole.SYSTEM, "You are helpful.", tokenizer)
window = append_turn(window, TurnRole.USER, "Hello!", tokenizer)
window = append_turn(window, TurnRole.ASSISTANT, "Hi there!", tokenizer)

# Trim to budget, always preserving system messages
trimmed = trim_to_budget(window, budget=20, tokenizer=tokenizer, preserve_system=True)
messages = trimmed.to_dicts()
```

### Declarative tool registry

```python
from electripy.ai.tool_registry import tool_from_function, ToolRegistry

def search(query: str, limit: int = 10) -> list[str]:
    """Search the knowledge base."""
    ...

registry = ToolRegistry()
registry.register(tool_from_function(search, name="search"))

# Export for OpenAI function-calling API
tools = registry.to_openai_tools()
```

### Policy gateway hooks for LLM flows

```python
from electripy.ai.llm_gateway import LlmGatewaySettings
from electripy.ai.policy_gateway import PolicyGateway, build_llm_policy_hooks

policy = PolicyGateway(rules=[...])
request_hook, response_hook = build_llm_policy_hooks(policy)

settings = LlmGatewaySettings(
    request_hook=request_hook,
    response_hook=response_hook,
)
```

### Specialist-agent collaboration runtime

```python
from electripy.ai.agent_collaboration import (
    AgentCollaborationRuntime,
    AgentTurnResult,
    CollaborationTask,
    make_message,
)

class Planner:
    def handle(self, message, *, task):
        return AgentTurnResult(
            produced_messages=[
                make_message(
                    task_id=task.task_id,
                    seq=1,
                    from_agent="planner",
                    to_agent="verifier",
                    content=f"plan::{task.objective}",
                )
            ]
        )

class Verifier:
    def handle(self, message, *, task):
        return AgentTurnResult(completed=True, outcome="verified")

runtime = AgentCollaborationRuntime(agents={"planner": Planner(), "verifier": Verifier()})
result = runtime.run(
    task=CollaborationTask(task_id="task-1", objective="triage"),
    entry_agent="planner",
    input_text="begin",
)
assert result.success
```
