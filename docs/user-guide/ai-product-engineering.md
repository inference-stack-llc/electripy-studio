# AI Product Engineering Utilities

ElectriPy Studio includes lightweight, composable Python components for advanced AI product development.

## What this adds

- Streaming chat primitives for sync and async token/delta handling.
- Deterministic agent runtime primitives for ordered tool execution.
- RAG quality metrics and retrieval drift comparison helpers.
- Hallucination-risk reduction helpers through grounding/citation checks.
- Response robustness helpers for JSON extraction, repair, and strict field validation.

## Component map

- `electripy.ai.streaming_chat`
- `electripy.ai.agent_runtime`
- `electripy.ai.rag_quality`
- `electripy.ai.hallucination_guard`
- `electripy.ai.response_robustness`

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
