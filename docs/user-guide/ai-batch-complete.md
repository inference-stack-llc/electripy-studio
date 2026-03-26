# Batch Complete

`batch_complete()` fans out many LLM requests in parallel with bounded
concurrency, an optional progress callback, and per-request error
isolation.

## When to use it

- You have 10–10 000 prompts to process and want to maximise
  throughput without melting your rate limit.
- You need **order-preserving** results — `results[i]` always
  corresponds to `requests[i]`.
- You want failed requests to capture the exception rather than crash
  the entire batch.

## Core concepts

| Symbol | Role |
|--------|------|
| `batch_complete()` | Main entry point — keyword-only, returns `list[BatchResult]`. |
| `BatchResult` | Type alias: `LlmResponse \| Exception`. |

## Basic example

```python
from electripy.ai.batch_complete import batch_complete
from electripy.ai.llm_gateway import build_llm_sync_client
from electripy.ai.llm_gateway.domain import LlmRequest, ChatMessage, MessageRole

port = build_llm_sync_client("openai")

requests = [
    LlmRequest(
        model="gpt-4o-mini",
        messages=[ChatMessage(role=MessageRole.USER, content=f"Summarise: {doc}")],
    )
    for doc in documents
]

results = batch_complete(
    port=port,
    requests=requests,
    max_concurrency=5,
    on_progress=lambda done, total: print(f"{done}/{total}"),
)

for r in results:
    if isinstance(r, Exception):
        print(f"FAILED: {r}")
    else:
        print(r.text[:80])
```

## Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `port` | `SyncLlmPort` | — | Any LLM adapter. |
| `requests` | `Sequence[LlmRequest]` | — | Ordered prompts. |
| `max_concurrency` | `int` | 5 | Max in-flight calls. |
| `timeout` | `float \| None` | `None` | Per-request timeout forwarded to the port. |
| `on_progress` | `Callable[[int, int], None] \| None` | `None` | `(completed, total)` callback. |

## Error handling

Each request is independent.  If one fails, the exception is captured
in the corresponding result slot — the rest of the batch continues.
This means you never lose partial work to one bad prompt.
