# Recipe: LLM Gateway Basics

This recipe shows a small, end-to-end example of using the
`electripy.ai.llm_gateway` component to call an LLM in a
provider-agnostic way, with retries, token budgeting, and structured
output.

The example intentionally uses a fake adapter so it can be run offline
without any real provider credentials.

## Files

- `fake_provider.py` – A tiny fake implementation of `SyncLlmPort` and
  `AsyncLlmPort` that returns deterministic responses.
- `run_sync.py` – Synchronous example using `LlmGatewaySyncClient`.
- `run_async.py` – Asynchronous example using `LlmGatewayAsyncClient`.

## Running the examples

From the project root:

```bash
python -m recipes.02_llm_gateway.run_sync
python -m recipes.02_llm_gateway.run_async
```

Both scripts should print deterministic output without performing any
network IO.
