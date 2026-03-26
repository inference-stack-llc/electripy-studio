# Fallback Chain

The **Fallback Chain** provides automatic provider failover for LLM
calls.  Wrap multiple `SyncLlmPort` adapters in a `FallbackChainPort`
and the chain tries each provider in order until one succeeds.

## When to use it

- You run multi-provider setups (OpenAI + Anthropic + local) and want
  seamless failover without retry loops in calling code.
- A primary provider is occasionally rate-limited or down.
- You want to track **which** provider handled each request.

## Core concepts

| Symbol | Role |
|--------|------|
| `FallbackChainPort` | Implements `SyncLlmPort`, wraps N providers in ranked order. |

On success the response carries
`metadata["_fallback_provider_index"]` — the zero-based index of the
provider that handled the call.

## Basic example

```python
from electripy.ai.fallback_chain import FallbackChainPort
from electripy.ai.llm_gateway import build_llm_sync_client

chain = FallbackChainPort(
    providers=[
        build_llm_sync_client("openai"),
        build_llm_sync_client("anthropic"),
        build_llm_sync_client("ollama"),
    ],
)

response = chain.complete(request)
print(response.metadata["_fallback_provider_index"])  # 0, 1, or 2
```

## Behaviour on failure

- Exceptions from non-final providers are **swallowed** (logged at
  `DEBUG` level).
- If **all** providers fail, the exception from the **last** provider
  is re-raised — giving you a clear error from the final fallback.

## Combining with other utilities

```python
from electripy.concurrency.circuit_breaker import CircuitBreaker

# Wrap individual providers in circuit breakers, then chain them.
cb_openai = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)
cb_anthropic = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)

chain = FallbackChainPort(
    providers=[
        cb_openai(openai_adapter.complete),
        cb_anthropic(anthropic_adapter.complete),
    ],
)
```
