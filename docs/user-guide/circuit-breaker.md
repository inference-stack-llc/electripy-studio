# Circuit Breaker

The **Circuit Breaker** protects against cascading failures from flaky
downstream services — LLM providers, databases, or any callable.

## When to use it

- A downstream provider is intermittently failing and you want to
  **fail fast** rather than pile up timeouts.
- You need a cooldown period before retrying a misbehaving service.
- You want to combine with `FallbackChainPort` so that a tripped
  breaker automatically diverts traffic to a backup provider.

## State machine

```
  ┌──────────┐  failure_threshold  ┌──────┐  recovery_timeout  ┌───────────┐
  │  CLOSED  │ ──────────────────► │ OPEN │ ─────────────────► │ HALF_OPEN │
  └──────────┘                     └──────┘                    └───────────┘
       ▲                                                            │
       │              success_threshold met                         │
       └────────────────────────────────────────────────────────────┘
       │                                                            │
       │              probe call fails                              │
       │                     ┌──────┐◄──────────────────────────────┘
       │                     │ OPEN │
       │                     └──────┘
```

## Core concepts

| Symbol | Role |
|--------|------|
| `CircuitBreaker` | Thread-safe FSM with `call()`, `state`, `reset()`. |
| `CircuitState` | StrEnum: `CLOSED`, `OPEN`, `HALF_OPEN`. |
| `CircuitOpenError` | Dataclass exception with `failures` and `recovery_timeout`. |

## Basic example

```python
from electripy.concurrency.circuit_breaker import CircuitBreaker, CircuitOpenError

cb = CircuitBreaker(failure_threshold=3, recovery_timeout=30.0)

try:
    result = cb.call(lambda: provider.complete(request))
except CircuitOpenError as e:
    print(f"Circuit open — {e.failures} failures, retry in {e.recovery_timeout}s")
    result = cached_fallback(request)
```

## Decorator usage

```python
cb = CircuitBreaker(failure_threshold=5, recovery_timeout=60.0)

@cb
def call_llm(prompt: str) -> str:
    return expensive_api_call(prompt)

# call_llm is now protected — raises CircuitOpenError when tripped.
```

## Parameters

| Param | Type | Default | Description |
|-------|------|---------|-------------|
| `failure_threshold` | `int` | 5 | Consecutive failures to trip open. |
| `recovery_timeout` | `float` | 30.0 | Seconds before half-open probe. |
| `success_threshold` | `int` | 1 | Consecutive successes in half-open to close. |

## Thread-safety

All state transitions are guarded by an internal lock.  Multiple
threads can call `cb.call(fn)` concurrently.
