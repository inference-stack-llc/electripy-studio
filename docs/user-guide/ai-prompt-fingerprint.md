# Prompt Fingerprint

`prompt_fingerprint()` produces a **deterministic SHA-256 hash** of an
LLM request, useful for caching, dedup, A/B bucketing, and audit trails.

## When to use it

- You need a stable key to identify "the same prompt" across runs.
- You want to detect prompt drift between deployments.
- You're building a custom cache or dedup layer on top of
  `LlmRequest`.

## Core concepts

| Symbol | Role |
|--------|------|
| `prompt_fingerprint(request)` | Returns a 64-character hex digest (full SHA-256). |
| `prompt_fingerprint_short(request, length=12)` | Truncated digest for logs and display. |

## What's hashed

The fingerprint is computed from a **canonical JSON** string containing:

- `model`
- `temperature`
- `messages` (role + content for each, in order)

Same inputs → same hash, always.

## Basic example

```python
from electripy.ai.prompt_fingerprint import prompt_fingerprint, prompt_fingerprint_short
from electripy.ai.llm_gateway.domain import LlmRequest, ChatMessage, MessageRole

request = LlmRequest(
    model="gpt-4o-mini",
    messages=[ChatMessage(role=MessageRole.USER, content="Hello!")],
)

print(prompt_fingerprint(request))        # "a3f2c8d1..."  (64 chars)
print(prompt_fingerprint_short(request))  # "a3f2c8d1e9b0" (12 chars)
```

## Cache key compatibility

This function uses the **same algorithm** as the LLM Caching Layer's
internal `compute_cache_key()`.  If you're building your own cache
backend or audit system, the keys will be identical:

```python
from electripy.ai.llm_cache.services import compute_cache_key
from electripy.ai.prompt_fingerprint import prompt_fingerprint

assert prompt_fingerprint(request) == compute_cache_key(request)
```
