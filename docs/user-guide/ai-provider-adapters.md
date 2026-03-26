# Provider Adapters

ElectriPy's LLM Gateway ships with **OpenAI**, **Anthropic**, and **Ollama**
adapters in addition to the generic HTTP-JSON adapter.  Each adapter
implements `SyncLlmPort`, so switching providers is a one-line change.

## When to use them

- You want to call **GPT** models via the OpenAI Chat Completions API.
- You want to call **Claude** models via the Anthropic Messages API.
- You run **Ollama** locally and need a zero-SDK adapter that talks
  HTTP directly.
- You want to swap providers without touching application logic.

## OpenAI adapter

Requires the official OpenAI SDK (`pip install openai`). The SDK
is lazy-imported — no import-time cost if you don't use this adapter.

```python
from electripy.ai.llm_gateway import (
    OpenAiSyncAdapter,
    LlmMessage,
    LlmRequest,
)

adapter = OpenAiSyncAdapter(api_key="sk-...")

request = LlmRequest(
    model="gpt-4o-mini",
    messages=[LlmMessage.user("Explain hexagonal architecture.")],
)

response = adapter.complete(request)
print(response.text)
```

### Custom base URL (Azure, OpenRouter, etc.)

```python
adapter = OpenAiSyncAdapter(
    api_key="sk-...",
    base_url="https://your-azure-endpoint.openai.azure.com/",
    organization="org-...",
)
```

## Anthropic adapter

Requires the official Anthropic SDK (`pip install anthropic`). The SDK
is lazy-imported — no import-time cost if you don't use this adapter.

```python
from electripy.ai.llm_gateway import (
    AnthropicSyncAdapter,
    LlmMessage,
    LlmRequest,
)

adapter = AnthropicSyncAdapter(api_key="sk-ant-...")

request = LlmRequest(
    model="claude-sonnet-4-20250514",
    messages=[LlmMessage.user("Explain hexagonal architecture.")],
)

response = adapter.complete(request)
print(response.text)
```

### System message handling

Anthropic's Messages API accepts system messages via a separate
`system` parameter rather than in the messages array.  The adapter
automatically extracts any system-role messages from `LlmRequest.messages`
and sends them correctly:

```python
request = LlmRequest(
    model="claude-sonnet-4-20250514",
    messages=[
        LlmMessage.system("You are a helpful coding assistant."),
        LlmMessage.user("Write a Python hello-world."),
    ],
)
# System message is extracted and sent via the `system` kwarg.
response = adapter.complete(request)
```

## Ollama adapter

Calls the Ollama `/api/chat` endpoint via `httpx` — no SDK needed. Just
point it at your running Ollama server:

```python
from electripy.ai.llm_gateway import (
    OllamaSyncAdapter,
    LlmMessage,
    LlmRequest,
)

adapter = OllamaSyncAdapter(base_url="http://localhost:11434")

request = LlmRequest(
    model="llama3",
    messages=[LlmMessage.user("Summarize the Unix philosophy.")],
)

response = adapter.complete(request)
print(response.text)
```

## Exception mapping

Both adapters map provider-specific errors to ElectriPy domain
exceptions so your retry and error-handling logic stays provider-agnostic:

| Provider condition      | Domain exception        |
|-------------------------|-------------------------|
| HTTP 429 / rate limit   | `RateLimitedError`      |
| HTTP 5xx / server error | `TransientLlmError`     |

## Using adapters with the gateway client

Adapters plug directly into the gateway orchestration layer for retries,
token budgets, safety hooks, and structured output:

```python
from electripy.ai.llm_gateway import (
    AnthropicSyncAdapter,
    LlmGatewaySettings,
    LlmGatewaySyncClient,
)

adapter = AnthropicSyncAdapter(api_key="sk-ant-...")
settings = LlmGatewaySettings(default_model="claude-sonnet-4-20250514")
client = LlmGatewaySyncClient(port=adapter, settings=settings)
```

## Integration with other components

- **LLM Caching** — wrap any adapter with `CachedLlmPort` for
  transparent response caching.
- **Replay Tape** — record Anthropic or Ollama calls for deterministic
  offline replay in tests.
- **Structured Output** — pass any adapter as the `llm_port` to
  `StructuredOutputExtractor`.
