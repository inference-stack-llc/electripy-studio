# LLM Gateway

The **LLM Gateway** is ElectriPy's provider-agnostic client for calling
Large Language Models (LLMs) in a consistent, production-minded way. It
wraps provider SDKs or HTTP APIs behind ports and adapters so your
application code only talks to stable domain models and services.

For full, in-depth documentation see the component README at:

- `src/electripy/ai/llm_gateway/README.md`

This page focuses on how to *use* the gateway from an ElectriPy
consumer.

## When to use it

Use the LLM Gateway when you want:

- A small, stable surface for calling chat-style LLMs.
- Built-in reliability (retries, rate-limit handling) and token budgeting.
- Strict structured JSON responses with validation + a repair attempt.
- Safety seams for redaction and prompt-guard checks.
- Easy provider swapping (OpenAI today; OpenRouter/Copilot/Grok/Claude via
  adapters tomorrow).

## Core concepts

- **Domain models**:
  - `LlmMessage` – a single chat message with a role and content.
  - `LlmRequest` – normalized request with model, messages, temperature,
    token budget, metadata.
  - `LlmResponse` – normalized response with text, optional `raw_json`,
    usage, finish reason, model, and metadata.
  - `StructuredOutputSpec` – small schema describing expected JSON keys
    and Python types.
- **Ports**:
  - `SyncLlmPort` / `AsyncLlmPort` – provider interfaces used by the
    gateway services.
- **Services**:
  - `LlmGatewaySyncClient` / `LlmGatewayAsyncClient` – orchestration
    layers that apply retries, budgets, safety, and structured output.
- **Providers**:
  - OpenAI SDK adapters: `OpenAiSyncAdapter`, `OpenAiAsyncAdapter`.
  - Generic HTTP+JSON adapters: `HttpJsonChatSyncAdapter`,
    `HttpJsonChatAsyncAdapter` for OpenAI-compatible HTTP APIs.
  - Factory helpers: `build_llm_sync_client`, `build_llm_async_client`.
  - Provider registry: `register_llm_provider`,
    `list_registered_llm_providers`.
  - Observability hook: `LlmGatewaySettings.on_llm_call` – optional
    callback invoked after each successful LLM call with
    `(request, response, latency_ms)`.

## Basic example: OpenAI text completion

```python
from electripy.ai.llm_gateway import (
    LlmMessage,
    LlmRequest,
    build_llm_sync_client,
)

client = build_llm_sync_client("openai")

request = LlmRequest(
    model="gpt-4o-mini",
    messages=[LlmMessage.user("Summarize hexagonal architecture in 3 bullets.")],
)

response = client.complete(request)
print(response.text)
```

This will use the OpenAI Python SDK under the hood via
`OpenAiSyncAdapter`, but your code only depends on the gateway and
domain models.

## Structured JSON example with safety hooks

```python
from electripy.ai.llm_gateway import (
    HeuristicPromptGuard,
    LlmGatewaySettings,
    LlmMessage,
    LlmRequest,
    SimpleRedactor,
    StructuredOutputSpec,
    build_llm_sync_client,
)

settings = LlmGatewaySettings(
    default_model="gpt-4o-mini",
    enable_safe_logging=True,
    redactor=SimpleRedactor(),
    prompt_guard=HeuristicPromptGuard(),
)

client = build_llm_sync_client("openai", settings=settings)

spec = StructuredOutputSpec(
    name="MeetingSummary",
    field_types={
        "title": str,
        "decisions": list,
        "action_items": list,
    },
)

request = LlmRequest(
    model="gpt-4o-mini",
    messages=[LlmMessage.user("Summarize this meeting as structured JSON: ...")],
    max_output_tokens=512,
)

response = client.complete(request, structured_spec=spec)
print(response.raw_json["decisions"])
```

Here the gateway:

- Enforces an input character budget.
- Runs the prompt through `HeuristicPromptGuard` before calling the
  provider.
- Requests JSON-only output matching the `StructuredOutputSpec`.
- Attempts one automatic repair if the JSON is invalid.
- Logs redacted prompt/response previews when `enable_safe_logging` is
  enabled.

## HTTP JSON providers (OpenRouter, Copilot, Grok, ...)

For providers that expose OpenAI-compatible chat APIs over HTTP+JSON,
use the `"http-json"` provider with the factory helper:

```python
from electripy.ai.llm_gateway import (
    LlmMessage,
    LlmRequest,
    build_llm_sync_client,
)

client = build_llm_sync_client(
    "http-json",
    base_url="https://api.openrouter.ai",
    path="/v1/chat/completions",
    api_key="sk-...",
)

request = LlmRequest(
    model="openai/gpt-4o-mini",
    messages=[LlmMessage.user("Hello via OpenRouter!")],
)

response = client.complete(request)
print(response.text)
```

If a provider has a different JSON schema, subclass
`HttpJsonChatSyncAdapter` / `HttpJsonChatAsyncAdapter` and override the
payload and parsing methods, or wire a custom adapter into the gateway
using `LlmGatewaySyncClient` / `LlmGatewayAsyncClient` directly.

## Custom providers via the registry

You can add your own providers without modifying ElectriPy's source by
registering factories with the provider registry.

```python
from electripy.ai.llm_gateway import (
  LlmGatewaySettings,
  LlmGatewaySyncClient,
  LlmMessage,
  LlmRequest,
  build_llm_sync_client,
  register_llm_provider,
)


def my_sync_factory(settings: LlmGatewaySettings, kwargs: dict) -> LlmGatewaySyncClient:
  # Create your own SyncLlmPort implementation here.
  port = MyCustomSyncPort(**kwargs)
  return LlmGatewaySyncClient(port=port, settings=settings)


register_llm_provider("my-provider", sync_factory=my_sync_factory)

client = build_llm_sync_client("my-provider", api_key="secret-key")

response = client.complete(
  LlmRequest(
    model="my-model",
    messages=[LlmMessage.user("Hello from a custom provider")],
  )
)
```

The factory helpers first consult the registry; if a custom provider is
registered, it is used, otherwise the built-in providers (such as
`"openai"` or `"http-json"`) are used as a fallback.

## Observability hook for metrics and tracing

`LlmGatewaySettings` exposes an `on_llm_call` hook that is invoked after
every successful call. This is a convenient place to emit metrics or
traces without coupling the gateway to any specific observability
library.

```python
from electripy.ai.llm_gateway import LlmGatewaySettings, build_llm_sync_client


def my_llm_hook(request, response, latency_ms: float) -> None:
  # Send metrics to your preferred backend here.
  metrics_client.observe("llm.latency_ms", latency_ms, {"model": response.model or request.model})


settings = LlmGatewaySettings(on_llm_call=my_llm_hook)
client = build_llm_sync_client("openai", settings=settings)
```

If the hook raises an exception, it is logged and ignored so that
observability issues never break core LLM functionality.
