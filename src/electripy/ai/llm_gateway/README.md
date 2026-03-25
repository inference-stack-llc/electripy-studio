# LLM Gateway

The LLM Gateway is a provider-agnostic, production-grade client for calling Large
Language Models (LLMs) in a consistent way.

It standardizes:

- Request/response modeling (`LlmRequest`, `LlmResponse`)
- Reliability behavior (retries, rate limits, transient failures)
- Structured JSON output with validation and repair
- Token budgeting and safety seams (redaction + prompt guard)
- Optional request/response policy hooks for pre/postflight enforcement

It currently ships with an OpenAI adapter and is designed to support additional
providers (Anthropic, Azure OpenAI, OpenRouter, etc.) via adapters.

## Provider selection (OpenAI, OpenRouter, Copilot, Grok, ...)

For most applications you should not construct adapters directly. Instead,
use the factory helpers:

- `build_llm_sync_client(provider: str, *, settings: LlmGatewaySettings | None = None, **kwargs)`
- `build_llm_async_client(provider: str, *, settings: LlmGatewaySettings | None = None, **kwargs)`

Supported provider names (case-insensitive):

- `"openai"` – Uses `OpenAiSyncAdapter` / `OpenAiAsyncAdapter`.
- `"http-json"`, `"openrouter"`, `"copilot"`, `"grok"`, `"claude-http"` – Use
  `HttpJsonChatSyncAdapter` / `HttpJsonChatAsyncAdapter` with an OpenAI-style
  HTTP+JSON chat API via `httpx`.

### Example: OpenAI via factory

```python
from electripy.ai.llm_gateway import (
  LlmMessage,
  LlmRequest,
  build_llm_sync_client,
)

client = build_llm_sync_client("openai")
response = client.complete(
  LlmRequest(
    model="gpt-4o-mini",
    messages=[LlmMessage.user("Hello from factory!")],
  )
)
print(response.text)
```

## Request/response policy hooks

`LlmGatewaySettings` includes two hook seams for deterministic policy
enforcement:

- `request_hook(request) -> request`
- `response_hook(request, response) -> response`

These hooks run inside the gateway execution path and can sanitize or
block content. Blocking hooks should raise `PolicyViolationError`.

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

### Example: OpenRouter / Copilot / Grok via HTTP JSON

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

response = client.complete(
  LlmRequest(
    model="openai/gpt-4o-mini",
    messages=[LlmMessage.user("Hello via OpenRouter!")],
  )
)
print(response.text)
```

## Ports & Adapters (provider swap concept)

The gateway defines:

- `SyncLlmPort` / `AsyncLlmPort`: minimal Protocols for sync/async LLM calls.
- `OpenAiSyncAdapter` / `OpenAiAsyncAdapter`: concrete adapters implementing these
  ports using the OpenAI SDK.
- `LlmGatewaySyncClient` / `LlmGatewayAsyncClient`: orchestration services that
  depend only on the ports.

To swap providers, implement the port Protocols for your provider and inject the
adapter into the gateway client.

## Reliability behavior

The gateway uses `RetryPolicy` and `LlmGatewaySettings` to control:

- `max_attempts`: maximum attempts (including the first).
- `initial_backoff_seconds` / `max_backoff_seconds`: exponential backoff configuration.
- `total_timeout_seconds`: upper bound on wall-clock time across all attempts.

Behavior:

- `RateLimitedError` (for example, HTTP 429) is retried, honoring `Retry-After`
  if available.
- `TransientLlmError` (5xx-like conditions) is retried with exponential backoff.
- Non-transient `LlmGatewayError` is not retried.
- If attempts or total time are exhausted, `RetryExhaustedError` is raised.

This provides a stable, documented trade-off between robustness and latency.

## Structured output mode

Structured output is enabled by passing a `StructuredOutputSpec` to
`complete(...)`:

- The gateway prepends a system message instructing the model to return only
  JSON matching the schema.
- The raw text is parsed as JSON.
- The parsed JSON is validated against the `StructuredOutputSpec` (required
  keys + value types).
- If validation fails, the gateway performs **one repair attempt**:
  - It calls the model again with a "repair" instruction and the previous invalid
    JSON.
- If validation still fails, a `StructuredOutputError` is raised.

Validation is intentionally simple and schema-like (object + primitive types) to
avoid heavy dependencies.

## Token budgeting

The gateway enforces a simple, deterministic token budget:

- It counts total characters across all prompt messages.
- It compares this number against:
  - `LlmRequest.max_input_chars` if set, otherwise
  - `LlmGatewaySettings.default_max_input_chars`.
- If the budget is exceeded, a `TokenBudgetExceededError` is raised **before**
  calling the provider.

This avoids silently truncating prompts and makes failures predictable. If you
prefer truncation, you can implement your own pre-processor before calling the
gateway.

`max_output_tokens` is passed through to the provider via the adapter when
supported.

## Safety seams (redaction + prompt guard)

The gateway exposes two safety hooks:

- `RedactorPort`:
  - `redact(text: str) -> str` to scrub PII/PHI.
  - Used by the gateway to safely log prompt/response summaries when
    `enable_safe_logging` is `True`.
  - Example implementation: `SimpleRedactor`.

- `PromptGuardPort`:
  - `assess(messages: Sequence[LlmMessage]) -> GuardResult`.
  - If `allowed` is `False`, the gateway raises `PromptRejectedError` and does
    not call the provider.
  - Example implementation: `HeuristicPromptGuard`, which scans for suspicious
    phrases (for example "ignore previous instructions", "exfiltrate").

Defaults:

- Safety hooks are **off** by default.
- Safe logging must be explicitly enabled via `LlmGatewaySettings` and a
  `RedactorPort`.

## Basic usage example (plain text)

```python
from electripy.ai.llm_gateway import (
    LlmGatewaySyncClient,
    LlmMessage,
    LlmRequest,
    OpenAiSyncAdapter,
)

adapter = OpenAiSyncAdapter()
client = LlmGatewaySyncClient(port=adapter)

request = LlmRequest(
    model="gpt-4o-mini",
    messages=[LlmMessage.user("Tell me a joke about electrons.")],
)

response = client.complete(request)
print(response.text)
```

## Advanced usage example (structured output + safety)

```python
from electripy.ai.llm_gateway import (
    HeuristicPromptGuard,
    LlmGatewaySyncClient,
    LlmGatewaySettings,
    LlmMessage,
    LlmRequest,
    OpenAiSyncAdapter,
    SimpleRedactor,
    StructuredOutputSpec,
)

adapter = OpenAiSyncAdapter()

settings = LlmGatewaySettings(
    default_model="gpt-4o-mini",
    enable_safe_logging=True,
    redactor=SimpleRedactor(),
    prompt_guard=HeuristicPromptGuard(),
)

client = LlmGatewaySyncClient(port=adapter, settings=settings)

spec = StructuredOutputSpec(
    name="WeatherReport",
    field_types={
        "location": str,
        "temperature_c": float,
        "condition": str,
    },
    description="Current weather conditions for a single location.",
)

request = LlmRequest(
    model="gpt-4o-mini",
    messages=[LlmMessage.user("Provide a JSON weather report for Berlin.")],
    max_output_tokens=256,
)

response = client.complete(request, structured_spec=spec)
assert response.raw_json is not None
print(response.raw_json["location"])
print(response.raw_json["temperature_c"])
```

## Swap guide: adding a new provider adapter

To add a new provider (for example Anthropic):

1. Implement the sync port::

    from electripy.ai.llm_gateway import LlmRequest, LlmResponse, SyncLlmPort
    from electripy.ai.llm_gateway.errors import RateLimitedError, TransientLlmError

    class AnthropicSyncAdapter(SyncLlmPort):
        def __init__(self, ...):
            self._client = ...

        def complete(self, request: LlmRequest, *, timeout: float | None = None) -> LlmResponse:
            try:
                # Map LlmRequest -> provider request
                ...
                # Map provider response -> LlmResponse
                return LlmResponse(...)
            except SomeProviderRateLimit as exc:
                raise RateLimitedError(
                    message="rate limited",
                    status_code=429,
                    retry_after_seconds=...,  # from provider
                ) from None
            except SomeProviderTransient as exc:
                raise TransientLlmError(message="temporary", status_code=503) from None
            except Exception as exc:  # noqa: BLE001
                raise TransientLlmError(message=str(exc)) from None

2. Implement the async port (optional but recommended) using `AsyncLlmPort`.

3. Inject your adapter into the gateway::

    adapter = AnthropicSyncAdapter(...)
    client = LlmGatewaySyncClient(port=adapter)

No changes are required in `LlmGatewaySyncClient` or `LlmGatewayAsyncClient`:
those classes only depend on the port Protocols, not on specific providers.

### Example: Claude-style HTTP adapter using `httpx`

Some providers (for example Anthropic Claude or custom gateways) expose a
JSON API that is *not* exactly OpenAI compatible. In that case you can
subclass the generic HTTP JSON adapter and override the payload/response
mapping only, keeping retries and safety behavior in the gateway layer.

```python
from typing import Any, Mapping

from electripy.ai.llm_gateway import HttpJsonChatSyncAdapter, LlmRequest, LlmResponse


class ClaudeHttpSyncAdapter(HttpJsonChatSyncAdapter):
  """Example adapter for a Claude-style HTTP JSON API.

  This sketch assumes the provider exposes an endpoint that accepts
  ``messages`` and returns a top-level JSON object with an ``output``
  field containing the assistant text. Adjust field names to your
  actual provider.
  """

  def _build_payload(self, request: LlmRequest) -> dict[str, Any]:
    return {
      "model": request.model,
      "messages": [
        {"role": message.role.value, "content": message.content}
        for message in request.messages
      ],
      "max_tokens": request.max_output_tokens,
      "temperature": request.temperature,
    }

  def _parse_response_json(self, data: Mapping[str, Any]) -> LlmResponse:
    # Many providers return something like {"output": "..."} or
    # {"content": [{"text": "..."}, ...]}. Adapt to your schema.
    text = str(data.get("output") or "")
    return LlmResponse(text=text, raw_json=data)
```

You can then wire this adapter into the gateway directly::

  adapter = ClaudeHttpSyncAdapter(base_url="https://api.claude.example")
  client = LlmGatewaySyncClient(port=adapter)

or add another branch in `build_llm_sync_client` / `build_llm_async_client`
for a dedicated provider name (for example `"claude"`).

## Public API surface

The primary symbols intended for external use are:

- Domain models:
  - `LlmRole`
  - `LlmMessage`
  - `LlmRequest`
  - `LlmResponse`
  - `StructuredOutputSpec`
- Ports / safety hooks:
  - `SyncLlmPort`, `AsyncLlmPort`
  - `RedactorPort`, `PromptGuardPort`, `GuardResult`
- Configuration:
  - `RetryPolicy`
  - `LlmGatewaySettings`
- Services:
  - `LlmGatewaySyncClient`
  - `LlmGatewayAsyncClient`
- Provider factories:
  - `build_llm_sync_client`
  - `build_llm_async_client`
- Adapters:
  - `OpenAiSyncAdapter`, `OpenAiAsyncAdapter`
  - `HttpJsonChatSyncAdapter`, `HttpJsonChatAsyncAdapter`
  - `SimpleRedactor`, `HeuristicPromptGuard`
- Errors:
  - `LlmGatewayError` (base)
  - `RateLimitedError`
  - `RetryExhaustedError`
  - `StructuredOutputError`
  - `TokenBudgetExceededError`
  - `PromptRejectedError`

All of these are exported from `electripy.ai.llm_gateway` and considered
part of the stable public surface for this component.
