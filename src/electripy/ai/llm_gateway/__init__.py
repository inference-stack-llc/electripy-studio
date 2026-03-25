"""LLM Gateway public API.

Purpose:
  - Expose a provider-agnostic, production-grade gateway for calling LLMs.
  - Provide sync/async clients, domain models, ports, adapters, and errors.

Guarantees:
  - Business logic depends only on Protocol ports, not on provider SDKs.
  - Provider-specific code is isolated in adapters.

Usage:
  Basic example::

    from electripy.ai.llm_gateway import (
        LlmGatewaySyncClient,
        LlmMessage,
        LlmRequest,
        OpenAiSyncAdapter,
    )

    adapter = OpenAiSyncAdapter()
    client = LlmGatewaySyncClient(port=adapter)
    response = client.complete(
        LlmRequest(
            model="gpt-4o-mini",
            messages=[LlmMessage.user("Say hi")],
        )
    )
"""

from __future__ import annotations

from .adapters import (
    HeuristicPromptGuard,
    HttpJsonChatAsyncAdapter,
    HttpJsonChatSyncAdapter,
    OpenAiAsyncAdapter,
    OpenAiSyncAdapter,
    SimpleRedactor,
)
from .config import LlmCallHook, LlmGatewaySettings, LlmRequestHook, LlmResponseHook, RetryPolicy
from .domain import (
    LlmMessage,
    LlmRequest,
    LlmResponse,
    LlmRole,
    StructuredOutputSpec,
)
from .errors import (
    LlmGatewayError,
    PolicyViolationError,
    PromptRejectedError,
    RateLimitedError,
    RetryExhaustedError,
    StructuredOutputError,
    TokenBudgetExceededError,
)
from .ports import AsyncLlmPort, GuardResult, PromptGuardPort, RedactorPort, SyncLlmPort
from .providers import build_llm_async_client, build_llm_sync_client
from .services import LlmGatewayAsyncClient, LlmGatewaySyncClient

__all__ = [
    # Domain models
    "LlmRole",
    "LlmMessage",
    "LlmRequest",
    "LlmResponse",
    "StructuredOutputSpec",
    # Ports
    "SyncLlmPort",
    "AsyncLlmPort",
    "RedactorPort",
    "PromptGuardPort",
    "GuardResult",
    # Adapters
    "OpenAiSyncAdapter",
    "OpenAiAsyncAdapter",
    "HttpJsonChatSyncAdapter",
    "HttpJsonChatAsyncAdapter",
    "SimpleRedactor",
    "HeuristicPromptGuard",
    # Services
    "LlmGatewaySyncClient",
    "LlmGatewayAsyncClient",
    "build_llm_sync_client",
    "build_llm_async_client",
    # Config
    "RetryPolicy",
    "LlmGatewaySettings",
    "LlmCallHook",
    "LlmRequestHook",
    "LlmResponseHook",
    # Errors
    "LlmGatewayError",
    "RateLimitedError",
    "RetryExhaustedError",
    "StructuredOutputError",
    "TokenBudgetExceededError",
    "PromptRejectedError",
    "PolicyViolationError",
]
