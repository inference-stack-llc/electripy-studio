"""Synchronous LLM Gateway recipe using a fake provider."""

from __future__ import annotations

from electripy.ai.llm_gateway import (
    LlmGatewaySettings,
    LlmGatewaySyncClient,
    LlmMessage,
    LlmRequest,
)

from .fake_provider import FakeSyncProvider


def main() -> None:
    provider = FakeSyncProvider()
    settings = LlmGatewaySettings()
    client = LlmGatewaySyncClient(port=provider, settings=settings)

    request = LlmRequest(
        model="fake-model",
        messages=[LlmMessage.user("Hello from sync recipe!")],
    )

    response = client.complete(request)
    print(response.text)


if __name__ == "__main__":  # pragma: no cover
    main()
