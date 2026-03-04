"""Asynchronous LLM Gateway recipe using a fake provider."""

from __future__ import annotations

import asyncio

from electripy.ai.llm_gateway import (
    LlmGatewayAsyncClient,
    LlmGatewaySettings,
    LlmMessage,
    LlmRequest,
)

from .fake_provider import FakeAsyncProvider


async def amain() -> None:
    provider = FakeAsyncProvider()
    settings = LlmGatewaySettings()
    client = LlmGatewayAsyncClient(port=provider, settings=settings)

    request = LlmRequest(
        model="fake-model",
        messages=[LlmMessage.user("Hello from async recipe!")],
    )

    response = await client.complete(request)
    print(response.text)


def main() -> None:
    asyncio.run(amain())


if __name__ == "__main__":  # pragma: no cover
    main()
