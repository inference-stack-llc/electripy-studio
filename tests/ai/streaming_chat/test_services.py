from __future__ import annotations

from electripy.ai.streaming_chat import StreamChunk, async_collect_text, collect_text


def test_collect_text_sync() -> None:
    chunks = [
        StreamChunk(index=0, delta_text="Hello"),
        StreamChunk(index=1, delta_text=" "),
        StreamChunk(index=2, delta_text="world", done=True),
    ]

    assert collect_text(chunks) == "Hello world"


async def test_collect_text_async() -> None:
    async def _chunks() -> object:
        yield StreamChunk(index=0, delta_text="A")
        yield StreamChunk(index=1, delta_text="B", done=True)

    assert await async_collect_text(_chunks()) == "AB"
