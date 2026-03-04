from __future__ import annotations

from electripy.ai.rag import ChunkingConfig, DeterministicChunker, Document, compute_content_hash


def test_compute_content_hash_is_stable_under_whitespace() -> None:
    text_a = "Hello\nworld"
    text_b = "Hello\r\nworld"
    hash_a = compute_content_hash(text_a, metadata={"k": "v"})
    hash_b = compute_content_hash(text_b, metadata={"k": "v"})
    assert hash_a == hash_b


def test_deterministic_chunker_produces_overlapping_chunks() -> None:
    config = ChunkingConfig(chunk_size_chars=10, overlap_chars=2)
    chunker = DeterministicChunker(config)
    doc = Document(id="d1", source_uri="memory://", text="abcdefghijklmnopqrstuvwxyz")
    chunks = chunker.chunk(doc)
    assert chunks, "expected at least one chunk"
    # Ensure chunk ids are stable and ordered.
    assert [c.id for c in chunks] == [f"d1:{i}" for i in range(len(chunks))]
    # Ensure overlap between consecutive chunks.
    for first, second in zip(chunks, chunks[1:], strict=False):
        assert first.text[-config.overlap_chars :] == second.text[: config.overlap_chars]

