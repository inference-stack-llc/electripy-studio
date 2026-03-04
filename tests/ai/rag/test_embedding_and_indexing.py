from __future__ import annotations

from collections.abc import Sequence

from electripy.ai.rag import (
    Chunk,
    DeterministicChunker,
    Document,
    EmbeddingGateway,
    EmbeddingGatewaySettings,
    EmbeddingPort,
    IndexingService,
    Query,
    RetrievalResult,
    RetrievalService,
)
from electripy.ai.rag.errors import EmbeddingTransientError


class FakeEmbeddingPort(EmbeddingPort):
    def __init__(self) -> None:
        self.calls: list[Sequence[str]] = []
        self.fail_first = False
        self._called = False

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:  # type: ignore[override]
        self.calls.append(list(texts))
        if self.fail_first and not self._called:
            self._called = True
            raise EmbeddingTransientError("temporary")
        return [[float(len(t))] for t in texts]


class FakeVectorStore:
    def __init__(self) -> None:
        self.vectors: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        for chunk, vector in zip(chunks, vectors, strict=True):
            self.vectors[chunk.id] = (chunk, list(vector))

    def query(self, vector: Sequence[float], *, top_k: int, filters: dict | None = None) -> list[tuple[Chunk, float]]:
        # Simple similarity: negative absolute difference in vector length.
        items = list(self.vectors.values())
        items.sort(key=lambda item: -abs(len(item[0].text) - int(vector[0])))
        results: list[tuple[Chunk, float]] = []
        for chunk, stored_vector in items[:top_k]:
            results.append((chunk, float(len(chunk.text))))
        return results

    def delete_by_document(self, document_id: str) -> None:
        keys = [k for k, (chunk, _) in self.vectors.items() if chunk.document_id == document_id]
        for key in keys:
            self.vectors.pop(key, None)


def test_embedding_gateway_batches_and_retries() -> None:
    port = FakeEmbeddingPort()
    port.fail_first = True
    settings = EmbeddingGatewaySettings(max_batch_size=2, max_retries=1, base_delay_s=0.0, max_delay_s=0.001)
    gateway = EmbeddingGateway(port=port, settings=settings, sleep_fn=lambda _s: None)
    vectors = gateway.embed_texts(["a", "bb", "ccc"])
    assert [v.vector for v in vectors] == [[1.0], [2.0], [3.0]]
    # Two batches expected: ["a", "bb"] and ["ccc"].
    assert len(port.calls) == 3  # initial + retry for first batch, then second batch


def test_index_and_retrieve_round_trip() -> None:
    chunker = DeterministicChunker()
    port = FakeEmbeddingPort()
    settings = EmbeddingGatewaySettings(max_batch_size=10)
    gateway = EmbeddingGateway(port=port, settings=settings, sleep_fn=lambda _s: None)
    store = FakeVectorStore()
    indexing = IndexingService(chunker=chunker, embedding_gateway=gateway, vector_store=store)
    retrieval = RetrievalService(embedding_gateway=gateway, vector_store=store)

    doc = Document(id="doc-1", source_uri="memory://", text="hello world")
    chunks = indexing.index_document(doc)
    assert chunks, "expected at least one chunk to be indexed"

    query = Query(text="hello", top_k=5)
    results = retrieval.retrieve(query)
    assert isinstance(results[0], RetrievalResult)
    assert results[0].chunk.document_id == "doc-1"

