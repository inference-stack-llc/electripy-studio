from __future__ import annotations

from electripy.ai.rag.domain import Chunk
from electripy.ai.rag.ports import EmbeddingPort
from electripy.ai.rag_eval_runner.adapters import FakeEmbeddingAdapter, InMemoryVectorStoreAdapter
from electripy.ai.rag_eval_runner.domain import QueryRecord, metric_key
from electripy.ai.rag_eval_runner.services import Evaluator


def test_fake_embedding_adapter_is_deterministic() -> None:
    adapter = FakeEmbeddingAdapter(dim=8)

    v1 = adapter.embed_texts(["hello", "world"])
    v2 = adapter.embed_texts(["hello", "world"])

    assert v1 == v2
    assert v1[0] != v1[1]


def test_in_memory_vector_store_is_deterministic_for_ties() -> None:
    store = InMemoryVectorStoreAdapter()

    c1 = Chunk(id="a:0", document_id="a", index=0, text="x")
    c2 = Chunk(id="b:0", document_id="b", index=0, text="y")

    # Use identical vectors so scores tie and ordering falls back to id.
    vector = [1.0, 0.0, 0.0]
    store.upsert([c1, c2], [vector, list(vector)])

    results = store.query(vector, top_k=2, filters=None)
    ids = [chunk.id for chunk, _ in results]

    assert ids == sorted(ids)


class DummyEmbeddingPort(EmbeddingPort):
    """Simple embedding port for tests.

    Assigns a unique basis vector per distinct text value. The same text
    always receives the same vector within the lifetime of the instance.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, list[float]] = {}

    def embed_texts(self, texts: list[str]) -> list[list[float]]:  # type: ignore[override]
        result: list[list[float]] = []
        for text in texts:
            if text not in self._vectors:
                index = len(self._vectors)
                vec = [0.0, 0.0, 0.0]
                vec[index % 3] = 1.0
                self._vectors[text] = vec
            result.append(list(self._vectors[text]))
        return result


def test_evaluator_metrics_include_mrr() -> None:
    store = InMemoryVectorStoreAdapter()
    embedder = DummyEmbeddingPort()

    # Two chunks, each aligned with its own query text.
    c1 = Chunk(id="c1", document_id="d1", index=0, text="q1")
    c2 = Chunk(id="c2", document_id="d2", index=0, text="q2")

    vectors = embedder.embed_texts([c1.text, c2.text])
    store.upsert([c1, c2], vectors)

    queries = [
        QueryRecord(id="q1", query="q1", relevant_ids=["c1"], metadata=None),
        QueryRecord(id="q2", query="q2", relevant_ids=["c2"], metadata=None),
    ]

    evaluator = Evaluator(vector_store=store, embedding_port=embedder)

    metrics, per_query = evaluator.evaluate(queries, top_k=1)

    hit_key = metric_key("hit_rate", 1)
    precision_key = metric_key("precision", 1)
    recall_key = metric_key("recall", 1)
    mrr_key = metric_key("mrr", 1)

    assert metrics[hit_key] == 1.0
    assert metrics[precision_key] == 1.0
    assert metrics[recall_key] == 1.0
    assert metrics[mrr_key] == 1.0

    assert len(per_query) == 2
    assert all(mrr_key in item.metrics for item in per_query)
