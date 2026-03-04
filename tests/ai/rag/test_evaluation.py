from __future__ import annotations

from electripy.ai.rag import (
    GroundTruthExample,
    RetrievalResult,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
)
from electripy.ai.rag.domain import Chunk


def _chunk(chunk_id: str) -> Chunk:
    return Chunk(id=chunk_id, document_id="d", index=0, text="t")


def test_hit_rate_precision_recall_at_k() -> None:
    truth = [
        GroundTruthExample(query_text="q1", relevant_chunk_ids=frozenset({"c1", "c2"})),
        GroundTruthExample(query_text="q2", relevant_chunk_ids=frozenset({"c3"})),
    ]

    results = {
        "q1": [
            RetrievalResult(chunk=_chunk("c1"), score=1.0),
            RetrievalResult(chunk=_chunk("cX"), score=0.5),
        ],
        "q2": [
            RetrievalResult(chunk=_chunk("cY"), score=0.9),
        ],
    }

    hit = hit_rate_at_k(results, ground_truth=truth, k=2)
    precision = precision_at_k(results, ground_truth=truth, k=2)
    recall = recall_at_k(results, ground_truth=truth, k=2)

    assert 0.0 <= hit <= 1.0
    assert 0.0 <= precision <= 1.0
    assert 0.0 <= recall <= 1.0
    # At least one query should have a hit.
    assert hit > 0.0
