from __future__ import annotations

from electripy.ai.rag_quality import (
    RetrievalSnapshot,
    hit_rate_at_k,
    mrr_at_k,
    precision_at_k,
    recall_at_k,
    retrieval_drift,
)


def test_quality_metrics_values() -> None:
    retrieved = ["a", "b", "c"]
    relevant = ["b", "x"]

    assert hit_rate_at_k(retrieved, relevant, 2) == 1.0
    assert precision_at_k(retrieved, relevant, 2) == 0.5
    assert recall_at_k(retrieved, relevant, 2) == 0.5
    assert mrr_at_k(retrieved, relevant, 3) == 0.5


def test_retrieval_drift() -> None:
    baseline = RetrievalSnapshot(query_id="q1", retrieved_ids=["a", "b", "c"])
    candidate = RetrievalSnapshot(query_id="q1", retrieved_ids=["b", "d", "c"])

    drift = retrieval_drift(baseline, candidate, k=3)

    assert drift.query_id == "q1"
    assert drift.overlap_ratio == 2 / 3
    assert drift.baseline_only == ["a"]
    assert drift.candidate_only == ["d"]
