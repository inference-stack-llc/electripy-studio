"""Metric and drift services for RAG quality monitoring."""

from __future__ import annotations

from .domain import DriftComparison, RetrievalSnapshot


def hit_rate_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Return 1.0 if any relevant item appears in top-k, else 0.0."""

    if k <= 0:
        raise ValueError("k must be positive")
    top = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return 1.0 if top & relevant else 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Compute precision@k for one query."""

    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved_ids[:k]
    if not top:
        return 0.0
    relevant = set(relevant_ids)
    hits = sum(1 for item in top if item in relevant)
    return hits / float(len(top))


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Compute recall@k for one query."""

    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant_ids:
        return 0.0
    top = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top & relevant) / float(len(relevant))


def mrr_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """Compute reciprocal rank of first relevant item within top-k."""

    if k <= 0:
        raise ValueError("k must be positive")
    relevant = set(relevant_ids)
    for idx, item in enumerate(retrieved_ids[:k], start=1):
        if item in relevant:
            return 1.0 / float(idx)
    return 0.0


def retrieval_drift(
    baseline: RetrievalSnapshot,
    candidate: RetrievalSnapshot,
    *,
    k: int,
) -> DriftComparison:
    """Compare two retrieval snapshots and return overlap drift summary."""

    if baseline.query_id != candidate.query_id:
        raise ValueError("query_id must match for drift comparison")
    if k <= 0:
        raise ValueError("k must be positive")

    b = baseline.retrieved_ids[:k]
    c = candidate.retrieved_ids[:k]
    set_b = set(b)
    set_c = set(c)
    overlap_ratio = len(set_b & set_c) / float(k)

    return DriftComparison(
        query_id=baseline.query_id,
        overlap_ratio=overlap_ratio,
        baseline_only=sorted(set_b - set_c),
        candidate_only=sorted(set_c - set_b),
    )
