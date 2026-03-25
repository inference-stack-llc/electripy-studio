"""Domain models for retrieval quality and drift assessment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class RetrievalSnapshot:
    """Top-k retrieval ids for a query under one configuration."""

    query_id: str
    retrieved_ids: list[str]


@dataclass(slots=True)
class DriftComparison:
    """Summary of retrieval drift between baseline and candidate."""

    query_id: str
    overlap_ratio: float
    baseline_only: list[str]
    candidate_only: list[str]
