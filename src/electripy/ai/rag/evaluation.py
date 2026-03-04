"""Evaluation utilities for the RAG indexing and retrieval kit.

Purpose:
  - Provide simple, deterministic retrieval quality metrics such as
    hit-rate@k, precision@k, and recall@k.

Guarantees:
  - Metrics are pure functions with explicit types and no side
    effects.

Usage:
  Basic example::

    from electripy.ai.rag import GroundTruthExample, hit_rate_at_k

    gt = [GroundTruthExample(query_text="hello", relevant_chunk_ids=frozenset({"c1"}))]
    score = hit_rate_at_k({"hello": results}, ground_truth=gt, k=5)
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from .domain import GroundTruthExample, RetrievalResult


def _build_truth_index(ground_truth: Sequence[GroundTruthExample]) -> dict[str, frozenset[str]]:
	index: dict[str, frozenset[str]] = {}
	for example in ground_truth:
		index[example.query_text] = example.relevant_chunk_ids
	return index


def hit_rate_at_k(
	results_by_query: Mapping[str, Sequence[RetrievalResult]],
	*,
	ground_truth: Sequence[GroundTruthExample],
	k: int,
) -> float:
	"""Compute hit-rate@k over a set of queries.

	Hit-rate@k is the fraction of queries for which at least one relevant
	chunk appears in the top ``k`` results.

	Args:
		results_by_query: Mapping from query text to retrieval results.
		ground_truth: Ground-truth examples containing relevant chunk ids.
		k: Cut-off rank.

	Returns:
		Hit-rate@k in the range [0.0, 1.0]. Returns 0.0 when there are no
		ground-truth examples.
	"""

	if k <= 0:
		raise ValueError("k must be positive")
	truth_index = _build_truth_index(ground_truth)
	if not truth_index:
		return 0.0

	hits = 0
	for query_text, relevant_ids in truth_index.items():
		results = list(results_by_query.get(query_text, ()))[:k]
		result_ids = {r.chunk.id for r in results}
		if result_ids & relevant_ids:
			hits += 1

	return hits / len(truth_index)


def precision_at_k(
	results_by_query: Mapping[str, Sequence[RetrievalResult]],
	*,
	ground_truth: Sequence[GroundTruthExample],
	k: int,
) -> float:
	"""Compute macro-averaged precision@k.

	Precision@k for a single query is the fraction of retrieved chunks in
	the top ``k`` that are relevant.
	"""

	if k <= 0:
		raise ValueError("k must be positive")
	truth_index = _build_truth_index(ground_truth)
	if not truth_index:
		return 0.0

	total_precision = 0.0
	for query_text, relevant_ids in truth_index.items():
		results = list(results_by_query.get(query_text, ()))[:k]
		if not results:
			continue
		result_ids = {r.chunk.id for r in results}
		true_positives = len(result_ids & relevant_ids)
		precision = true_positives / len(results)
		total_precision += precision

	return total_precision / len(truth_index)


def recall_at_k(
	results_by_query: Mapping[str, Sequence[RetrievalResult]],
	*,
	ground_truth: Sequence[GroundTruthExample],
	k: int,
) -> float:
	"""Compute macro-averaged recall@k.

	Recall@k for a single query is the fraction of relevant chunks that
	appear in the top ``k`` results.
	"""

	if k <= 0:
		raise ValueError("k must be positive")
	truth_index = _build_truth_index(ground_truth)
	if not truth_index:
		return 0.0

	total_recall = 0.0
	for query_text, relevant_ids in truth_index.items():
		results = list(results_by_query.get(query_text, ()))[:k]
		if not relevant_ids:
			continue
		result_ids = {r.chunk.id for r in results}
		true_positives = len(result_ids & relevant_ids)
		recall = true_positives / len(relevant_ids)
		total_recall += recall

	return total_recall / len(truth_index)

