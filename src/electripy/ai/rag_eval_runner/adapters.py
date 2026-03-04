"""Adapters and fakes for the RAG evaluation runner.

This module provides:

- ``FakeEmbeddingAdapter`` – deterministic, stateless embeddings derived
  from text hashing, suitable for tests and offline runs.
- ``InMemoryVectorStoreAdapter`` – simple in-memory vector store using
  cosine similarity with deterministic tie-breaking.

Both adapters implement the ports defined in :mod:`electripy.ai.rag` and
are intentionally minimal to keep dependencies small and behaviour
predictable.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence

from electripy.ai.rag.domain import Chunk
from electripy.ai.rag.ports import EmbeddingPort, VectorStorePort


class FakeEmbeddingAdapter(EmbeddingPort):
    """Deterministic embedding adapter based on SHA-256 hashing.

    The adapter produces fixed-size embedding vectors whose components
    are derived from the SHA-256 digest of the input text. The mapping
    is purely functional and does not involve any randomness, making it
    suitable for reproducible tests.

    Example:
        >>> adapter = FakeEmbeddingAdapter()
        >>> vectors = adapter.embed_texts(["hello", "world"])
        >>> len(vectors) == 2
        True
    """

    def __init__(self, *, dim: int = 16) -> None:
        if dim <= 0:
            raise ValueError("dim must be positive")
        self._dim = dim

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        return [self._embed_single(text) for text in texts]

    def _embed_single(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        # Use bytes from the digest to populate the vector deterministically.
        values: list[float] = []
        for i in range(self._dim):
            # Wrap around the digest if needed.
            b = digest[i % len(digest)]
            # Map byte to [-0.5, 0.5] and then scale.
            values.append((float(b) / 255.0) - 0.5)
        # L2-normalise to keep cosine similarity well-behaved.
        norm = math.sqrt(sum(v * v for v in values)) or 1.0
        return [v / norm for v in values]


class InMemoryVectorStoreAdapter(VectorStorePort):
    """In-memory vector store implementing :class:`VectorStorePort`.

    Notes:
        - Stores vectors in process memory only; suitable for tests and
          local evaluation runs.
        - Uses cosine similarity for ranking and breaks ties
          deterministically by chunk id.
    """

    def __init__(self) -> None:
        self._store: dict[str, tuple[Chunk, list[float]]] = {}

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        for chunk, vector in zip(chunks, vectors):
            self._store[chunk.id] = (chunk, list(vector))

    def query(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Mapping[str, object] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")
        if not self._store:
            return []

        # For now, filters are ignored; they are present to satisfy the
        # protocol and keep a future extension point.
        del filters

        norm_q = math.sqrt(sum(float(v) * float(v) for v in vector)) or 1.0
        scores: list[tuple[Chunk, float]] = []
        for chunk_id, (chunk, stored_vec) in self._store.items():
            dot = 0.0
            norm_v = 0.0
            for a, b in zip(vector, stored_vec):
                fa = float(a)
                fb = float(b)
                dot += fa * fb
                norm_v += fb * fb
            norm_v = math.sqrt(norm_v) or 1.0
            score = dot / (norm_q * norm_v)
            scores.append((chunk, score))

        # Deterministic ordering: sort by descending score, then chunk id.
        scores.sort(key=lambda item: (-item[1], item[0].id))
        return scores[:top_k]

    def delete_by_document(self, document_id: str) -> None:
        to_delete = [cid for cid, (chunk, _) in self._store.items() if chunk.document_id == document_id]
        for cid in to_delete:
            self._store.pop(cid, None)
