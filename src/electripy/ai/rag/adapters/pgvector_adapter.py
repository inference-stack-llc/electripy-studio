"""PostgreSQL + pgvector adapter for the RAG kit.

Purpose:
    Implement :class:`VectorStorePort` using PostgreSQL with the
    ``pgvector`` extension.

Design:
    This adapter is intentionally minimal and focuses on a single
    schema comprising ``documents``, ``chunks``, and ``embeddings``
    tables.
    The adapter does not manage connections itself; callers provide a
    connection factory returning psycopg connections.

Notes:
    The :mod:`psycopg` package is imported lazily inside methods to
    avoid hard runtime dependencies for environments that do not use
    this adapter (for example, during unit tests where a fake
    ``VectorStorePort`` is used).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..domain import Chunk
from ..errors import VectorStoreError
from ..ports import VectorStorePort


ConnectionFactory = Callable[[], Any]


DOCUMENTS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_documents (
    id TEXT PRIMARY KEY,
    source_uri TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    metadata JSONB
);
"""


CHUNKS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_chunks (
    id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL REFERENCES rag_documents (id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    metadata JSONB,
    chunk_hash TEXT NOT NULL
);
"""


EMBEDDINGS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS rag_embeddings (
    chunk_id TEXT PRIMARY KEY REFERENCES rag_chunks (id) ON DELETE CASCADE,
    embedding vector NOT NULL
);
"""


class PgVectorAdapter(VectorStorePort):
    """PostgreSQL + pgvector implementation of :class:`VectorStorePort`.

    Args:
        connection_factory: Callable returning a psycopg connection.
        dimension: Dimensionality of the embeddings stored in ``embedding``.
    """

    def __init__(self, connection_factory: ConnectionFactory, *, dimension: int) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        self._connection_factory = connection_factory
        self._dimension = dimension
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        conn = self._connection_factory()
        try:
            with conn:
                with conn.cursor() as cur:  # type: ignore[call-arg]
                    cur.execute(DOCUMENTS_TABLE_SQL)
                    cur.execute(CHUNKS_TABLE_SQL)
                    cur.execute(EMBEDDINGS_TABLE_SQL)
        finally:
            conn.close()

    def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
        if len(chunks) != len(vectors):
            raise VectorStoreError("chunks and vectors length mismatch")
        if not chunks:
            return

        conn = self._connection_factory()
        try:
            with conn:
                with conn.cursor() as cur:  # type: ignore[call-arg]
                    for chunk, vector in zip(chunks, vectors, strict=True):
                        if len(vector) != self._dimension:
                            raise VectorStoreError("embedding dimension mismatch")
                        cur.execute(
                            """
INSERT INTO rag_documents (id, source_uri, content_hash, metadata)
VALUES (%s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    source_uri = EXCLUDED.source_uri,
    content_hash = EXCLUDED.content_hash,
    metadata = EXCLUDED.metadata
""",
                            (
                                chunk.document_id,
                                "",  # source_uri is not tracked at this layer
                                chunk.chunk_hash,
                                chunk.metadata,
                            ),
                        )
                        cur.execute(
                            """
INSERT INTO rag_chunks (id, document_id, chunk_index, text, metadata, chunk_hash)
VALUES (%s, %s, %s, %s, %s, %s)
ON CONFLICT (id) DO UPDATE SET
    document_id = EXCLUDED.document_id,
    chunk_index = EXCLUDED.chunk_index,
    text = EXCLUDED.text,
    metadata = EXCLUDED.metadata,
    chunk_hash = EXCLUDED.chunk_hash
""",
                            (
                                chunk.id,
                                chunk.document_id,
                                chunk.index,
                                chunk.text,
                                chunk.metadata,
                                chunk.chunk_hash,
                            ),
                        )
                        cur.execute(
                            """
INSERT INTO rag_embeddings (chunk_id, embedding)
VALUES (%s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    embedding = EXCLUDED.embedding
""",
                            (chunk.id, vector),
                        )
        finally:
            conn.close()

    def query(
        self,
        vector: Sequence[float],
        *,
        top_k: int,
        filters: Mapping[str, object] | None = None,
    ) -> list[tuple[Chunk, float]]:
        if len(vector) != self._dimension:
            raise VectorStoreError("embedding dimension mismatch for query")
        if top_k <= 0:
            return []

        conn = self._connection_factory()
        try:
            with conn:
                with conn.cursor() as cur:  # type: ignore[call-arg]
                    conditions = ["e.chunk_id = c.id", "c.document_id = d.id"]
                    params: list[object] = [vector, top_k]
                    if filters:
                        conditions.append("d.metadata @> %s")
                        params.insert(1, filters)

                    where_clause = " AND ".join(conditions)
                    query_sql = f"""
SELECT c.id,
       c.document_id,
       c.chunk_index,
       c.text,
       c.metadata,
       c.chunk_hash,
       1 - (e.embedding <-> %s) AS score
FROM rag_embeddings e
JOIN rag_chunks c ON e.chunk_id = c.id
JOIN rag_documents d ON c.document_id = d.id
WHERE {where_clause}
ORDER BY e.embedding <-> %s
LIMIT %s
"""

                    cur.execute(query_sql, params)
                    rows = list(cur.fetchall())
                    results: list[tuple[Chunk, float]] = []
                    for row in rows:
                        chunk = Chunk(
                            id=row[0],
                            document_id=row[1],
                            index=int(row[2]),
                            text=row[3],
                            metadata=row[4],
                        )
                        score = float(row[6])
                        results.append((chunk, score))
                    return results
        finally:
            conn.close()

    def delete_by_document(self, document_id: str) -> None:
        conn = self._connection_factory()
        try:
            with conn:
                with conn.cursor() as cur:  # type: ignore[call-arg]
                    cur.execute("DELETE FROM rag_documents WHERE id = %s", (document_id,))
        finally:
            conn.close()
