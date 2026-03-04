"""Services and orchestration for the RAG indexing and retrieval kit.

Purpose:
  - Provide deterministic chunking, embedding orchestration, indexing,
    and retrieval flows on top of the RAG ports.

Guarantees:
  - Business logic depends only on domain models, configuration, and
    ports, not on third-party clients.

Usage:
  Basic example::

    from electripy.ai.rag import (
        ChunkingConfig,
        DeterministicChunker,
        Document,
        EmbeddingGateway,
        IndexingService,
        Query,
        RetrievalService,
    )

    chunker = DeterministicChunker(ChunkingConfig())
    embedding_gateway = EmbeddingGateway(port=my_embedding_port)
    indexing = IndexingService(chunker=chunker, embedding_gateway=embedding_gateway, vector_store=my_store)
    retrieval = RetrievalService(embedding_gateway=embedding_gateway, vector_store=my_store)

    doc = Document(id="doc-1", source_uri="memory://", text="Hello world")
    indexing.index_document(doc)
    results = retrieval.retrieve(Query(text="hello", top_k=3))
"""

from __future__ import annotations

import random
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from electripy.core.logging import get_logger

from .config import ChunkingConfig, EmbeddingGatewaySettings
from .domain import (
    Chunk,
    Document,
    EmbeddingVector,
    Query,
    RetrievalResult,
)
from .errors import EmbeddingError, EmbeddingTransientError, IndexingError, RetrievalError
from .ports import ChunkerPort, EmbeddingPort, VectorStorePort

logger = get_logger(__name__)


@dataclass(slots=True)
class DeterministicChunker(ChunkerPort):
    """Deterministic character-based chunker.

    The chunker operates on character offsets with a fixed window size
    and overlap. It does not attempt semantic splitting; this can be
    added later by plugging in a different :class:`ChunkerPort`
    implementation.

    Args:
        config: Chunking configuration.
    """

    config: ChunkingConfig = field(default_factory=ChunkingConfig)

    def chunk(self, document: Document) -> list[Chunk]:  # type: ignore[override]
        """Chunk a document deterministically.

        Args:
            document: Document to chunk.

        Returns:
            List of :class:`Chunk` instances.
        """

        text = document.text
        length = len(text)
        if length == 0:
            return []

        chunks: list[Chunk] = []
        chunk_size = self.config.chunk_size_chars
        overlap = self.config.overlap_chars
        step = max(1, chunk_size - overlap)
        start = 0
        index = 0

        while start < length:
            end = min(start + chunk_size, length)
            chunk_text = text[start:end]
            chunk_metadata: Mapping[str, object] | None = document.metadata
            chunk_id = f"{document.id}:{index}"
            chunks.append(
                Chunk(
                    id=chunk_id,
                    document_id=document.id,
                    index=index,
                    text=chunk_text,
                    metadata=chunk_metadata,
                ),
            )
            index += 1
            if end >= length:
                break
            start += step

        return chunks


@dataclass(slots=True)
class EmbeddingGateway:
    """Batching and retry orchestration for embeddings.

    This service wraps an :class:`EmbeddingPort` implementation, enforcing
    batch limits and retrying transient failures using exponential
    backoff with jitter.

    Args:
        port: Embedding provider port.
        settings: Embedding gateway settings.
        sleep_fn: Optional sleep function used between retries (overridable
            for tests).
    """

    port: EmbeddingPort
    settings: EmbeddingGatewaySettings = field(default_factory=EmbeddingGatewaySettings)
    sleep_fn: callable = time.sleep

    def embed_texts(self, texts: Sequence[str]) -> list[EmbeddingVector]:
        """Embed a sequence of texts into dense vectors.

        Args:
            texts: Texts to embed.

        Returns:
            List of :class:`EmbeddingVector` objects, aligned with ``texts``.
        """

        if not texts:
            return []

        max_batch = self.settings.max_batch_size
        vectors: list[EmbeddingVector] = []
        for batch_start in range(0, len(texts), max_batch):
            batch = list(texts[batch_start : batch_start + max_batch])
            embedded_batch = self._embed_batch_with_retry(batch)
            for idx, vec in enumerate(embedded_batch):
                text_index = batch_start + idx
                vec_id = f"emb-{text_index}"
                vectors.append(EmbeddingVector(id=vec_id, vector=list(vec)))
        return vectors

    def _embed_batch_with_retry(self, batch: list[str]) -> list[list[float]]:
        attempt = 0
        while True:
            try:
                return self.port.embed_texts(batch)
            except EmbeddingTransientError as exc:
                if attempt >= self.settings.max_retries:
                    raise EmbeddingError("Embedding retries exhausted") from exc
                delay = self._calculate_delay_s(attempt)
                logger.info(
                    "Retrying transient embedding error",
                    extra={"attempt": attempt + 1, "delay_s": delay},
                )
                self.sleep_fn(delay)
                attempt += 1
                continue
            except EmbeddingError:
                raise
            except Exception as exc:  # noqa: BLE001
                raise EmbeddingError("Unexpected embedding failure") from exc

    def _calculate_delay_s(self, attempt_index: int) -> float:
        base = self.settings.base_delay_s * (2**attempt_index)
        base = min(base, self.settings.max_delay_s)
        if self.settings.jitter_ratio <= 0.0:
            return base
        low = max(0.0, base * (1.0 - self.settings.jitter_ratio))
        high = base * (1.0 + self.settings.jitter_ratio)
        return float(random.uniform(low, high))


@dataclass(slots=True)
class IndexingService:
    """Orchestrate ingest -> chunk -> embed -> upsert flows.

    The service keeps a simple in-memory cache of document content
    hashes to avoid unnecessary re-embedding within the lifetime of the
    process.

    Args:
        chunker: Deterministic chunker.
        embedding_gateway: Embedding gateway.
        vector_store: Vector store port implementation.
    """

    chunker: ChunkerPort
    embedding_gateway: EmbeddingGateway
    vector_store: VectorStorePort
    _document_hashes: dict[str, str] = field(default_factory=dict, init=False)

    def index_document(self, document: Document, *, force: bool = False) -> list[Chunk]:
        """Index a document into the vector store.

        Args:
            document: Document to index.
            force: If True, forces re-indexing even when the content hash has
                not changed.

        Returns:
            List of chunks that were (re-)indexed.
        """

        try:
            materialised = document.with_computed_hash()
            existing_hash = self._document_hashes.get(materialised.id)
            if not force and existing_hash == materialised.content_hash:
                logger.debug(
                    "Skipping re-index; content hash unchanged",
                    extra={"document_id": materialised.id},
                )
                return []

            chunks = self.chunker.chunk(materialised)
            if not chunks:
                self.vector_store.delete_by_document(materialised.id)
                self._document_hashes[materialised.id] = materialised.content_hash or ""
                return []

            vectors = self.embedding_gateway.embed_texts([chunk.text for chunk in chunks])
            if len(vectors) != len(chunks):
                raise IndexingError("Embedding gateway returned mismatched vector count")

            self.vector_store.upsert(chunks, [vec.vector for vec in vectors])
            self._document_hashes[materialised.id] = materialised.content_hash or ""
            return chunks
        except EmbeddingError as exc:
            raise IndexingError("Embedding failed during indexing") from exc
        except Exception as exc:  # noqa: BLE001
            raise IndexingError("Unexpected error during indexing") from exc

    def delete_document(self, document_id: str) -> None:
        """Delete a document and its chunks from the index.

        Args:
            document_id: Identifier of the document to delete.
        """

        self.vector_store.delete_by_document(document_id)
        self._document_hashes.pop(document_id, None)


@dataclass(slots=True)
class RetrievalService:
    """Service for retrieving relevant chunks for a query.

    Args:
        embedding_gateway: Embedding gateway used to embed queries.
        vector_store: Vector store port implementation.
    """

    embedding_gateway: EmbeddingGateway
    vector_store: VectorStorePort

    def retrieve(self, query: Query) -> list[RetrievalResult]:
        """Retrieve top-k relevant chunks for a query.

        Args:
            query: Query object containing text, k, and optional filters.

        Returns:
            List of retrieval results ordered by descending score.
        """

        try:
            vectors = self.embedding_gateway.embed_texts([query.text])
            if not vectors:
                return []
            query_vector = vectors[0].vector
            pairs = self.vector_store.query(query_vector, top_k=query.top_k, filters=query.filters)
            return [RetrievalResult(chunk=chunk, score=score) for chunk, score in pairs]
        except EmbeddingError as exc:
            raise RetrievalError("Embedding failed during retrieval") from exc
        except Exception as exc:  # noqa: BLE001
            raise RetrievalError("Unexpected error during retrieval") from exc


def load_document_from_text(
    id: str,
    text: str,
    *,
    source_uri: str = "memory://",
    metadata: Mapping[str, object] | None = None,
) -> Document:
    """Create a :class:`Document` from raw text.

    Args:
        id: Document identifier.
        text: Raw text content.
        source_uri: Logical source location; defaults to ``"memory://"``.
        metadata: Optional metadata mapping.

    Returns:
        Document instance.
    """

    return Document(id=id, source_uri=source_uri, text=text, metadata=metadata)


def load_document_from_bytes(
    id: str,
    data: bytes,
    *,
    source_uri: str,
    metadata: Mapping[str, object] | None = None,
    encoding: str = "utf-8",
) -> Document:
    """Create a :class:`Document` from bytes using a simple encoding strategy.

    The strategy attempts to decode using the provided encoding and, on
    ``UnicodeDecodeError``, falls back to UTF-8 with ``errors="replace"``.
    This behaviour is deterministic and avoids external dependencies.

    Args:
        id: Document identifier.
        data: Raw bytes.
        source_uri: Logical source location.
        metadata: Optional metadata mapping.
        encoding: Preferred text encoding (defaults to ``"utf-8"``).

    Returns:
        Document instance.
    """

    try:
        text = data.decode(encoding)
    except UnicodeDecodeError:
        text = data.decode("utf-8", errors="replace")
    return Document(id=id, source_uri=source_uri, text=text, metadata=metadata)


def load_document_from_file(
    path: str | Path,
    *,
    metadata: Mapping[str, object] | None = None,
    encoding: str = "utf-8",
) -> Document:
    """Load a text document from a file path.

    Only plain-text files are supported for the MVP. The file is read
    with the specified encoding, defaulting to UTF-8.

    Args:
        path: Filesystem path to a text file.
        metadata: Optional metadata mapping.
        encoding: Text encoding used when reading the file.

    Returns:
        Document instance.
    """

    p = Path(path)
    text = p.read_text(encoding=encoding)
    return Document(id=str(p), source_uri=p.as_uri(), text=text, metadata=metadata)

