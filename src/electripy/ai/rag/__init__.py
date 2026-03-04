"""Public API for the RAG indexing and retrieval kit.

Purpose:
  - Expose the main domain models, ports, adapters, services, and
    evaluation helpers for retrieval-augmented generation (RAG).

Guarantees:
  - All public types are defined in this package; no third-party
    client types leak across the boundary.
  - Components are fully typed and safe to use with static analysis
    tools such as mypy or pyright.

Usage:
  Basic example::

    from electripy.ai.rag import (
        DeterministicChunker,
        Document,
        EmbeddingGateway,
        IndexingService,
        Query,
        RetrievalService,
    )

    chunker = DeterministicChunker()
    embedding_gateway = EmbeddingGateway(port=my_embedding_port)
    indexing = IndexingService(chunker=chunker, embedding_gateway=embedding_gateway, vector_store=my_store)
    retrieval = RetrievalService(embedding_gateway=embedding_gateway, vector_store=my_store)

    doc = Document(id="doc-1", source_uri="memory://", text="Hello world")
    indexing.index_document(doc)
    results = retrieval.retrieve(Query(text="hello", top_k=3))
"""

from __future__ import annotations

from .config import ChunkingConfig, EmbeddingGatewaySettings
from .domain import (
    Chunk,
    Document,
    EmbeddingVector,
    Query,
    RetrievalResult,
    compute_content_hash,
)
from .errors import (
    EmbeddingError,
    IndexingError,
    RagError,
    RetrievalError,
    VectorStoreError,
)
from .evaluation import (
    GroundTruthExample,
    hit_rate_at_k,
    precision_at_k,
    recall_at_k,
)
from .ports import ChunkerPort, EmbeddingPort, VectorStorePort
from .services import (
    DeterministicChunker,
    EmbeddingGateway,
    IndexingService,
    RetrievalService,
)

__all__ = [
    "Chunk",
    "ChunkerPort",
    "ChunkingConfig",
    "DeterministicChunker",
    "Document",
    "compute_content_hash",
    "EmbeddingError",
    "EmbeddingGateway",
    "EmbeddingGatewaySettings",
    "EmbeddingPort",
    "EmbeddingVector",
    "GroundTruthExample",
    "IndexingError",
    "IndexingService",
    "Query",
    "RagError",
    "RetrievalError",
    "RetrievalResult",
    "RetrievalService",
    "VectorStoreError",
    "VectorStorePort",
    "hit_rate_at_k",
    "precision_at_k",
    "recall_at_k",
]

