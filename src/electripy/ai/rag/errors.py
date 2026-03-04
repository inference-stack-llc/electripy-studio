"""Domain exceptions for the RAG indexing and retrieval kit.

Purpose:
  - Provide explicit, typed failure modes for the RAG component.
  - Shield callers from third-party database or embedding client errors.

Guarantees:
  - No raw third-party exceptions cross the public boundary.
  - Exceptions carry concise diagnostics without leaking sensitive
    payloads such as full document contents.

Usage:
  Basic example::

    from electripy.ai.rag import RagError

    try:
        indexing_service.index_document(doc)
    except RagError as exc:
        handle(exc)
"""

from __future__ import annotations

from dataclasses import dataclass

from electripy.core.errors import ElectriPyError


class RagError(ElectriPyError):
    """Base exception for all RAG-related failures."""


class EmbeddingError(RagError):
    """Raised when embedding operations fail.

    Typical causes include provider errors, invalid inputs, or retries
    exhausted at the embedding gateway layer.
    """


class EmbeddingTransientError(EmbeddingError):
    """Raised for transient embedding failures that may succeed on retry."""


class VectorStoreError(RagError):
    """Raised when vector store operations fail."""


class IndexingError(RagError):
    """Raised when indexing orchestration fails."""


class RetrievalError(RagError):
    """Raised when retrieval orchestration fails."""


@dataclass(slots=True)
class EvaluationError(RagError):
    """Raised when evaluation metrics cannot be computed.

    Attributes:
        message: Human-readable description of the failure.
    """

    message: str

    def __str__(self) -> str:  # pragma: no cover - trivial formatting
        return self.message
