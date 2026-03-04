"""Ports (Protocols) for the RAG indexing and retrieval kit.

Purpose:
  - Define minimal capabilities required from embedding providers,
    vector stores, and chunking engines.

Guarantees:
  - Business logic depends only on these Protocols.
  - Adapters are free to use any third-party libraries to implement
    them.

Usage:
  Basic example::

    class InMemoryVectorStore(VectorStorePort):
        def upsert(self, chunks, vectors) -> None:
            ...

        def query(self, vector, top_k, filters=None):
            ...

        def delete_by_document(self, document_id: str) -> None:
            ...
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from .domain import Chunk, Document


@runtime_checkable
class EmbeddingPort(Protocol):
	"""Port for embedding text sequences.

	Implementations should be pure functions from a list of texts to a
	list of embedding vectors of equal length.
	"""

	def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
		"""Embed a batch of texts.

		Args:
			texts: Sequence of input strings.

		Returns:
			List of dense vectors, one per input text.
		"""

		...


@runtime_checkable
class VectorStorePort(Protocol):
	"""Port for vector store operations used by the RAG kit.

	The protocol intentionally uses a small API surface to keep adapters
	simple and swappable.
	"""

	def upsert(self, chunks: Sequence[Chunk], vectors: Sequence[list[float]]) -> None:
		"""Insert or update embeddings for the given chunks.

		Args:
			chunks: Chunks to index.
			vectors: Embedding vectors aligned with ``chunks``.
		"""

		...

	def query(
		self,
		vector: Sequence[float],
		*,
		top_k: int,
		filters: Mapping[str, object] | None = None,
	) -> list[tuple[Chunk, float]]:
		"""Query the vector store for nearest neighbours.

		Args:
			vector: Query embedding vector.
			top_k: Maximum number of results to return.
			filters: Optional metadata filters.

		Returns:
			List of ``(Chunk, score)`` pairs.
		"""

		...

	def delete_by_document(self, document_id: str) -> None:
		"""Delete all entries associated with a document id.

		Args:
			document_id: Identifier of the document to remove.
		"""

		...



@runtime_checkable
class ChunkerPort(Protocol):
	"""Port for deterministic document chunking.

	Implementations must be deterministic: the same input document always
	produces the same sequence of chunks.
	"""

	def chunk(self, document: Document) -> list[Chunk]:
		"""Chunk a document into smaller, overlapping units.

		Args:
			document: Document-like object with ``id`` and ``text``
				attributes. Typed as ``object`` in the Protocol to avoid a
				tight coupling to the concrete :class:`Document` dataclass.

		Returns:
			List of :class:`Chunk` instances.
		"""

		...

