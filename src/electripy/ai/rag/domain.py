"""Domain models for the RAG indexing and retrieval kit.

Purpose:
  - Represent documents, chunks, embeddings, queries, and retrieval
    results in a framework-agnostic way.
  - Provide stable hashing utilities for deterministic content and
    chunk identifiers.

Guarantees:
  - No third-party client or database types appear in this module.
  - Data models are fully typed and suitable for use with static
    type checkers.

Usage:
  Basic example::

    from electripy.ai.rag import Document, Query

    doc = Document(id="doc-1", source_uri="memory://", text="Hello world")
    query = Query(text="hello", top_k=3)
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final


def _normalise_whitespace(text: str) -> str:
	"""Return text with normalised newlines and collapsed spaces.

	The normalisation is deterministic and aims to make hashing stable
	across common formatting differences (Windows vs Unix newlines,
	multiple spaces or tabs).

	Args:
		text: Raw input text.

	Returns:
		Normalised text.
	"""

	# Normalise newlines.
	value = text.replace("\r\n", "\n").replace("\r", "\n")
	# Collapse runs of spaces and tabs to a single space.
	value = re.sub(r"[ \t]+", " ", value)
	# Strip trailing spaces on each line for determinism.
	value = "\n".join(part.rstrip() for part in value.splitlines())
	return value


def compute_content_hash(text: str, metadata: Mapping[str, object] | None = None) -> str:
	"""Compute a stable SHA-256 hash for document content.

	The hash is based on normalised text and an optional JSON
	representation of metadata with sorted keys.

	Args:
		text: Document text.
		metadata: Optional metadata mapping.

	Returns:
		Hex-encoded SHA-256 digest.
	"""

	normalised = _normalise_whitespace(text)
	hash_obj = hashlib.sha256()
	hash_obj.update(normalised.encode("utf-8"))
	if metadata is not None:
		metadata_json = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
		hash_obj.update(b"|")
		hash_obj.update(metadata_json.encode("utf-8"))
	return hash_obj.hexdigest()


def compute_chunk_hash(text: str, document_id: str, index: int) -> str:
	"""Compute a stable SHA-256 hash for a chunk.

	Args:
		text: Chunk text.
		document_id: Identifier of the parent document.
		index: Zero-based index of the chunk within the document.

	Returns:
		Hex-encoded SHA-256 digest.
	"""

	normalised = _normalise_whitespace(text)
	hash_obj = hashlib.sha256()
	hash_obj.update(document_id.encode("utf-8"))
	hash_obj.update(b"|")
	hash_obj.update(str(index).encode("utf-8"))
	hash_obj.update(b"|")
	hash_obj.update(normalised.encode("utf-8"))
	return hash_obj.hexdigest()


@dataclass(slots=True)
class Document:
	"""Ingested document to be indexed.

	Attributes:
		id: Stable document identifier (for example, a UUID).
		source_uri: Logical source location (filesystem path, URL, etc.).
		text: Normalised document text.
		metadata: Optional metadata associated with the document.
		content_hash: Stable hash of the document content and metadata.
	"""

	id: str
	source_uri: str
	text: str
	metadata: Mapping[str, object] | None = None
	content_hash: str | None = None

	def with_computed_hash(self) -> "Document":
		"""Return a shallow copy of the document with ``content_hash`` set.

		Returns:
			Document: Document instance with a computed content hash.
		"""

		return Document(
			id=self.id,
			source_uri=self.source_uri,
			text=self.text,
			metadata=self.metadata,
			content_hash=compute_content_hash(self.text, self.metadata),
		)


@dataclass(slots=True)
class Chunk:
	"""Deterministic document chunk.

	Attributes:
		id: Stable chunk identifier derived from document id, index, and
			chunk hash.
		document_id: Identifier of the parent document.
		index: Zero-based index of the chunk within the document.
		text: Chunk text.
		metadata: Optional metadata, typically inherited from the document
			and extended with chunk-local information.
		chunk_hash: Stable hash of the chunk.
	"""

	id: str
	document_id: str
	index: int
	text: str
	metadata: Mapping[str, object] | None = None
	chunk_hash: str = field(init=False)

	def __post_init__(self) -> None:
		self.chunk_hash = compute_chunk_hash(self.text, self.document_id, self.index)


@dataclass(slots=True)
class EmbeddingVector:
	"""Embedding vector associated with a text unit.

	Attributes:
		id: Identifier for the embedding (for example, chunk id).
		vector: Dense embedding values.
		metadata: Optional metadata (for example, source information).
	"""

	id: str
	vector: list[float]
	metadata: Mapping[str, object] | None = None


@dataclass(slots=True)
class Query:
	"""User query for retrieval.

	Attributes:
		text: Raw query text.
		top_k: Maximum number of chunks to retrieve.
		filters: Optional metadata filters passed to the vector store.
	"""

	text: str
	top_k: int = 5
	filters: Mapping[str, object] | None = None

	def __post_init__(self) -> None:
		if self.top_k <= 0:
			raise ValueError("top_k must be positive")


@dataclass(slots=True)
class RetrievalResult:
	"""Single retrieval result.

	Attributes:
		chunk: Retrieved chunk.
		score: Similarity score as returned by the vector store. Higher is
			better for inner-product style similarity.
	"""

	chunk: Chunk
	score: float


@dataclass(slots=True)
class GroundTruthExample:
	"""Ground-truth mapping from query text to relevant chunk ids.

	Attributes:
		query_text: Text of the query.
		relevant_chunk_ids: Set of chunk identifiers considered relevant.
	"""

	query_text: str
	relevant_chunk_ids: frozenset[str]


EMPTY_RESULTS: Final[list[RetrievalResult]] = []

