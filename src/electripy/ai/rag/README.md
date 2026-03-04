# RAG Indexing & Retrieval Kit

This component provides a production-minded, framework-agnostic toolkit for
retrieval-augmented generation (RAG). It focuses on deterministic document
ingestion and chunking, batching and retrying embeddings, pluggable vector
stores, and simple retrieval evaluation metrics.

## What it is / why it exists

- Normalised domain models for documents, chunks, embeddings, queries, and
  retrieval results.
- Deterministic character-based chunking with configurable size and overlap.
- Embedding gateway that batches inputs and retries transient failures.
- Vector store abstraction with a PostgreSQL + pgvector adapter.
- Orchestration services for indexing and retrieval.
- Simple retrieval metrics (hit-rate@k, precision@k, recall@k).

Use this kit when you need a small, explicit RAG surface that is easy to
reason about and test locally without binding your application to a specific
framework.

## End-to-end flow

1. **Ingest**: Convert raw text/bytes/files into `Document` instances.
2. **Chunk**: Use `DeterministicChunker` (or another `ChunkerPort`)
	implementation to create overlapping `Chunk` objects.
3. **Embed**: Call `EmbeddingGateway` with an `EmbeddingPort` implementation
	(for example, `OpenAiEmbeddingAdapter`).
4. **Index**: Use `IndexingService` to upsert chunks and embeddings into a
	`VectorStorePort` implementation (for example, `PgVectorAdapter`).
5. **Retrieve**: Use `RetrievalService` to embed queries and obtain the top-k
	`RetrievalResult` objects.
6. **Evaluate**: Run metrics such as `hit_rate_at_k`, `precision_at_k`, and
	`recall_at_k` over ground-truth data.

## Deterministic chunk IDs & incremental indexing

- Document content and chunk hashes are computed using normalised text and
  JSON-serialised metadata, ensuring stability across minor formatting
  changes.
- `IndexingService` keeps an in-memory cache of `content_hash` values per
  document id and skips re-indexing when the hash has not changed (unless the
  caller sets `force=True`).
- For long-running processes or multi-instance deployments, this strategy can
  be extended by persisting hashes in your own store; the service interface is
  small on purpose.

## Default adapters and swapping

- **Embeddings**: `OpenAiEmbeddingAdapter` implements `EmbeddingPort` on top
  of the official OpenAI SDK.
- **Vector store**: `PgVectorAdapter` implements `VectorStorePort` using
  PostgreSQL and the `pgvector` extension.

Applications depend only on the ports and services; adapters live in
`electripy.ai.rag.adapters` and can be replaced without changing calling
code.

To add a new vector store (for example, Qdrant):

1. Implement `VectorStorePort` for your backend.
2. Use that adapter when constructing `IndexingService` and
	`RetrievalService`.

To swap embedding providers:

1. Implement `EmbeddingPort` for the new provider.
2. Use that port when constructing `EmbeddingGateway`.

## Usage examples

### Basic: index a few documents and query

```python
from electripy.ai.rag import (
	 ChunkingConfig,
	 DeterministicChunker,
	 Document,
	 EmbeddingGateway,
	 EmbeddingGatewaySettings,
	 IndexingService,
	 Query,
	 RetrievalService,
)
from electripy.ai.rag.adapters.openai_embeddings_adapter import OpenAiEmbeddingAdapter
from electripy.ai.rag.adapters.pgvector_adapter import PgVectorAdapter


def make_pg_connection():
	 import psycopg

	 return psycopg.connect("postgresql://user:pass@localhost/db")


embedding_port = OpenAiEmbeddingAdapter(model="text-embedding-3-small")
gateway = EmbeddingGateway(port=embedding_port, settings=EmbeddingGatewaySettings())
vector_store = PgVectorAdapter(connection_factory=make_pg_connection, dimension=1536)

chunker = DeterministicChunker(ChunkingConfig(chunk_size_chars=800, overlap_chars=200))
indexing = IndexingService(chunker=chunker, embedding_gateway=gateway, vector_store=vector_store)
retrieval = RetrievalService(embedding_gateway=gateway, vector_store=vector_store)

docs = [
	 Document(id="doc-1", source_uri="file://notes1", text="RAG systems combine search and generation."),
	 Document(id="doc-2", source_uri="file://notes2", text="Deterministic chunking simplifies caching."),
]

for doc in docs:
	 indexing.index_document(doc)

results = retrieval.retrieve(Query(text="What is RAG?", top_k=3))
for result in results:
	 print(result.score, result.chunk.document_id, result.chunk.text[:60])
```

### Advanced: evaluation with ground truth

```python
from electripy.ai.rag import (
	 GroundTruthExample,
	 RetrievalResult,
	 hit_rate_at_k,
	 precision_at_k,
	 recall_at_k,
)


# Assume you already have `results_by_query` populated with RetrievalResult
# instances keyed by query text.
ground_truth = [
	 GroundTruthExample(query_text="q1", relevant_chunk_ids=frozenset({"chunk-1", "chunk-2"})),
	 GroundTruthExample(query_text="q2", relevant_chunk_ids=frozenset({"chunk-5"})),
]

hit = hit_rate_at_k(results_by_query, ground_truth=ground_truth, k=5)
precision = precision_at_k(results_by_query, ground_truth=ground_truth, k=5)
recall = recall_at_k(results_by_query, ground_truth=ground_truth, k=5)

print({"hit@5": hit, "precision@5": precision, "recall@5": recall})
```

## Swap guide

### Add a Qdrant adapter

1. Create a new module, for example
	`electripy/ai/rag/adapters/qdrant_adapter.py`.
2. Implement `VectorStorePort`:

	- `upsert`: write chunk + embedding records into Qdrant.
	- `query`: perform a similarity search and return `(Chunk, score)`
	  pairs.
	- `delete_by_document`: delete all vectors whose stored metadata
	  matches the document id.

3. Use your adapter when constructing `IndexingService` and
	`RetrievalService`.

### Swap embedding provider

1. Implement `EmbeddingPort` for the new provider (for example,
	a local sentence-transformers model).
2. Map provider-specific errors into `EmbeddingError` or
	`EmbeddingTransientError` so the `EmbeddingGateway` can apply the
	same retry semantics.
3. Plug this adapter into `EmbeddingGateway` instead of
	`OpenAiEmbeddingAdapter`.

