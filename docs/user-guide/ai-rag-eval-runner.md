# RAG Evaluation Runner

The **RAG Evaluation Runner** provides a small, framework-agnostic
pipeline for benchmarking retrieval quality across different chunking
and embedding configurations using simple JSONL datasets.

It is designed for Python power users who want to:

- Point the runner at a corpus and queries file.
- Configure chunking and embedding variants.
- Run matrix experiments and collect metrics.
- Gate CI on minimum retrieval quality.

## Dataset format

Both corpus and queries are JSONL files. Blank lines and lines starting
with `#` are ignored.

### Corpus JSONL

Each line is a JSON object with the following fields:

- `id` (str, required)
- `text` (str, required)
- `source_uri` (str, optional)
- `metadata` (object, optional)

Example:

```json
{"id": "doc-1", "text": "Hello world", "source_uri": "memory://", "metadata": {"topic": "greeting"}}
```

### Queries JSONL

Each line is a JSON object with the following fields:

- `id` (str, required)
- `query` (str, required)
- `relevant_ids` (list[str], required) – chunk ids considered relevant
- `metadata` (object, optional)

Example:

```json
{"id": "q1", "query": "hello", "relevant_ids": ["doc-1:0"]}
```

## Experiment matrix

An evaluation run describes a matrix of experiments:

- `chunk_variants` – different chunking configurations.
- `embedder_variants` – logical embedders (for example `"fake"`,
  `"openai"`).
- `top_k_values` – list of cut-off ranks.

The runner expands these into experiments:

```text
experiments = chunk_variants × embedder_variants × top_k_values
```

Each experiment produces aggregate metrics per `k` and, optionally, a
per-query breakdown.

## Metrics

For each `k`, the runner computes:

- **Hit rate@k** – fraction of queries with at least one relevant chunk
  in the top-`k` results.
- **Precision@k** – macro-averaged precision.
- **Recall@k** – macro-averaged recall.
- **MRR@k** – mean reciprocal rank.

Metrics are computed using deterministic utilities from
`electripy.ai.rag.evaluation` plus a small local MRR implementation.

## CLI usage

A Typer-based CLI command is exposed as:

```bash
electripy rag eval --corpus corpus.jsonl --queries queries.jsonl \
  --top-k 3,5,10 --chunk-size 500 --chunk-overlap 100 --embedder fake \
  --report-json out.json --report-csv out.csv
```

Key options:

- `--corpus PATH` – corpus JSONL file.
- `--queries PATH` – queries JSONL file.
- `--top-k 3,5,10` – comma-separated list of cut-offs.
- `--chunk-size` / `--chunk-overlap` – basic chunking config.
- `--chunker-config PATH` – optional JSON file for advanced chunking
  config; takes precedence over `--chunk-size` / `--chunk-overlap`.
- `--embedder` – one or more embedders (for example `"fake"`),
  optionally as a comma-separated list.
- `--report-json` / `--report-csv` – report output paths.
- `--fail-under` – thresholds such as `hit_rate@5=0.85` for CI gating.

## Determinism and reproducibility

- Fake embeddings are deterministic functions of the input text.
- The in-memory vector store uses cosine similarity with deterministic
  tie-breaking on chunk id.
- Experiments are expanded in a stable order.
- Experiment ids are computed as SHA-256 hashes of the configuration.

## Extensibility

To plug in a custom chunker or embedder, implement the existing RAG
ports (`ChunkerPort`, `EmbeddingPort`, `VectorStorePort`) and wire them
into your own orchestration, or extend the helpers in
`electripy.ai.rag_eval_runner.services`.

See also the component-level README at
`src/electripy/ai/rag_eval_runner/README.md`.
