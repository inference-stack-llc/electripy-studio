# Recipe: RAG Evaluation Runner

This recipe shows how to run the RAG Evaluation Runner end-to-end to
benchmark retrieval quality and gate CI based on minimum metrics.

## Scenario

You have:

- A **corpus** of documents in JSONL.
- A **queries** file with labeled relevant chunks.
- One or more retrieval configurations you want to compare.

You want to:

- Quickly run experiments across chunking/embedding variants.
- Generate JSON/CSV reports.
- Fail CI if metrics drop below agreed thresholds.

## Files

Example layout:

```text
project/
  data/
    corpus.jsonl
    queries.jsonl
  reports/
    rag_eval_report.json
    rag_eval_report.csv
```

## CLI usage

Run a basic evaluation using the built-in fake embedder:

```bash
electripy rag eval \
  --corpus data/corpus.jsonl \
  --queries data/queries.jsonl \
  --top-k 3,5,10 \
  --chunk-size 500 \
  --chunk-overlap 100 \
  --embedder fake \
  --report-json reports/rag_eval_report.json \
  --report-csv reports/rag_eval_report.csv
```

To gate CI on hit rate@5 >= 0.85 across all experiments:

```bash
electripy rag eval \
  --corpus data/corpus.jsonl \
  --queries data/queries.jsonl \
  --top-k 5 \
  --chunk-size 500 \
  --chunk-overlap 100 \
  --embedder fake \
  --report-json reports/rag_eval_report.json \
  --fail-under hit_rate@5=0.85
```

If the threshold is not met, the command exits with a non-zero status,
which is suitable for CI pipelines.

## Tips

- Start with the fake embedder to validate datasets and wiring.
- Add real embedders by wiring `EmbeddingPort` implementations into
  your own orchestration, keeping the CLI arguments stable.
- Store reports in a long-lived bucket or artifact store to track
  retrieval quality over time.
