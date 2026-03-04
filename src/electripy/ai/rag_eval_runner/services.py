"""Services and orchestration for the RAG evaluation runner.

Purpose:
  - Load corpus and query datasets from JSONL files.
  - Build temporary indices using the existing RAG kit ports.
  - Run matrix experiments over chunkers, embedders, and top-k values.
  - Compute deterministic retrieval metrics and prepare reports.

Guarantees:
  - Orchestration depends only on ElectriPy domain models and ports,
    never directly on third-party clients.
  - All operations are synchronous and deterministic given the same
    inputs.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path

from electripy.ai.rag.config import ChunkingConfig, EmbeddingGatewaySettings
from electripy.ai.rag.domain import Chunk, Document, GroundTruthExample, RetrievalResult
from electripy.ai.rag.errors import EmbeddingError
from electripy.ai.rag.evaluation import hit_rate_at_k, precision_at_k, recall_at_k
from electripy.ai.rag.ports import ChunkerPort, EmbeddingPort, VectorStorePort
from electripy.ai.rag.services import DeterministicChunker, EmbeddingGateway, IndexingService
from electripy.core.logging import get_logger
from electripy.core.typing import JSONDict

from .adapters import FakeEmbeddingAdapter, InMemoryVectorStoreAdapter
from .domain import (
    _METRIC_NAME_HIT_RATE,
    _METRIC_NAME_MRR,
    _METRIC_NAME_PRECISION,
    _METRIC_NAME_RECALL,
    CorpusRecord,
    EmbedderVariant,
    ExperimentConfig,
    ExperimentResult,
    PerQueryMetrics,
    QueryRecord,
    RunMetadata,
    RunResult,
    metric_key,
)
from .errors import DatasetFormatError, EvalRunnerError, ExperimentConfigError, RagEvalError

logger = get_logger(__name__)


@dataclass(slots=True)
class DatasetLoader:
    """Load corpus and query datasets from JSONL files.

    The loader accepts files where:

    - Blank lines are ignored.
    - Lines starting with ``#`` are treated as comments and skipped.
    - Each remaining line must be a valid JSON object.

    The schema is validated strictly and failures raise
    :class:`DatasetFormatError`.
    """

    corpus_path: Path
    queries_path: Path

    def load(self) -> tuple[list[CorpusRecord], list[QueryRecord]]:
        corpus_records = list(self._load_corpus())
        query_records = list(self._load_queries())
        if not corpus_records:
            raise DatasetFormatError("Corpus dataset is empty")
        if not query_records:
            raise DatasetFormatError("Queries dataset is empty")
        return corpus_records, query_records

    def _iter_jsonl(self, path: Path) -> Iterable[JSONDict]:
        try:
            with path.open("r", encoding="utf-8") as f:
                for idx, raw_line in enumerate(f, start=1):
                    line = raw_line.strip()
                    if not line or line.startswith("#"):
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError as exc:  # noqa: TRY003
                        raise DatasetFormatError(
                            f"Invalid JSON on line {idx} of {path}: {exc.msg}",
                        ) from exc
                    if not isinstance(obj, dict):
                        raise DatasetFormatError(
                            f"Line {idx} of {path} must contain a JSON object",
                        )
                    yield obj
        except FileNotFoundError as exc:  # pragma: no cover - basic filesystem error
            raise DatasetFormatError(f"File not found: {path}") from exc

    def _load_corpus(self) -> Iterable[CorpusRecord]:
        for obj in self._iter_jsonl(self.corpus_path):
            if "id" not in obj or "text" not in obj:
                raise DatasetFormatError("Corpus record missing required fields 'id' or 'text'")
            record_id = str(obj["id"])
            text = str(obj["text"])
            source_uri = obj.get("source_uri")
            metadata = obj.get("metadata")
            if source_uri is not None and not isinstance(source_uri, str):
                raise DatasetFormatError("source_uri must be a string when present")
            if metadata is not None and not isinstance(metadata, Mapping):
                raise DatasetFormatError("metadata must be an object when present")
            yield CorpusRecord(
                id=record_id,
                text=text,
                source_uri=source_uri,
                metadata=metadata,
            )

    def _load_queries(self) -> Iterable[QueryRecord]:
        for obj in self._iter_jsonl(self.queries_path):
            if "id" not in obj or "query" not in obj or "relevant_ids" not in obj:
                raise DatasetFormatError(
                    "Query record missing required fields 'id', 'query', or 'relevant_ids'",
                )
            record_id = str(obj["id"])
            query = str(obj["query"])
            relevant_ids_raw = obj["relevant_ids"]
            metadata = obj.get("metadata")
            if not isinstance(relevant_ids_raw, list) or not all(
                isinstance(x, str) for x in relevant_ids_raw
            ):
                raise DatasetFormatError("relevant_ids must be a list of strings")
            if metadata is not None and not isinstance(metadata, Mapping):
                raise DatasetFormatError("metadata must be an object when present")
            yield QueryRecord(
                id=record_id,
                query=query,
                relevant_ids=list(relevant_ids_raw),
                metadata=metadata,
            )


@dataclass(slots=True)
class IndexBuilder:
    """Build an index for a single experiment.

    The builder uses the existing RAG kit's deterministic chunker and
    embedding gateway on top of the provided ports.
    """

    chunker: ChunkerPort
    embedding_gateway: EmbeddingGateway
    vector_store: VectorStorePort

    def build_index(self, corpus: Sequence[CorpusRecord]) -> list[Chunk]:
        indexing = IndexingService(
            chunker=self.chunker,
            embedding_gateway=self.embedding_gateway,
            vector_store=self.vector_store,
        )
        chunks: list[Chunk] = []
        for record in corpus:
            document = Document(
                id=record.id,
                source_uri=record.source_uri or "memory://",
                text=record.text,
                metadata=record.metadata,
            )
            chunks.extend(indexing.index_document(document, force=True))
        return chunks


def _mean_reciprocal_rank_at_k(
    results_by_query: Mapping[str, Sequence[RetrievalResult]],
    *,
    ground_truth: Sequence[GroundTruthExample],
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not ground_truth:
        return 0.0

    truth_index: dict[str, frozenset[str]] = {
        example.query_text: example.relevant_chunk_ids for example in ground_truth
    }

    total_rr = 0.0
    for query_text, relevant_ids in truth_index.items():
        results = list(results_by_query.get(query_text, ()))[:k]
        rr = 0.0
        for rank, result in enumerate(results, start=1):
            if result.chunk.id in relevant_ids:
                rr = 1.0 / float(rank)
                break
        total_rr += rr

    return total_rr / float(len(truth_index))


@dataclass(slots=True)
class Evaluator:
    """Evaluate retrieval quality for a single experiment."""

    vector_store: VectorStorePort
    embedding_port: EmbeddingPort

    def evaluate(
        self,
        queries: Sequence[QueryRecord],
        *,
        top_k: int,
    ) -> tuple[Mapping[str, float], Sequence[PerQueryMetrics]]:
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        results_by_query: dict[str, list[RetrievalResult]] = {}
        ground_truth: list[GroundTruthExample] = []

        for query in queries:
            [vector] = self.embedding_port.embed_texts([query.query])
            neighbours = self.vector_store.query(vector, top_k=top_k, filters=None)
            retrieval_results = [
                RetrievalResult(chunk=chunk, score=score) for chunk, score in neighbours
            ]
            # Use query id as the key to avoid ambiguity for duplicate texts.
            key = query.id
            results_by_query[key] = retrieval_results
            ground_truth.append(
                GroundTruthExample(
                    query_text=key,
                    relevant_chunk_ids=frozenset(query.relevant_ids),
                ),
            )

        metrics: dict[str, float] = {}
        metrics[metric_key(_METRIC_NAME_HIT_RATE, top_k)] = hit_rate_at_k(
            results_by_query,
            ground_truth=ground_truth,
            k=top_k,
        )
        metrics[metric_key(_METRIC_NAME_PRECISION, top_k)] = precision_at_k(
            results_by_query,
            ground_truth=ground_truth,
            k=top_k,
        )
        metrics[metric_key(_METRIC_NAME_RECALL, top_k)] = recall_at_k(
            results_by_query,
            ground_truth=ground_truth,
            k=top_k,
        )
        metrics[metric_key(_METRIC_NAME_MRR, top_k)] = _mean_reciprocal_rank_at_k(
            results_by_query,
            ground_truth=ground_truth,
            k=top_k,
        )

        per_query: list[PerQueryMetrics] = []
        for query in queries:
            key = query.id
            results = results_by_query.get(key, [])
            # Per-query metrics mirror the aggregate definitions but are
            # computed for a single query.
            if results:
                retrieved_ids = [r.chunk.id for r in results]
            else:
                retrieved_ids = []
            relevant_ids = set(query.relevant_ids)

            true_positives = len(set(retrieved_ids) & relevant_ids)
            precision = float(true_positives) / float(len(retrieved_ids)) if retrieved_ids else 0.0
            recall = float(true_positives) / float(len(relevant_ids)) if relevant_ids else 0.0
            hit = 1.0 if true_positives > 0 else 0.0

            rr = 0.0
            for rank, cid in enumerate(retrieved_ids, start=1):
                if cid in relevant_ids:
                    rr = 1.0 / float(rank)
                    break

            per_query_metrics = {
                metric_key(_METRIC_NAME_HIT_RATE, top_k): hit,
                metric_key(_METRIC_NAME_PRECISION, top_k): precision,
                metric_key(_METRIC_NAME_RECALL, top_k): recall,
                metric_key(_METRIC_NAME_MRR, top_k): rr,
            }

            per_query.append(
                PerQueryMetrics(
                    query_id=query.id,
                    query_text=query.query,
                    metrics=per_query_metrics,
                ),
            )

        return metrics, per_query


@dataclass(slots=True)
class FailUnderThreshold:
    """Threshold constraint for CI-friendly gating.

    Attributes:
        metric_key: Metric name with ``@k`` suffix, for example
            ``"hit_rate@5"``.
        minimum: Minimum acceptable value for the metric.
    """

    metric_key: str
    minimum: float


def parse_fail_under_threshold(raw: str) -> FailUnderThreshold:
    """Parse a ``--fail-under`` expression.

    The expected format is ``"<metric@k>=<value>"``, for example
    ``"hit_rate@5=0.85"``.
    """

    if "=" not in raw:
        raise ExperimentConfigError(
            "fail-under threshold must be of the form '<metric@k>=<value>'",
        )
    key, value_str = raw.split("=", 1)
    key = key.strip()
    try:
        minimum = float(value_str)
    except ValueError as exc:  # noqa: TRY003
        raise ExperimentConfigError(f"Invalid numeric threshold value in '{raw}'") from exc
    return FailUnderThreshold(metric_key=key, minimum=minimum)


def parse_top_k_csv(raw: str) -> list[int]:
    """Parse a comma-separated list of integer ``k`` values."""

    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ExperimentConfigError("top-k list must not be empty")
    values: list[int] = []
    for part in parts:
        try:
            value = int(part)
        except ValueError as exc:  # noqa: TRY003
            raise ExperimentConfigError(f"Invalid integer in top-k list: '{part}'") from exc
        if value <= 0:
            raise ExperimentConfigError("All top-k values must be positive")
        values.append(value)
    # Deduplicate while preserving order for determinism.
    seen: set[int] = set()
    unique_values: list[int] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            unique_values.append(value)
    return unique_values


@dataclass(slots=True)
class ReportWriter:
    """Serialise run results into JSON and CSV artefacts."""

    def to_json_dict(self, result: RunResult) -> JSONDict:
        data: JSONDict = {
            "run_metadata": asdict(result.run_metadata),
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "chunker_name": exp.chunker_name,
                    "embedder_name": exp.embedder_name,
                    "top_k": exp.top_k,
                    "aggregate_metrics": dict(exp.aggregate_metrics),
                    "per_query": [
                        {
                            "query_id": pq.query_id,
                            "query_text": pq.query_text,
                            "metrics": dict(pq.metrics),
                        }
                        for pq in exp.per_query
                    ],
                }
                for exp in result.experiments
            ],
        }
        return data

    def write_json(self, path: Path, result: RunResult) -> None:
        data = self.to_json_dict(result)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def write_csv(self, path: Path, result: RunResult) -> None:
        import csv

        path.parent.mkdir(parents=True, exist_ok=True)
        fieldnames = [
            "experiment_id",
            "chunker_name",
            "embedder_name",
            "top_k",
        ]
        # Collect the union of metric keys for a stable header.
        metric_keys: list[str] = []
        seen: set[str] = set()
        for exp in result.experiments:
            for key in exp.aggregate_metrics.keys():
                if key not in seen:
                    seen.add(key)
                    metric_keys.append(key)
        fieldnames.extend(metric_keys)

        with path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for exp in result.experiments:
                row: dict[str, object] = {
                    "experiment_id": exp.experiment_id,
                    "chunker_name": exp.chunker_name,
                    "embedder_name": exp.embedder_name,
                    "top_k": exp.top_k,
                }
                for key in metric_keys:
                    row[key] = exp.aggregate_metrics.get(key, 0.0)
                writer.writerow(row)


def _build_experiment_id(
    *,
    run_metadata: RunMetadata,
    chunk_variant: ChunkingConfig,
    embedder: EmbedderVariant,
    top_k: int,
) -> str:
    payload = {
        "run_name": run_metadata.name,
        "chunk_size_chars": chunk_variant.chunk_size_chars,
        "overlap_chars": chunk_variant.overlap_chars,
        "embedder_name": embedder.name,
        "embedder_provider": embedder.provider,
        "embedder_params": dict(embedder.params) if embedder.params is not None else {},
        "top_k": top_k,
    }
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return sha256(raw.encode("utf-8")).hexdigest()


def _build_default_embedding_port(name: str) -> EmbeddingPort:
    if name.lower() == "fake":
        return FakeEmbeddingAdapter()
    if name.lower() == "openai":
        try:
            from electripy.ai.rag.adapters.openai_embeddings_adapter import (  # noqa: PLC0415
                OpenAiEmbeddingAdapter,
            )
        except ImportError as exc:  # pragma: no cover - depends on optional package
            raise ExperimentConfigError(
                "OpenAI embedder requested but openai package is not installed",
            ) from exc
        return OpenAiEmbeddingAdapter(model="text-embedding-3-small")
    raise ExperimentConfigError(f"Unsupported embedder name: {name}")


def run_experiments(
    config: ExperimentConfig,
) -> RunResult:
    """Run all experiments described by ``config`` synchronously.

    Returns:
        RunResult containing aggregate and per-query metrics.

    Raises:
        RagEvalError: For configuration, dataset, or orchestration
            errors.
    """

    loader = DatasetLoader(corpus_path=config.corpus_path, queries_path=config.queries_path)
    corpus, queries = loader.load()

    experiments: list[ExperimentResult] = []

    for chunk_variant in config.chunk_variants:
        chunker = DeterministicChunker(config=chunk_variant.config)
        for embedder_variant in config.embedder_variants:
            embedding_port = _build_default_embedding_port(embedder_variant.name)
            gateway = EmbeddingGateway(
                port=embedding_port,
                settings=EmbeddingGatewaySettings(),
            )
            vector_store: VectorStorePort = InMemoryVectorStoreAdapter()
            index_builder = IndexBuilder(
                chunker=chunker,
                embedding_gateway=gateway,
                vector_store=vector_store,
            )
            try:
                index_builder.build_index(corpus)
            except EmbeddingError as exc:
                raise EvalRunnerError("Embedding failed during indexing") from exc

            evaluator = Evaluator(vector_store=vector_store, embedding_port=embedding_port)

            for top_k in config.top_k_values:
                experiment_id = _build_experiment_id(
                    run_metadata=config.run_metadata,
                    chunk_variant=chunk_variant.config,
                    embedder=embedder_variant,
                    top_k=top_k,
                )
                metrics, per_query = evaluator.evaluate(queries, top_k=top_k)
                experiments.append(
                    ExperimentResult(
                        experiment_id=experiment_id,
                        chunker_name=chunk_variant.name,
                        embedder_name=embedder_variant.name,
                        top_k=top_k,
                        aggregate_metrics=metrics,
                        per_query=per_query if config.include_per_query_breakdown else [],
                    ),
                )

    return RunResult(run_metadata=config.run_metadata, experiments=experiments)


def enforce_fail_under_thresholds(
    *,
    result: RunResult,
    thresholds: Sequence[FailUnderThreshold],
) -> None:
    """Raise :class:`RagEvalError` if any threshold is violated.

    The policy is conservative: for each threshold, **all** experiments
    must meet or exceed the specified minimum value. This keeps CI
    semantics predictable.
    """

    for threshold in thresholds:
        for exp in result.experiments:
            value = exp.aggregate_metrics.get(threshold.metric_key)
            if value is None:
                raise RagEvalError(
                    f"Metric '{threshold.metric_key}' not found in experiment '{exp.experiment_id}'",
                )
            if value < threshold.minimum:
                raise RagEvalError(
                    "Threshold not met: "
                    f"{threshold.metric_key}={value:.4f} < {threshold.minimum:.4f} "
                    f"for experiment {exp.experiment_id}",
                )


def build_default_experiment_config(
    *,
    corpus_path: Path,
    queries_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    embedder_names: Sequence[str],
    top_k_values: Sequence[int],
    run_name: str | None = None,
    output_json: Path | None = None,
    output_csv: Path | None = None,
    include_per_query: bool = True,
) -> ExperimentConfig:
    """Helper to construct a basic :class:`ExperimentConfig`.

    This is primarily used by the CLI wiring.
    """

    if not embedder_names:
        raise ExperimentConfigError("At least one embedder name must be provided")

    chunk_config = ChunkingConfig(chunk_size_chars=chunk_size, overlap_chars=chunk_overlap)

    from .domain import ChunkingVariant as DomainChunkingVariant  # noqa: PLC0415

    chunk_variants = [
        DomainChunkingVariant(name="default", config=chunk_config),
    ]

    embedder_variants = [
        EmbedderVariant(name=name, provider=name, params=None) for name in embedder_names
    ]

    timestamp_iso = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    run_metadata = RunMetadata(
        name=run_name or "rag-eval-run",
        timestamp_iso=timestamp_iso,
        git_sha=None,
    )

    return ExperimentConfig(
        run_metadata=run_metadata,
        corpus_path=corpus_path,
        queries_path=queries_path,
        chunk_variants=chunk_variants,
        embedder_variants=embedder_variants,
        top_k_values=list(top_k_values),
        output_json_path=output_json,
        output_csv_path=output_csv,
        include_per_query_breakdown=include_per_query,
    )
