# AI Evals

**AI Evals** is ElectriPy's production-ready evaluation framework for AI
systems.  It provides a structured, dataset-driven pipeline for scoring
model outputs, comparing runs against baselines, and generating
CI-friendly reports — replacing ad hoc scripts and notebooks with a
reusable, extensible system.

## When to use it

- You need **offline evaluation** of LLM outputs against ground truth
  datasets before deploying to production.
- You want **retrieval quality scoring** (hit@k, recall@k, MRR@k) for
  RAG pipelines.
- You need **tool-call correctness** evaluation for agentic AI systems.
- You want **regression detection** — compare a new model or prompt
  revision against a known baseline and fail CI if metrics drop.
- You want **structured, machine-readable reports** (JSON, Markdown)
  for engineering review and automated gates.

## Core concepts

- **Domain models**:
    - `EvalCase` — a single test case with input, ground truth,
      expected tool calls, expected retrieval, and metadata.
    - `EvalDataset` — a named collection of cases.
    - `EvalMetric` — a named metric value with optional pass/fail
      threshold.
    - `EvalScore` — a per-case metric from a specific scorer.
    - `EvalResult` — full result for a case: all scores, pass/fail,
      failures.
    - `EvalSummary` — aggregate summary: total, passed, failed,
      pass rate, per-metric averages.
    - `EvalRun` — a complete run with ID, timestamp, summary, and
      artifacts.
    - `RegressionComparison` / `RegressionDelta` — baseline-vs-current
      delta analysis.
- **Ports** (protocol interfaces):
    - `ScorerPort` — score a model output against a case.
    - `DatasetLoaderPort` — load datasets from external sources.
    - `ReportWriterPort` — write summaries to files or APIs.
    - `ArtifactStorePort` — persist evaluation artifacts.
    - `ModelInvocationPort` — invoke a model during offline eval runs.
- **Built-in scorers**:
    - `ExactMatchScorer` — exact string match against reference.
    - `NormalizedTextScorer` — case-insensitive, whitespace-normalized.
    - `ContainsScorer` — checks substring presence.
    - `JsonStructureScorer` — validates JSON field presence and types.
    - `RetrievalScorer` — hit@k, recall@k, MRR@k via `rag_quality`.
    - `ToolCallScorer` — tool name and argument correctness.
    - `ThresholdScorer` — wraps any scorer with pass/fail thresholds.
    - `CompositeScorer` — runs multiple scorers together.
- **Adapters**:
    - `JsonlDatasetLoader` — loads JSONL dataset files.
    - `JsonReportWriter` / `MarkdownReportWriter` — report output.
    - `FileArtifactStore` — saves artifacts to disk.
    - `CallbackModelInvocation` — wraps a callable as a model port.
- **Service**:
    - `EvalRunner` — orchestrates the full dataset → score → summarize
      → compare pipeline.

## Quick start

```python
from electripy.ai.evals import (
    EvalCase,
    EvalDataset,
    EvalRunner,
    ExactMatchScorer,
    GroundTruth,
    ThresholdScorer,
)

dataset = EvalDataset(
    name="capitals",
    cases=(
        EvalCase(
            case_id="q1",
            input="Capital of France?",
            ground_truth=GroundTruth(reference_output="Paris"),
        ),
        EvalCase(
            case_id="q2",
            input="Capital of Japan?",
            ground_truth=GroundTruth(reference_output="Tokyo"),
        ),
    ),
)

runner = EvalRunner(
    scorers=[
        ThresholdScorer(
            inner=ExactMatchScorer(),
            thresholds={"exact_match": 1.0},
        ),
    ],
)

run = runner.run_dataset(
    dataset,
    outputs={"q1": "Paris", "q2": "Tokyo"},
)

print(f"Pass rate: {run.summary.pass_rate:.0%}")  # 100%
```

## Dataset format (JSONL)

Each line is a JSON object representing one case:

```json
{"id": "q1", "input": "Capital of France?", "reference_output": "Paris"}
{"id": "q2", "input": "Capital of Japan?", "reference_output": "Tokyo", "acceptable_alternatives": ["Tōkyō"]}
```

Load with:

```python
from electripy.ai.evals import JsonlDatasetLoader

dataset = JsonlDatasetLoader().load("tests/fixtures/capitals.jsonl")
```

### Tool-call cases

```json
{"id": "t1", "input": "Weather in NYC?", "expected_tool_calls": [{"name": "get_weather", "expected_args": {"city": "NYC"}}]}
```

### Retrieval cases

```json
{"id": "r1", "input": "How to deploy?", "expected_retrieval": {"expected_ids": ["doc-deploy-1", "doc-deploy-2"], "k": 5}}
```

## Custom scorer

Implement `ScorerPort`:

```python
from electripy.ai.evals import EvalCase, EvalScore, EvalMetric

class SentimentScorer:
    @property
    def name(self) -> str:
        return "sentiment"

    def score(self, case, actual_output, **kwargs):
        is_positive = "good" in actual_output.lower()
        return [
            EvalScore(
                case_id=case.case_id,
                scorer_name=self.name,
                metric=EvalMetric(name="sentiment_positive", value=1.0 if is_positive else 0.0),
            ),
        ]
```

## Retrieval evaluation

Score RAG retrieval quality with `RetrievalScorer`:

```python
from electripy.ai.evals import (
    EvalCase,
    EvalRunner,
    RetrievalExpectation,
    RetrievalScorer,
)

case = EvalCase(
    case_id="r1",
    input="How to deploy?",
    expected_retrieval=RetrievalExpectation(
        expected_ids=("doc-deploy-1", "doc-deploy-2"),
        k=5,
    ),
)

runner = EvalRunner(scorers=[RetrievalScorer()])
result = runner.score_case(
    case, "",
    retrieved_ids=["doc-deploy-1", "doc-other", "doc-deploy-2", "x", "y"],
)

for score in result.scores:
    print(f"{score.metric.name}: {score.metric.value:.2f}")
# hit_at_k: 1.00
# recall_at_k: 1.00
# mrr_at_k: 1.00
```

## Tool-call evaluation

Score tool invocation correctness with `ToolCallScorer`:

```python
from electripy.ai.evals import (
    EvalCase,
    EvalRunner,
    ToolCallExpectation,
    ToolCallScorer,
)

case = EvalCase(
    case_id="t1",
    expected_tool_calls=(
        ToolCallExpectation(
            tool_name="get_weather",
            expected_args={"city": "NYC"},
        ),
    ),
)

runner = EvalRunner(scorers=[ToolCallScorer()])
result = runner.score_case(
    case, "",
    tool_calls=[{"name": "get_weather", "arguments": {"city": "NYC"}}],
)

for score in result.scores:
    print(f"{score.metric.name}: {score.metric.value:.2f}")
# tool_name_match: 1.00
# tool_arg_match: 1.00
```

## Regression detection and CI gating

Compare a current run against a baseline and fail on regressions:

```python
from electripy.ai.evals import EvalRunner, EvalMetric, EvalSummary

runner = EvalRunner()

baseline = EvalSummary(
    dataset_name="test",
    metrics=(
        EvalMetric(name="exact_match", value=0.9),
        EvalMetric(name="recall_at_k", value=0.85),
    ),
)

current = EvalSummary(
    dataset_name="test",
    metrics=(
        EvalMetric(name="exact_match", value=0.7),
        EvalMetric(name="recall_at_k", value=0.88),
    ),
)

comparison = runner.compare_runs(
    baseline,
    current,
    thresholds={"exact_match": 0.05, "recall_at_k": 0.05},
    fail_on_regression=True,  # raises RegressionError
)
```

For CI integration, set `fail_on_regression=True`.  The runner raises
`RegressionError` listing the regressed metrics, suitable for build
failure.

## Reports

### JSON report

```python
from electripy.ai.evals import JsonReportWriter

writer = JsonReportWriter()
writer.write(run.summary, "reports/eval_report.json")
```

### Markdown report

```python
from electripy.ai.evals import MarkdownReportWriter

writer = MarkdownReportWriter()
writer.write(run.summary, "reports/eval_report.md")
```

### Artifacts

```python
from electripy.ai.evals import EvalArtifact, FileArtifactStore

store = FileArtifactStore(base_dir="eval_artifacts")
artifact = EvalArtifact(
    name="full_scores.json",
    format="json",
    content='{"detail": "..."}',
)
path = store.save(artifact, run.run_id)
```

## Model invocation during eval

If outputs are not pre-computed, provide a model port:

```python
from electripy.ai.evals import CallbackModelInvocation, EvalRunner

model = CallbackModelInvocation(
    callback=lambda text, **kw: my_llm_client.complete(text),
)

runner = EvalRunner(
    scorers=[ExactMatchScorer()],
    model=model,
)

run = runner.run_dataset(dataset)  # invokes model for each case
```

## Error handling

- `EvalError` — base exception for the evals framework.
- `DatasetLoadError` — raised when a dataset cannot be loaded or parsed.
- `ScorerError` — raised when a scorer encounters an unrecoverable error.
- `RegressionError` — raised when regression comparison fails a CI gate.

All errors extend `ElectriPyError(Exception)`.
