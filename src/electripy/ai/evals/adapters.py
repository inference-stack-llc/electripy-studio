"""Adapters for the evaluation framework.

Provides concrete implementations of the ports defined in
:mod:`electripy.ai.evals.ports`:

- **JsonlDatasetLoader** — loads eval cases from ``.jsonl`` files.
- **JsonReportWriter** — writes evaluation summaries as JSON.
- **MarkdownReportWriter** — writes evaluation summaries as Markdown.
- **FileArtifactStore** — saves artifacts to a directory.
- **CallbackModelInvocation** — wraps a callable as a model port.

Example::

    from electripy.ai.evals.adapters import JsonlDatasetLoader

    loader = JsonlDatasetLoader()
    dataset = loader.load("tests/fixtures/capitals.jsonl")
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from electripy.core.logging import get_logger

from .domain import (
    EvalArtifact,
    EvalCase,
    EvalDataset,
    EvalSummary,
    GroundTruth,
    RetrievalExpectation,
    ToolCallExpectation,
)
from .errors import DatasetLoadError

__all__ = [
    "CallbackModelInvocation",
    "FileArtifactStore",
    "JsonReportWriter",
    "JsonlDatasetLoader",
    "MarkdownReportWriter",
]

logger = get_logger(__name__)


# ── Dataset loading ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class JsonlDatasetLoader:
    """Loads evaluation datasets from JSONL files.

    Each line in the JSONL file is a JSON object representing one
    :class:`~electripy.ai.evals.domain.EvalCase`.  Blank lines and
    lines starting with ``#`` are ignored.

    Expected JSON keys per line:

    - ``id`` (str, required) — maps to ``case_id``
    - ``input`` (str) — the prompt or query
    - ``reference_output`` (str) — expected answer
    - ``acceptable_alternatives`` (list[str])
    - ``expected_tool_calls`` (list of {name, expected_args, allow_extra_args})
    - ``expected_retrieval`` ({expected_ids, k})
    - ``metadata`` (dict)
    """

    dataset_name: str = ""

    def load(self, source: str) -> EvalDataset:
        """Load a dataset from a JSONL file path.

        Args:
            source: Path to the JSONL file.

        Returns:
            An evaluation dataset.

        Raises:
            DatasetLoadError: If the file cannot be read or parsed.
        """
        path = Path(source)
        if not path.exists():
            raise DatasetLoadError(f"Dataset file not found: {source}")

        cases: list[EvalCase] = []
        try:
            text = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise DatasetLoadError(f"Cannot read {source}: {exc}") from exc

        for line_num, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetLoadError(
                    f"Invalid JSON on line {line_num} of {source}: {exc}"
                ) from exc

            cases.append(_parse_case(obj, line_num, source))

        name = self.dataset_name or path.stem
        return EvalDataset(name=name, cases=tuple(cases))


def _parse_case(obj: dict[str, Any], line_num: int, source: str) -> EvalCase:
    """Parse a JSON object into an EvalCase."""
    case_id = obj.get("id")
    if not case_id:
        raise DatasetLoadError(
            f"Missing 'id' on line {line_num} of {source}"
        )

    ground_truth: GroundTruth | None = None
    if "reference_output" in obj:
        ground_truth = GroundTruth(
            reference_output=obj["reference_output"],
            acceptable_alternatives=tuple(
                obj.get("acceptable_alternatives", [])
            ),
        )

    tool_calls: list[ToolCallExpectation] = []
    for tc in obj.get("expected_tool_calls", []):
        tool_calls.append(
            ToolCallExpectation(
                tool_name=tc["name"],
                expected_args=tc.get("expected_args", {}),
                allow_extra_args=tc.get("allow_extra_args", True),
            )
        )

    retrieval: RetrievalExpectation | None = None
    if "expected_retrieval" in obj:
        r = obj["expected_retrieval"]
        retrieval = RetrievalExpectation(
            expected_ids=tuple(r["expected_ids"]),
            k=r.get("k", 5),
        )

    return EvalCase(
        case_id=str(case_id),
        input=obj.get("input", ""),
        ground_truth=ground_truth,
        expected_tool_calls=tuple(tool_calls),
        expected_retrieval=retrieval,
        metadata=obj.get("metadata", {}),
    )


# ── Report writers ───────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class JsonReportWriter:
    """Writes evaluation summaries as JSON files."""

    indent: int = 2

    def write(self, summary: EvalSummary, destination: str) -> None:
        """Write the summary as a JSON file.

        Args:
            summary: The evaluation summary.
            destination: File path for the JSON output.
        """
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _summary_to_dict(summary)
        path.write_text(
            json.dumps(data, indent=self.indent, default=str) + "\n",
            encoding="utf-8",
        )
        logger.info("Wrote JSON report to %s", destination)


@dataclass(frozen=True, slots=True)
class MarkdownReportWriter:
    """Writes evaluation summaries as Markdown files."""

    def write(self, summary: EvalSummary, destination: str) -> None:
        """Write the summary as a Markdown file.

        Args:
            summary: The evaluation summary.
            destination: File path for the Markdown output.
        """
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = _summary_to_markdown(summary)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        logger.info("Wrote Markdown report to %s", destination)


def _summary_to_dict(summary: EvalSummary) -> dict[str, Any]:
    """Convert an EvalSummary to a serializable dict."""
    return {
        "dataset_name": summary.dataset_name,
        "total": summary.total,
        "passed": summary.passed,
        "failed": summary.failed,
        "pass_rate": summary.pass_rate,
        "metrics": [
            {
                "name": m.name,
                "value": m.value,
                "threshold": m.threshold,
                "passed": m.passed,
            }
            for m in summary.metrics
        ],
        "results": [
            {
                "case_id": r.case_id,
                "passed": r.passed,
                "actual_output": r.actual_output[:500],
                "scores": [
                    {
                        "scorer": s.scorer_name,
                        "metric": s.metric.name,
                        "value": s.metric.value,
                        "threshold": s.metric.threshold,
                    }
                    for s in r.scores
                ],
                "failures": [
                    {"reason": f.reason, "details": f.details}
                    for f in r.failures
                ],
            }
            for r in summary.results
        ],
    }


def _summary_to_markdown(summary: EvalSummary) -> list[str]:
    """Convert an EvalSummary to Markdown lines."""
    lines: list[str] = [
        f"# Eval Report: {summary.dataset_name}",
        "",
        f"**Total**: {summary.total} | "
        f"**Passed**: {summary.passed} | "
        f"**Failed**: {summary.failed} | "
        f"**Pass Rate**: {summary.pass_rate:.1%}",
        "",
    ]

    if summary.metrics:
        lines.append("## Aggregate Metrics")
        lines.append("")
        lines.append("| Metric | Value | Threshold | Passed |")
        lines.append("|--------|-------|-----------|--------|")
        for m in summary.metrics:
            thresh = f"{m.threshold:.4f}" if m.threshold is not None else "—"
            passed = "✅" if m.passed else "❌"
            lines.append(f"| {m.name} | {m.value:.4f} | {thresh} | {passed} |")
        lines.append("")

    if summary.results:
        lines.append("## Per-Case Results")
        lines.append("")
        for r in summary.results:
            status = "✅" if r.passed else "❌"
            lines.append(f"### {status} {r.case_id}")
            lines.append("")
            if r.scores:
                for s in r.scores:
                    lines.append(
                        f"- **{s.scorer_name}** / {s.metric.name}: "
                        f"{s.metric.value:.4f}"
                    )
            if r.failures:
                for f in r.failures:
                    lines.append(f"- **FAIL**: {f.reason}")
            lines.append("")

    return lines


# ── Artifact store ───────────────────────────────────────────────────


@dataclass(slots=True)
class FileArtifactStore:
    """Saves evaluation artifacts to a directory on disk.

    Attributes:
        base_dir: Root directory for artifact storage.
    """

    base_dir: str = "eval_artifacts"

    def save(self, artifact: EvalArtifact, run_id: str) -> str:
        """Save an artifact to disk.

        Args:
            artifact: The artifact to save.
            run_id: Run identifier for namespacing.

        Returns:
            The file path where the artifact was saved.
        """
        run_dir = Path(self.base_dir) / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / artifact.name
        path.write_text(artifact.content, encoding="utf-8")
        logger.info("Saved artifact %s to %s", artifact.name, path)
        return str(path)


# ── Model invocation adapter ─────────────────────────────────────────


@dataclass(slots=True)
class CallbackModelInvocation:
    """Wraps a callable as a :class:`ModelInvocationPort`.

    Attributes:
        callback: A callable that takes ``(input_text, metadata)`` and
            returns the model output string.
    """

    callback: Callable[..., str] = field(default=lambda text, **kw: "")

    def invoke(
        self,
        input_text: str,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Invoke the callback.

        Args:
            input_text: The prompt text.
            metadata: Optional metadata.

        Returns:
            The model's text output.
        """
        return self.callback(input_text, **(metadata or {}))
