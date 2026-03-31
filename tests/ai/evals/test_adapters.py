"""Tests for electripy.ai.evals.adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from electripy.ai.evals.adapters import (
    CallbackModelInvocation,
    FileArtifactStore,
    JsonlDatasetLoader,
    JsonReportWriter,
    MarkdownReportWriter,
)
from electripy.ai.evals.domain import (
    EvalArtifact,
    EvalMetric,
    EvalResult,
    EvalSummary,
)
from electripy.ai.evals.errors import DatasetLoadError
from electripy.ai.evals.ports import (
    ArtifactStorePort,
    DatasetLoaderPort,
    ModelInvocationPort,
    ReportWriterPort,
)

# ── JsonlDatasetLoader ──────────────────────────────────────────────


class TestJsonlDatasetLoader:
    def test_load_basic(self, tmp_path: Path) -> None:
        data = tmp_path / "test.jsonl"
        data.write_text(
            '{"id": "q1", "input": "Hello", "reference_output": "Hi"}\n'
            '{"id": "q2", "input": "Bye", "reference_output": "Goodbye"}\n'
        )
        loader = JsonlDatasetLoader()
        dataset = loader.load(str(data))
        assert dataset.name == "test"
        assert dataset.size == 2
        assert dataset.cases[0].case_id == "q1"
        assert dataset.cases[0].ground_truth is not None
        assert dataset.cases[0].ground_truth.reference_output == "Hi"

    def test_skips_blank_and_comments(self, tmp_path: Path) -> None:
        data = tmp_path / "test.jsonl"
        data.write_text("# header comment\n" "\n" '{"id": "q1", "input": "Hello"}\n' "\n")
        loader = JsonlDatasetLoader()
        dataset = loader.load(str(data))
        assert dataset.size == 1

    def test_custom_name(self, tmp_path: Path) -> None:
        data = tmp_path / "data.jsonl"
        data.write_text('{"id": "q1"}\n')
        loader = JsonlDatasetLoader(dataset_name="my_dataset")
        dataset = loader.load(str(data))
        assert dataset.name == "my_dataset"

    def test_file_not_found(self) -> None:
        with pytest.raises(DatasetLoadError, match="not found"):
            JsonlDatasetLoader().load("/nonexistent/path.jsonl")

    def test_invalid_json(self, tmp_path: Path) -> None:
        data = tmp_path / "bad.jsonl"
        data.write_text("not json\n")
        with pytest.raises(DatasetLoadError, match="Invalid JSON"):
            JsonlDatasetLoader().load(str(data))

    def test_missing_id(self, tmp_path: Path) -> None:
        data = tmp_path / "noid.jsonl"
        data.write_text('{"input": "hello"}\n')
        with pytest.raises(DatasetLoadError, match="Missing 'id'"):
            JsonlDatasetLoader().load(str(data))

    def test_alternatives(self, tmp_path: Path) -> None:
        data = tmp_path / "alt.jsonl"
        data.write_text(
            '{"id": "q1", "reference_output": "A",' ' "acceptable_alternatives": ["a", "AA"]}\n'
        )
        dataset = JsonlDatasetLoader().load(str(data))
        gt = dataset.cases[0].ground_truth
        assert gt is not None
        assert gt.acceptable_alternatives == ("a", "AA")

    def test_tool_calls(self, tmp_path: Path) -> None:
        data = tmp_path / "tools.jsonl"
        data.write_text(
            '{"id": "q1", "expected_tool_calls": '
            '[{"name": "search", "expected_args": {"q": "test"}}]}\n'
        )
        dataset = JsonlDatasetLoader().load(str(data))
        assert len(dataset.cases[0].expected_tool_calls) == 1
        assert dataset.cases[0].expected_tool_calls[0].tool_name == "search"

    def test_retrieval(self, tmp_path: Path) -> None:
        data = tmp_path / "ret.jsonl"
        data.write_text(
            '{"id": "q1", "expected_retrieval": ' '{"expected_ids": ["d1", "d2"], "k": 3}}\n'
        )
        dataset = JsonlDatasetLoader().load(str(data))
        ret = dataset.cases[0].expected_retrieval
        assert ret is not None
        assert ret.expected_ids == ("d1", "d2")
        assert ret.k == 3

    def test_satisfies_port(self) -> None:
        assert isinstance(JsonlDatasetLoader(), DatasetLoaderPort)


# ── JsonReportWriter ─────────────────────────────────────────────────


def _make_summary() -> EvalSummary:
    return EvalSummary(
        dataset_name="test",
        total=2,
        passed=1,
        failed=1,
        metrics=(EvalMetric(name="exact_match", value=0.5),),
        results=(
            EvalResult(case_id="q1", passed=True, actual_output="Paris"),
            EvalResult(case_id="q2", passed=False, actual_output="London"),
        ),
    )


class TestJsonReportWriter:
    def test_writes_json(self, tmp_path: Path) -> None:
        dest = tmp_path / "report.json"
        JsonReportWriter().write(_make_summary(), str(dest))
        data = json.loads(dest.read_text())
        assert data["dataset_name"] == "test"
        assert data["total"] == 2
        assert data["passed"] == 1
        assert data["pass_rate"] == 0.5
        assert len(data["metrics"]) == 1
        assert len(data["results"]) == 2

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        dest = tmp_path / "sub" / "dir" / "report.json"
        JsonReportWriter().write(_make_summary(), str(dest))
        assert dest.exists()

    def test_satisfies_port(self) -> None:
        assert isinstance(JsonReportWriter(), ReportWriterPort)


# ── MarkdownReportWriter ─────────────────────────────────────────────


class TestMarkdownReportWriter:
    def test_writes_markdown(self, tmp_path: Path) -> None:
        dest = tmp_path / "report.md"
        MarkdownReportWriter().write(_make_summary(), str(dest))
        content = dest.read_text()
        assert "# Eval Report: test" in content
        assert "**Total**: 2" in content
        assert "50.0%" in content

    def test_contains_metrics_table(self, tmp_path: Path) -> None:
        dest = tmp_path / "report.md"
        MarkdownReportWriter().write(_make_summary(), str(dest))
        content = dest.read_text()
        assert "exact_match" in content
        assert "| Metric |" in content

    def test_satisfies_port(self) -> None:
        assert isinstance(MarkdownReportWriter(), ReportWriterPort)


# ── FileArtifactStore ────────────────────────────────────────────────


class TestFileArtifactStore:
    def test_saves_artifact(self, tmp_path: Path) -> None:
        store = FileArtifactStore(base_dir=str(tmp_path / "artifacts"))
        artifact = EvalArtifact(
            name="scores.json",
            format="json",
            content='{"x": 1}',
        )
        path = store.save(artifact, "run-001")
        assert Path(path).exists()
        assert Path(path).read_text() == '{"x": 1}'

    def test_namespaced_by_run(self, tmp_path: Path) -> None:
        store = FileArtifactStore(base_dir=str(tmp_path / "artifacts"))
        a = EvalArtifact(name="data.txt", content="hello")
        path = store.save(a, "run-abc")
        assert "run-abc" in path

    def test_satisfies_port(self) -> None:
        assert isinstance(FileArtifactStore(), ArtifactStorePort)


# ── CallbackModelInvocation ──────────────────────────────────────────


class TestCallbackModelInvocation:
    def test_invokes_callback(self) -> None:
        model = CallbackModelInvocation(
            callback=lambda text, **kw: text.upper(),
        )
        result = model.invoke("hello")
        assert result == "HELLO"

    def test_passes_metadata(self) -> None:
        calls: list[dict] = []

        def _cb(text: str, **kw: str) -> str:
            calls.append({"text": text, **kw})
            return "ok"

        model = CallbackModelInvocation(callback=_cb)
        model.invoke("test", metadata={"model": "gpt-4o"})
        assert calls[0]["model"] == "gpt-4o"

    def test_satisfies_port(self) -> None:
        assert isinstance(CallbackModelInvocation(), ModelInvocationPort)
