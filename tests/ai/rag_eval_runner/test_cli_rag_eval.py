from __future__ import annotations

import json
from pathlib import Path

from typer.testing import CliRunner

from electripy.cli.app import app

runner = CliRunner()


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(record) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_rag_eval_cli_smoke(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"
    json_out = tmp_path / "report.json"
    csv_out = tmp_path / "report.csv"

    _write_jsonl(
        corpus_path,
        [{"id": "d1", "text": "hello world"}],
    )
    _write_jsonl(
        queries_path,
        [{"id": "q1", "query": "hello", "relevant_ids": ["d1:0"]}],
    )

    result = runner.invoke(
        app,
        [
            "rag",
            "eval",
            "--corpus",
            str(corpus_path),
            "--queries",
            str(queries_path),
            "--top-k",
            "1",
            "--embedder",
            "fake",
            "--report-json",
            str(json_out),
            "--report-csv",
            str(csv_out),
        ],
    )

    assert result.exit_code == 0
    assert "Experiment" in result.stdout
    assert json_out.exists()
    assert csv_out.exists()


def test_rag_eval_cli_fail_under_threshold(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"

    _write_jsonl(
        corpus_path,
        [{"id": "d1", "text": "hello world"}],
    )
    _write_jsonl(
        queries_path,
        [{"id": "q1", "query": "hello", "relevant_ids": ["d1:0"]}],
    )

    # Set an impossibly high threshold to force failure.
    result = runner.invoke(
        app,
        [
            "rag",
            "eval",
            "--corpus",
            str(corpus_path),
            "--queries",
            str(queries_path),
            "--top-k",
            "1",
            "--embedder",
            "fake",
            "--fail-under",
            "hit_rate@1=1.1",
        ],
    )

    assert result.exit_code != 0
    assert "RAG evaluation error" in result.stdout
