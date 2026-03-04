from __future__ import annotations

import json
from pathlib import Path

import pytest

from electripy.ai.rag_eval_runner.errors import DatasetFormatError
from electripy.ai.rag_eval_runner.services import DatasetLoader


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    lines = [json.dumps(record) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_dataset_loader_parses_valid_files(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"

    corpus_records = [
        {"id": "d1", "text": "Hello", "source_uri": "memory://", "metadata": {"k": "v"}},
        {"id": "d2", "text": "World"},
    ]
    query_records = [
        {"id": "q1", "query": "hello", "relevant_ids": ["d1:0"]},
        {"id": "q2", "query": "world", "relevant_ids": ["d2:0"], "metadata": {"x": 1}},
    ]

    _write_jsonl(corpus_path, corpus_records)
    _write_jsonl(queries_path, query_records)

    loader = DatasetLoader(corpus_path=corpus_path, queries_path=queries_path)
    corpus, queries = loader.load()

    assert len(corpus) == 2
    assert corpus[0].id == "d1"
    assert corpus[0].source_uri == "memory://"
    assert len(queries) == 2
    assert queries[0].relevant_ids == ["d1:0"]


def test_dataset_loader_ignores_blank_and_comment_lines(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"

    corpus_path.write_text(
        "\n".join(
            [
                "# comment",
                json.dumps({"id": "d1", "text": "Hello"}),
                "",
                json.dumps({"id": "d2", "text": "World"}),
            ],
        )
        + "\n",
        encoding="utf-8",
    )
    queries_path.write_text(
        "\n".join(
            [
                json.dumps({"id": "q1", "query": "hello", "relevant_ids": ["d1:0"]}),
                "# another comment",
            ],
        )
        + "\n",
        encoding="utf-8",
    )

    loader = DatasetLoader(corpus_path=corpus_path, queries_path=queries_path)
    corpus, queries = loader.load()

    assert len(corpus) == 2
    assert len(queries) == 1


def test_dataset_loader_raises_on_missing_fields(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"

    _write_jsonl(corpus_path, [{"id": "d1"}])  # missing text
    _write_jsonl(queries_path, [{"id": "q1", "query": "q", "relevant_ids": []}])

    loader = DatasetLoader(corpus_path=corpus_path, queries_path=queries_path)

    with pytest.raises(DatasetFormatError):
        loader.load()


def test_dataset_loader_raises_on_invalid_relevant_ids(tmp_path: Path) -> None:
    corpus_path = tmp_path / "corpus.jsonl"
    queries_path = tmp_path / "queries.jsonl"

    _write_jsonl(corpus_path, [{"id": "d1", "text": "Hello"}])
    _write_jsonl(queries_path, [{"id": "q1", "query": "q", "relevant_ids": "not-a-list"}])

    loader = DatasetLoader(corpus_path=corpus_path, queries_path=queries_path)

    with pytest.raises(DatasetFormatError):
        loader.load()
