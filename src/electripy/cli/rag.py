"""RAG-related CLI commands.

This module wires the RAG evaluation runner into the main CLI.
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from electripy.ai.rag_eval_runner.domain import RunResult
from electripy.ai.rag_eval_runner.errors import RagEvalError
from electripy.ai.rag_eval_runner.services import (
    ReportWriter,
    build_default_experiment_config,
    enforce_fail_under_thresholds,
    parse_fail_under_threshold,
    parse_top_k_csv,
    run_experiments,
)

app = typer.Typer(name="rag", help="RAG utilities")

_console = Console()


@app.command("eval")
def rag_eval(
    corpus: Path = typer.Option(..., help="Path to corpus JSONL file"),
    queries: Path = typer.Option(..., help="Path to queries JSONL file"),
    top_k: str = typer.Option("5", help="Comma-separated list of k values (e.g. '3,5,10')"),
    chunk_size: int = typer.Option(1000, help="Chunk size in characters"),
    chunk_overlap: int = typer.Option(200, help="Chunk overlap in characters"),
    chunker_config: Path | None = typer.Option(
        None,
        help="Optional JSON file describing advanced chunking configs",
    ),
    embedder: str = typer.Option(
        "fake",
        help="Embedder name(s), comma-separated (e.g. 'fake' or 'fake,openai')",
    ),
    report_json: Path | None = typer.Option(None, help="Path to JSON report output"),
    report_csv: Path | None = typer.Option(None, help="Path to CSV report output"),
    fail_under: list[str] = typer.Option(
        [],
        help="Fail if metric threshold not met (e.g. 'hit_rate@5=0.85')",
    ),
    seed: int | None = typer.Option(
        None,
        help="Reserved for future use; runs are deterministic without it",
    ),
    verbose: bool = typer.Option(False, help="Enable verbose console output"),
) -> None:
    """Run RAG evaluation experiments over the given datasets."""

    del seed  # Currently unused but reserved for future deterministic hooks.

    try:
        top_k_values = parse_top_k_csv(top_k)
        embedder_names = [name.strip() for name in embedder.split(",") if name.strip()]
        if not embedder_names:
            raise typer.BadParameter("At least one embedder name must be provided")

        if chunker_config is not None:
            # When a JSON config is provided, it takes precedence over
            # the simple size/overlap options. For now we expect a
            # single-object file compatible with ChunkingConfig.
            data = json.loads(chunker_config.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                raise typer.BadParameter("chunker-config JSON must be an object")
            chunk_size_val = int(data.get("chunk_size_chars", chunk_size))
            chunk_overlap_val = int(data.get("overlap_chars", chunk_overlap))
        else:
            chunk_size_val = chunk_size
            chunk_overlap_val = chunk_overlap

        config = build_default_experiment_config(
            corpus_path=corpus,
            queries_path=queries,
            chunk_size=chunk_size_val,
            chunk_overlap=chunk_overlap_val,
            embedder_names=embedder_names,
            top_k_values=top_k_values,
        )

        result = run_experiments(config)

        thresholds = [parse_fail_under_threshold(expr) for expr in fail_under]
        if thresholds:
            enforce_fail_under_thresholds(result=result, thresholds=thresholds)

        writer = ReportWriter()
        if report_json is not None:
            writer.write_json(report_json, result)
        if report_csv is not None:
            writer.write_csv(report_csv, result)

        _render_console_summary(result, verbose=verbose)

    except RagEvalError as exc:
        _console.print(f"[red]RAG evaluation error:[/red] {exc}")
        raise typer.Exit(code=1) from exc


def _render_console_summary(result: RunResult, *, verbose: bool) -> None:
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Experiment ID", overflow="fold")
    table.add_column("Chunker")
    table.add_column("Embedder")
    table.add_column("k")
    table.add_column("hit_rate")
    table.add_column("precision")
    table.add_column("recall")
    table.add_column("mrr")

    def _get(aggregate_metrics: dict[str, float], name: str) -> float:
        """Return the first metric matching ``name`` or ``0.0``.

        This helper searches for keys like ``"hit_rate@5"`` by
        checking for the ``name`` prefix followed by ``"@"``.
        """

        for key, value in aggregate_metrics.items():
            if key.startswith(name + "@"):  # e.g. hit_rate@5
                return float(value)
        return 0.0

    for exp in result.experiments:
        metrics = dict(exp.aggregate_metrics)
        table.add_row(
            exp.experiment_id[:8],
            exp.chunker_name,
            exp.embedder_name,
            str(exp.top_k),
            f"{_get(metrics, 'hit_rate'):.3f}",
            f"{_get(metrics, 'precision'):.3f}",
            f"{_get(metrics, 'recall'):.3f}",
            f"{_get(metrics, 'mrr'):.3f}",
        )

    _console.print(table)

    if verbose:
        _console.print()
        _console.print(
            "[dim]Per-query metrics are available in the JSON report when enabled.[/dim]"
        )
