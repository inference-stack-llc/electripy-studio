#!/usr/bin/env python3
"""Data processor CLI tool - Recipe example."""

import asyncio
from pathlib import Path

import typer
from rich.console import Console

from electripy import get_logger, setup_logging
from electripy.concurrency import AsyncTokenBucketRateLimiter, async_retry
from electripy.io import read_jsonl, write_jsonl

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


@async_retry(max_attempts=3, delay=1.0)
async def process_record(
    record: dict[str, object], limiter: AsyncTokenBucketRateLimiter
) -> dict[str, object]:
    """Process a single record with retry and rate limiting."""
    async with limiter:
        # Simulate API call or processing
        await asyncio.sleep(0.1)

        # Transform the record
        return {
            "id": record["id"],
            "processed": True,
            "result": int(record.get("value", 0)) * 2,
        }


async def process_file_async(
    input_path: Path,
    output_path: Path,
    rate_limit: int,
) -> None:
    """Process file asynchronously with rate limiting."""
    limiter = AsyncTokenBucketRateLimiter(rate=rate_limit, capacity=rate_limit)

    # Read all records
    records = list(read_jsonl(input_path))
    logger.info(f"Processing {len(records)} records")

    # Process with concurrency
    tasks = [process_record(record, limiter) for record in records]
    results = await asyncio.gather(*tasks)

    # Write results
    write_jsonl(output_path, results)
    logger.info(f"Wrote {len(results)} results to {output_path}")


@app.command()
def process(
    input_file: Path = typer.Argument(..., help="Input JSONL file"),
    output_file: Path = typer.Argument(..., help="Output JSONL file"),
    rate_limit: int = typer.Option(10, help="Requests per second"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Process JSONL data with rate limiting and retry."""
    # Setup
    setup_logging(level="DEBUG" if verbose else "INFO")

    console.print(f"[cyan]Processing {input_file} -> {output_file}[/cyan]")

    # Validate input
    if not input_file.exists():
        console.print(f"[red]Error: {input_file} does not exist[/red]")
        raise typer.Exit(1)

    # Process
    try:
        asyncio.run(process_file_async(input_file, output_file, rate_limit))
        console.print("[green]✓ Processing complete![/green]")
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1) from e


@app.command()
def doctor() -> None:
    """Check tool health."""
    from electripy.cli.app import doctor as electripy_doctor

    electripy_doctor()


if __name__ == "__main__":
    app()
