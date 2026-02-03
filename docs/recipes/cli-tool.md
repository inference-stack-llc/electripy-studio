# Recipe: Building a CLI Tool

Learn how to build a production-ready CLI tool using ElectriPy components.

## Overview

This recipe demonstrates building a data processing CLI tool that:
- Reads data from JSONL files
- Processes data with retry logic
- Rate limits API calls
- Provides health checks

## Complete Example

```python
#!/usr/bin/env python3
"""Data processor CLI tool."""

import asyncio
from pathlib import Path
import typer
from rich.console import Console

from electripy import Config, get_logger, setup_logging
from electripy.concurrency import async_retry, AsyncTokenBucketRateLimiter
from electripy.io import read_jsonl, write_jsonl

app = typer.Typer()
console = Console()
logger = get_logger(__name__)


@async_retry(max_attempts=3, delay=1.0)
async def process_record(record: dict, limiter: AsyncTokenBucketRateLimiter) -> dict:
    """Process a single record with retry and rate limiting."""
    async with limiter:
        # Simulate API call or processing
        await asyncio.sleep(0.1)
        
        # Transform the record
        return {
            "id": record["id"],
            "processed": True,
            "result": record.get("value", 0) * 2,
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
    config = Config.from_env()
    
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
        raise typer.Exit(1)


@app.command()
def doctor() -> None:
    """Check tool health."""
    from electripy.cli.app import doctor as electripy_doctor
    electripy_doctor()


if __name__ == "__main__":
    app()
```

## Usage

Save the above as `data_processor.py` and run:

```bash
# Process a file
python data_processor.py input.jsonl output.jsonl --rate-limit 10

# With verbose output
python data_processor.py input.jsonl output.jsonl -v

# Check health
python data_processor.py doctor
```

## Key Features

1. **Async Processing**: Uses asyncio for concurrent operations
2. **Rate Limiting**: Controls request rate to external services
3. **Retry Logic**: Automatically retries failed operations
4. **Logging**: Structured logging with configurable levels
5. **Error Handling**: Graceful error handling with user feedback
6. **Health Checks**: Built-in diagnostics via doctor command

## Customization

Extend this recipe by:
- Adding more processing steps
- Implementing different data sources (CSV, databases)
- Adding authentication and API client setup
- Creating additional CLI commands
- Adding progress bars with Rich
