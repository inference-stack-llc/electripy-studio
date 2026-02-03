# CLI Tool Recipe

This recipe demonstrates building a production-ready CLI tool using ElectriPy components.

## Overview

This example shows how to:
- Build a CLI tool with Typer
- Process JSONL files
- Implement retry logic
- Add rate limiting
- Include health checks

## Files

- `data_processor.py` - Main CLI tool implementation
- `sample_input.jsonl` - Sample input data for testing
- `README.md` - This file

## Quick Start

1. Install ElectriPy:
   ```bash
   cd ../..
   pip install -e .
   ```

2. Run the processor:
   ```bash
   python data_processor.py sample_input.jsonl output.jsonl
   ```

3. Check health:
   ```bash
   python data_processor.py doctor
   ```

## Features

- **Async Processing**: Concurrent data processing with asyncio
- **Rate Limiting**: Token bucket algorithm for API rate limiting
- **Retry Logic**: Automatic retry with exponential backoff
- **Rich Output**: Beautiful terminal output with Rich
- **Type Safety**: Full type hints for better IDE support

## Customization

Extend this example by:
- Adding database connections
- Implementing API clients
- Adding authentication
- Creating more CLI commands
- Adding progress bars

## See Also

- Full recipe documentation: ../../docs/recipes/cli-tool.md
- ElectriPy documentation: ../../docs/
