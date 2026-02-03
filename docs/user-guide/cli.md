# CLI Module

Command-line interface built with Typer and Rich.

## Overview

ElectriPy provides a production-ready CLI with built-in health checks and diagnostics.

## Commands

### doctor

Check installation health and dependencies.

```bash
electripy doctor
```

Output includes:
- Python version check (requires 3.11+)
- ElectriPy version
- Configuration status
- Dependency verification

Example output:
```
ElectriPy Doctor
==================================================
Check                          Status          Details
Python Version                 ✓ OK           3.11.5
ElectriPy Version              ✓ OK           0.1.0
Configuration                  ✓ OK           Log level: INFO
typer package                  ✓ OK           0.9.0
rich package                   ✓ OK           13.5.0

✓ ElectriPy is ready to use!
```

### version

Display ElectriPy version.

```bash
electripy version
```

### Global Options

- `--verbose, -v`: Enable verbose output
- `--help`: Show help message

## Extending the CLI

Create custom commands by extending the Typer app:

```python
from electripy.cli import app
import typer

@app.command()
def custom_command(
    name: str = typer.Option(..., help="Your name")
):
    """Custom command example."""
    typer.echo(f"Hello, {name}!")

if __name__ == "__main__":
    app()
```

## Using in Scripts

Import and use the CLI app programmatically:

```python
from typer.testing import CliRunner
from electripy.cli import app

runner = CliRunner()
result = runner.invoke(app, ["doctor"])
print(result.stdout)
```
