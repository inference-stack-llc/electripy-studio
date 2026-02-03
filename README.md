# ElectriPy Studio

Production-minded Python components and recipes (cookbook) by Inference Stack.

[![CI](https://github.com/reactlabs-dev/electripy-studio/actions/workflows/ci.yml/badge.svg)](https://github.com/reactlabs-dev/electripy-studio/actions/workflows/ci.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

ElectriPy Studio is a curated collection of production-ready Python components and recipes designed to accelerate development while maintaining high code quality standards.

## Features

- 🔧 **Core Components**: Configuration, logging, error handling, and type utilities
- ⚡ **Concurrency**: Retry mechanisms (sync/async) and async token bucket rate limiter
- 📁 **I/O**: JSONL read/write utilities for efficient data processing
- 💻 **CLI**: Typer-based command-line interface with health checks

## Quick Start

### Installation

```bash
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

### Verify Installation

```bash
electripy doctor
```

### Basic Usage

```python
from electripy import Config, get_logger
from electripy.concurrency import retry, AsyncTokenBucketRateLimiter
from electripy.io import read_jsonl, write_jsonl

# Configuration
config = Config.from_env()
logger = get_logger(__name__)

# Retry with exponential backoff
@retry(max_attempts=3, delay=1.0, backoff=2.0)
def fetch_data():
    return api_call()

# Rate limiting
limiter = AsyncTokenBucketRateLimiter(rate=10, capacity=10)
async with limiter:
    await rate_limited_operation()

# JSONL I/O
data = [{"id": 1, "name": "Alice"}, {"id": 2, "name": "Bob"}]
write_jsonl("output.jsonl", data)

for record in read_jsonl("output.jsonl"):
    print(record)
```

## Documentation

Full documentation is available in the [docs/](docs/) directory:

- [Installation Guide](docs/getting-started/installation.md)
- [Quickstart](docs/getting-started/quickstart.md)
- [User Guide](docs/user-guide/core.md)
- [Recipes](docs/recipes/cli-tool.md)
- [API Reference](docs/api.md)

Build and serve docs locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Project Structure

```
electripy-studio/
├── src/electripy/          # Main package
│   ├── core/               # Config, logging, errors, typing
│   ├── concurrency/        # Retry & rate limiting
│   ├── io/                 # JSONL utilities
│   └── cli/                # CLI commands
├── tests/                  # Test suite
├── docs/                   # Documentation
├── recipes/                # Example recipes
│   └── 01_cli_tool/        # CLI tool example
├── packages/               # NPM packages
│   └── electripy-cli/      # NPM CLI wrapper
├── pyproject.toml          # Project config
├── mkdocs.yml              # Docs config
└── LICENSE                 # MIT License
```

## Development

### Running Tests

```bash
pytest tests/ -v
```

With coverage:

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### Code Quality

```bash
# Linting
ruff check .

# Formatting
black .

# Type checking
mypy src/
```

### CI/CD

GitHub Actions automatically runs tests, linting, and type checking on all pull requests. See [.github/workflows/ci.yml](.github/workflows/ci.yml).

## Recipes

Check out the [recipes/](recipes/) directory for complete examples:

- [01_cli_tool](recipes/01_cli_tool/) - Building a production-ready CLI tool

## Requirements

- Python 3.11 or higher
- Dependencies managed via `pyproject.toml`

## License

MIT License - See [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please ensure all tests pass and code quality checks succeed before submitting PRs.

## Links

- [GitHub Repository](https://github.com/reactlabs-dev/electripy-studio)
- [Documentation](docs/)
- [Issue Tracker](https://github.com/reactlabs-dev/electripy-studio/issues)
