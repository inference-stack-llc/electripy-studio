# Installation

## Requirements

- Python 3.11 or higher
- pip, poetry, or uv for package management

## Install from PyPI (when available)

```bash
pip install electripy
```

## Install from Source

Clone the repository and install in editable mode:

```bash
git clone https://github.com/inference-stack-llc/electripy-studio.git
cd electripy-studio
pip install -e .
```

## Install Development Dependencies

For development work, install with dev dependencies:

```bash
pip install -e ".[dev]"
```

This includes:
- pytest for testing
- ruff for linting
- black for code formatting
- mypy for type checking

## Install Documentation Dependencies

To build the documentation locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

## Verify Installation

Check that ElectriPy is properly installed:

```bash
electripy doctor
```

This command verifies your Python version, dependencies, and configuration.
