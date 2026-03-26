# Contributing to ElectriPy Studio

Thank you for your interest in contributing! This guide covers everything you need to get started.

## Development setup

```bash
git clone https://github.com/inference-stack-llc/electripy-studio.git
cd electripy-studio
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

## Quality gates (must pass before merge)

| Check | Command |
| --- | --- |
| Lint | `ruff check .` |
| Format | `black --check .` |
| Type-check | `mypy src/` |
| Tests | `pytest tests/ -v` |
| Docs | `mkdocs build --strict` |

CI runs all five on Python 3.11 and 3.12.

## Submitting changes

1. Fork the repo and create a feature branch from `main`.
2. Make your changes. Follow the existing code style (frozen dataclasses, `__all__` exports, `__slots__` on service classes, `Sequence`/`Mapping` in Protocol signatures).
3. Add or update tests — we target offline, deterministic unit tests.
4. Run the full quality-gate suite locally.
5. Open a pull request with a clear description of the change.

## Commit messages

Use clear, imperative-mood subject lines:

```
feat(ai): add token budget overflow metric
fix(policy_gateway): handle empty replacement string
docs: add model router user guide
```

## Architecture

ElectriPy follows a **hexagonal (ports & adapters)** architecture:

- `domain.py` — frozen value objects and enums
- `ports.py` — Protocol-based abstractions
- `adapters.py` — concrete implementations
- `services.py` — orchestration logic
- `errors.py` — module-specific exceptions

When adding a new module, mirror this structure.

## Reporting issues

Open a [GitHub issue](https://github.com/inference-stack-llc/electripy-studio/issues) with:

- A minimal reproduction or failing test case
- Python version and OS
- Full traceback (if applicable)

## Code of conduct

All contributors are expected to follow our [Code of Conduct](CODE_OF_CONDUCT.md).
