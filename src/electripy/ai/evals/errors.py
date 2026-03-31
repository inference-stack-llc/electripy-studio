"""Exception hierarchy for the evaluation framework.

All evaluation exceptions extend :class:`EvalError` which itself
derives from :class:`ElectriPyError`, keeping the error hierarchy
consistent with the rest of the ElectriPy codebase.

Example::

    from electripy.ai.evals.errors import DatasetLoadError

    try:
        dataset = loader.load("missing.jsonl")
    except DatasetLoadError as exc:
        print(f"Could not load dataset: {exc}")
"""

from __future__ import annotations

from electripy.core.errors import ElectriPyError

__all__ = [
    "DatasetLoadError",
    "EvalError",
    "RegressionError",
    "ScorerError",
]


class EvalError(ElectriPyError):
    """Base exception for evaluation framework failures."""


class DatasetLoadError(EvalError):
    """Raised when a dataset cannot be loaded or parsed."""


class ScorerError(EvalError):
    """Raised when a scorer encounters an unrecoverable error."""


class RegressionError(EvalError):
    """Raised when a regression comparison fails a CI gate."""
