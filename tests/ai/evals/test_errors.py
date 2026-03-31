"""Tests for electripy.ai.evals.errors."""

from __future__ import annotations

import pytest

from electripy.ai.evals.errors import (
    DatasetLoadError,
    EvalError,
    RegressionError,
    ScorerError,
)
from electripy.core.errors import ElectriPyError


class TestErrorHierarchy:
    def test_base_extends_electripy(self) -> None:
        assert issubclass(EvalError, ElectriPyError)

    def test_dataset_load_error(self) -> None:
        assert issubclass(DatasetLoadError, EvalError)

    def test_scorer_error(self) -> None:
        assert issubclass(ScorerError, EvalError)

    def test_regression_error(self) -> None:
        assert issubclass(RegressionError, EvalError)


class TestErrorInstances:
    def test_dataset_load_str(self) -> None:
        exc = DatasetLoadError("file not found")
        assert "file not found" in str(exc)

    def test_scorer_error_str(self) -> None:
        exc = ScorerError("scorer failed")
        assert "scorer failed" in str(exc)

    def test_regression_error_str(self) -> None:
        exc = RegressionError("accuracy dropped")
        assert "accuracy" in str(exc)

    def test_catch_as_electripy_error(self) -> None:
        with pytest.raises(ElectriPyError):
            raise DatasetLoadError("test")

    def test_catch_as_eval_error(self) -> None:
        with pytest.raises(EvalError):
            raise RegressionError("test")
