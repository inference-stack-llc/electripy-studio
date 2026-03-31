"""Tests for electripy.ai.workload_router.errors."""

from __future__ import annotations

import pytest

from electripy.ai.workload_router.errors import (
    BudgetExceededError,
    ConstraintViolationError,
    NoCandidateError,
    WorkloadRouterError,
)
from electripy.core.errors import ElectriPyError


class TestErrorHierarchy:
    def test_base_extends_electripy(self) -> None:
        assert issubclass(WorkloadRouterError, ElectriPyError)

    def test_no_candidate_error(self) -> None:
        assert issubclass(NoCandidateError, WorkloadRouterError)

    def test_constraint_violation_error(self) -> None:
        assert issubclass(ConstraintViolationError, WorkloadRouterError)

    def test_budget_exceeded_error(self) -> None:
        assert issubclass(BudgetExceededError, WorkloadRouterError)


class TestErrorInstances:
    def test_no_candidate_str(self) -> None:
        exc = NoCandidateError("no models available")
        assert "no models" in str(exc)

    def test_budget_exceeded_str(self) -> None:
        exc = BudgetExceededError("all exceed $0.01/1k")
        assert "$0.01" in str(exc)

    def test_catch_as_electripy_error(self) -> None:
        with pytest.raises(ElectriPyError):
            raise NoCandidateError("test")

    def test_catch_as_base(self) -> None:
        with pytest.raises(WorkloadRouterError):
            raise BudgetExceededError("test")
