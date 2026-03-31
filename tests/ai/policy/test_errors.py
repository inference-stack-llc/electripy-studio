"""Tests for policy engine errors."""

from __future__ import annotations

from electripy.ai.policy import (
    ApprovalError,
    EscalationError,
    EvidenceError,
    PolicyEngineError,
    PolicyPackError,
)
from electripy.core.errors import ElectriPyError


class TestErrorHierarchy:
    def test_base_inherits_electripy_error(self) -> None:
        assert issubclass(PolicyEngineError, ElectriPyError)

    def test_all_errors_inherit_base(self) -> None:
        for cls in (PolicyPackError, EvidenceError, ApprovalError, EscalationError):
            assert issubclass(cls, PolicyEngineError)

    def test_errors_carry_message(self) -> None:
        err = ApprovalError("token expired")
        assert str(err) == "token expired"

    def test_errors_are_catchable_as_electripy(self) -> None:
        try:
            raise EvidenceError("missing justification")
        except ElectriPyError as exc:
            assert "missing justification" in str(exc)
