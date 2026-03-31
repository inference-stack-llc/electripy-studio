"""Tests for skills errors."""

from __future__ import annotations

from electripy.ai.skills import (
    AssetResolutionError,
    ManifestLoadError,
    SkillError,
    SkillNotFoundError,
    SkillValidationError,
    TemplateRenderError,
)
from electripy.core.errors import ElectriPyError


class TestErrorHierarchy:
    def test_base_inherits_electripy_error(self) -> None:
        assert issubclass(SkillError, ElectriPyError)

    def test_all_errors_inherit_base(self) -> None:
        for cls in (
            ManifestLoadError,
            AssetResolutionError,
            SkillValidationError,
            SkillNotFoundError,
            TemplateRenderError,
        ):
            assert issubclass(cls, SkillError)

    def test_errors_carry_message(self) -> None:
        err = ManifestLoadError("bad manifest")
        assert str(err) == "bad manifest"

    def test_errors_are_catchable_as_electripy(self) -> None:
        try:
            raise AssetResolutionError("file not found")
        except ElectriPyError as exc:
            assert "file not found" in str(exc)
