"""Tests for SkillService and convenience functions."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest

from electripy.ai.skills import (
    SkillNotFoundError,
    SkillValidationError,
)
from electripy.ai.skills.domain import (
    SkillExecutionContext,
    SkillManifest,
    SkillPackage,
    SkillResolverResult,
    SkillValidationResult,
)
from electripy.ai.skills.services import (
    SkillService,
    get_entry_instructions,
    list_skills,
    load_skill,
    read_skill_manifest,
    resolve_skill,
    validate_skill,
)

# ── Observer stub ────────────────────────────────────────────────────


@dataclass(slots=True)
class _StubObserver:
    loaded: list[str] = field(default_factory=list)
    validated: list[str] = field(default_factory=list)
    resolved: list[str] = field(default_factory=list)

    def on_load(self, package: SkillPackage) -> None:
        self.loaded.append(package.manifest.name)

    def on_validate(
        self,
        manifest: SkillManifest,
        result: SkillValidationResult,
    ) -> None:
        self.validated.append(manifest.name)

    def on_resolve(
        self,
        package: SkillPackage,
        result: SkillResolverResult,
    ) -> None:
        self.resolved.append(package.manifest.name)


# ── SkillService ─────────────────────────────────────────────────────


class TestSkillServiceLoad:
    def test_load_registers(self, skill_dir: Path) -> None:
        svc = SkillService()
        pkg = svc.load(str(skill_dir))
        assert svc.registry.get("code-review") is pkg

    def test_load_notifies_observer(self, skill_dir: Path) -> None:
        obs = _StubObserver()
        svc = SkillService(observer=obs)
        svc.load(str(skill_dir))
        assert obs.loaded == ["code-review"]


class TestSkillServiceValidate:
    def test_validate_valid(self, skill_dir: Path) -> None:
        svc = SkillService()
        result = svc.validate(str(skill_dir))
        assert result.valid

    def test_validate_fail_on_error(self, tmp_path: Path) -> None:
        # Create a skill whose entry instruction file is missing.
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "manifest.json").write_text(
            '{"name":"bad","version":"1.0.0","entry_instruction":"missing.md"}',
            encoding="utf-8",
        )
        svc = SkillService()
        with pytest.raises(SkillValidationError, match="validation failed"):
            svc.validate(str(bad), fail_on_error=True)

    def test_validate_notifies_observer(self, skill_dir: Path) -> None:
        obs = _StubObserver()
        svc = SkillService(observer=obs)
        svc.validate(str(skill_dir))
        assert obs.validated == ["code-review"]


class TestSkillServiceResolve:
    def test_resolve_basic(self, skill_dir: Path) -> None:
        svc = SkillService()
        pkg = svc.load(str(skill_dir))
        ctx = SkillExecutionContext(
            variables=(("code", "x=1"), ("reviewer", "A"), ("findings", "ok")),
        )
        result = svc.resolve(pkg, ctx)
        assert "x=1" in result.instructions.entry_instruction
        assert result.unresolved_variables == ()

    def test_resolve_notifies_observer(self, skill_dir: Path) -> None:
        obs = _StubObserver()
        svc = SkillService(observer=obs)
        pkg = svc.load(str(skill_dir))
        svc.resolve(pkg, SkillExecutionContext())
        assert obs.resolved == ["code-review"]


class TestSkillServiceListDirectory:
    def test_list_skills_in_directory(self, skills_parent: Path) -> None:
        svc = SkillService()
        manifests = svc.list_skills_in_directory(str(skills_parent))
        names = sorted(m.name for m in manifests)
        assert names == ["code-review", "minimal"]

    def test_list_skills_in_nonexistent_directory(self, tmp_path: Path) -> None:
        svc = SkillService()
        assert svc.list_skills_in_directory(str(tmp_path / "nope")) == []

    def test_list_skips_invalid(self, skills_parent: Path) -> None:
        # Add a directory with an invalid manifest.
        broken = skills_parent / "broken"
        broken.mkdir()
        (broken / "manifest.json").write_text("bad json", encoding="utf-8")
        svc = SkillService()
        manifests = svc.list_skills_in_directory(str(skills_parent))
        names = [m.name for m in manifests]
        assert "broken" not in names
        assert "code-review" in names


class TestSkillServiceGetRegistered:
    def test_get_registered_existing(self, skill_dir: Path) -> None:
        svc = SkillService()
        svc.load(str(skill_dir))
        pkg = svc.get_registered("code-review")
        assert pkg.manifest.name == "code-review"

    def test_get_registered_missing(self) -> None:
        svc = SkillService()
        with pytest.raises(SkillNotFoundError, match="not found"):
            svc.get_registered("nonexistent")


class TestSkillServiceReadManifest:
    def test_read_manifest(self, skill_dir: Path) -> None:
        svc = SkillService()
        m = svc.read_manifest(str(skill_dir))
        assert m.name == "code-review"


class TestSkillServiceGetEntryInstructions:
    def test_get_entry_instructions(self, skill_dir: Path) -> None:
        svc = SkillService()
        pkg = svc.load(str(skill_dir))
        text = svc.get_entry_instructions(pkg)
        assert "code reviewer" in text


# ── Convenience functions ────────────────────────────────────────────


class TestConvenienceFunctions:
    def test_load_skill(self, skill_dir: Path) -> None:
        pkg = load_skill(str(skill_dir))
        assert pkg.manifest.name == "code-review"

    def test_validate_skill(self, skill_dir: Path) -> None:
        result = validate_skill(str(skill_dir))
        assert result.valid

    def test_resolve_skill(self, skill_dir: Path) -> None:
        pkg = load_skill(str(skill_dir))
        ctx = SkillExecutionContext(
            variables=(("code", "y"), ("reviewer", "B"), ("findings", "done")),
        )
        result = resolve_skill(pkg, ctx)
        assert "y" in result.instructions.entry_instruction

    def test_list_skills(self, skills_parent: Path) -> None:
        manifests = list_skills(str(skills_parent))
        assert len(manifests) >= 1

    def test_read_skill_manifest(self, skill_dir: Path) -> None:
        m = read_skill_manifest(str(skill_dir))
        assert m.name == "code-review"

    def test_get_entry_instructions(self, skill_dir: Path) -> None:
        pkg = load_skill(str(skill_dir))
        text = get_entry_instructions(pkg)
        assert "code reviewer" in text
