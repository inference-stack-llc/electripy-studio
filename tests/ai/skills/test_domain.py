"""Tests for skills domain models."""

from __future__ import annotations

from electripy.ai.skills import (
    AssetKind,
    SkillAsset,
    SkillCapability,
    SkillDependency,
    SkillExecutionContext,
    SkillInstructionSet,
    SkillManifest,
    SkillMetadata,
    SkillPackage,
    SkillResolverResult,
    SkillValidationResult,
    SkillVersion,
    ValidationDiagnostic,
    ValidationSeverity,
)


class TestSkillVersion:
    def test_parse_basic(self) -> None:
        v = SkillVersion.parse("1.2.3")
        assert v.major == 1
        assert v.minor == 2
        assert v.patch == 3
        assert v.prerelease == ""
        assert v.build == ""

    def test_parse_prerelease(self) -> None:
        v = SkillVersion.parse("2.0.0-beta.1")
        assert v.prerelease == "beta.1"

    def test_parse_build_metadata(self) -> None:
        v = SkillVersion.parse("1.0.0+build.42")
        assert v.build == "build.42"

    def test_parse_full(self) -> None:
        v = SkillVersion.parse("1.0.0-rc.1+build.42")
        assert v.prerelease == "rc.1"
        assert v.build == "build.42"

    def test_parse_invalid_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError, match="Invalid semantic version"):
            SkillVersion.parse("not-a-version")

    def test_parse_incomplete_raises(self) -> None:
        import pytest

        with pytest.raises(ValueError):
            SkillVersion.parse("1.2")

    def test_str_basic(self) -> None:
        assert str(SkillVersion(1, 2, 3)) == "1.2.3"

    def test_str_prerelease(self) -> None:
        assert str(SkillVersion(1, 0, 0, prerelease="alpha")) == "1.0.0-alpha"

    def test_str_build(self) -> None:
        assert str(SkillVersion(1, 0, 0, build="abc")) == "1.0.0+abc"

    def test_str_roundtrip(self) -> None:
        original = "2.1.0-beta.3+build.99"
        assert str(SkillVersion.parse(original)) == original

    def test_ordering_major(self) -> None:
        assert SkillVersion(2, 0, 0) > SkillVersion(1, 9, 9)

    def test_ordering_minor(self) -> None:
        assert SkillVersion(1, 2, 0) > SkillVersion(1, 1, 9)

    def test_ordering_patch(self) -> None:
        assert SkillVersion(1, 0, 2) > SkillVersion(1, 0, 1)

    def test_prerelease_lower_than_release(self) -> None:
        assert SkillVersion(1, 0, 0, prerelease="beta") < SkillVersion(1, 0, 0)

    def test_ordering_equal(self) -> None:
        a = SkillVersion(1, 0, 0)
        b = SkillVersion(1, 0, 0)
        assert a <= b
        assert a >= b
        assert not a < b

    def test_immutability(self) -> None:
        v = SkillVersion(1, 0, 0)
        try:
            v.major = 2  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


class TestSkillAsset:
    def test_creation(self) -> None:
        asset = SkillAsset(
            name="main.md",
            kind=AssetKind.INSTRUCTION,
            relative_path="instructions/main.md",
            description="Entry point",
        )
        assert asset.name == "main.md"
        assert asset.kind == AssetKind.INSTRUCTION
        assert asset.relative_path == "instructions/main.md"

    def test_all_asset_kinds(self) -> None:
        kinds = list(AssetKind)
        assert len(kinds) == 5
        assert AssetKind.INSTRUCTION in kinds
        assert AssetKind.TEMPLATE in kinds


class TestSkillInstructionSet:
    def test_full_text(self) -> None:
        iset = SkillInstructionSet(
            entry_instruction="Main instruction.",
            fragments=(("style", "Style guide."),),
        )
        assert "Main instruction." in iset.full_text
        assert "Style guide." in iset.full_text

    def test_get_fragment(self) -> None:
        iset = SkillInstructionSet(
            entry_instruction="Entry.",
            fragments=(("a", "Fragment A"), ("b", "Fragment B")),
        )
        assert iset.get_fragment("a") == "Fragment A"
        assert iset.get_fragment("b") == "Fragment B"
        assert iset.get_fragment("c") is None


class TestSkillDependency:
    def test_creation(self) -> None:
        dep = SkillDependency(skill_name="base-reviewer", version_constraint=">=1.0.0")
        assert dep.skill_name == "base-reviewer"
        assert dep.version_constraint == ">=1.0.0"

    def test_default_constraint(self) -> None:
        dep = SkillDependency(skill_name="other")
        assert dep.version_constraint == ""


class TestSkillMetadata:
    def test_creation(self) -> None:
        meta = SkillMetadata(
            author="Test Author",
            license="MIT",
            capabilities=("chat", "rag"),
            tags=("utility",),
        )
        assert meta.author == "Test Author"
        assert len(meta.capabilities) == 2


class TestSkillManifest:
    def test_creation(self) -> None:
        manifest = SkillManifest(
            name="test-skill",
            version=SkillVersion(1, 0, 0),
            description="A test skill",
            entry_instruction="main.md",
        )
        assert manifest.name == "test-skill"
        assert manifest.version.major == 1
        assert manifest.assets == ()
        assert manifest.dependencies == ()


class TestSkillCapability:
    def test_all_capabilities(self) -> None:
        caps = list(SkillCapability)
        assert len(caps) == 9
        assert SkillCapability.CHAT.value == "chat"
        assert SkillCapability.RAG.value == "rag"


class TestValidationDiagnostic:
    def test_creation(self) -> None:
        d = ValidationDiagnostic(
            severity=ValidationSeverity.ERROR,
            code="MISSING_NAME",
            message="Name is required.",
            path="manifest.name",
        )
        assert d.severity == ValidationSeverity.ERROR
        assert d.code == "MISSING_NAME"


class TestSkillValidationResult:
    def test_valid_result(self) -> None:
        result = SkillValidationResult(valid=True)
        assert result.valid
        assert result.errors == ()
        assert result.warnings == ()

    def test_result_with_errors(self) -> None:
        result = SkillValidationResult(
            valid=False,
            diagnostics=(
                ValidationDiagnostic(
                    severity=ValidationSeverity.ERROR,
                    code="E1",
                    message="Error 1",
                ),
                ValidationDiagnostic(
                    severity=ValidationSeverity.WARNING,
                    code="W1",
                    message="Warning 1",
                ),
            ),
        )
        assert not result.valid
        assert len(result.errors) == 1
        assert len(result.warnings) == 1


class TestSkillExecutionContext:
    def test_get_variable(self) -> None:
        ctx = SkillExecutionContext(
            variables=(("code", "print('hi')"), ("name", "test")),
        )
        assert ctx.get_variable("code") == "print('hi')"
        assert ctx.get_variable("name") == "test"
        assert ctx.get_variable("missing") is None


class TestSkillPackage:
    def test_creation(self) -> None:
        pkg = SkillPackage(
            manifest=SkillManifest(
                name="test",
                version=SkillVersion(1, 0, 0),
            ),
            root_path="/tmp/test",
        )
        assert pkg.manifest.name == "test"
        assert pkg.root_path == "/tmp/test"
        assert pkg.loaded_at is not None


class TestSkillResolverResult:
    def test_creation(self) -> None:
        result = SkillResolverResult(
            instructions=SkillInstructionSet(entry_instruction="Hello"),
            rendered_templates=(("report", "# Report"),),
            unresolved_variables=("missing_var",),
        )
        assert result.instructions.entry_instruction == "Hello"
        assert len(result.rendered_templates) == 1
        assert "missing_var" in result.unresolved_variables
