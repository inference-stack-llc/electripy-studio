"""Tests for skills adapters."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from electripy.ai.skills import (
    AssetKind,
    AssetResolutionError,
    ManifestLoadError,
    SkillVersion,
    TemplateRenderError,
    ValidationSeverity,
)
from electripy.ai.skills.adapters import (
    DefaultSkillResolver,
    DefaultSkillValidator,
    FileSystemAssetReader,
    FileSystemSkillLoader,
    InMemorySkillRegistry,
    MustacheStyleRenderer,
)
from electripy.ai.skills.domain import (
    SkillAsset,
    SkillExecutionContext,
    SkillInstructionSet,
    SkillManifest,
    SkillPackage,
)


# ── FileSystemAssetReader ────────────────────────────────────────────


class TestFileSystemAssetReader:
    def test_read_existing_file(self, skill_dir: Path) -> None:
        reader = FileSystemAssetReader()
        content = reader.read(str(skill_dir), "instructions/main.md")
        assert "code reviewer" in content

    def test_read_missing_file_raises(self, skill_dir: Path) -> None:
        reader = FileSystemAssetReader()
        with pytest.raises(AssetResolutionError, match="not found"):
            reader.read(str(skill_dir), "nonexistent.md")

    def test_exists_true(self, skill_dir: Path) -> None:
        reader = FileSystemAssetReader()
        assert reader.exists(str(skill_dir), "instructions/main.md")

    def test_exists_false(self, skill_dir: Path) -> None:
        reader = FileSystemAssetReader()
        assert not reader.exists(str(skill_dir), "nonexistent.md")

    def test_path_traversal_blocked(self, skill_dir: Path) -> None:
        reader = FileSystemAssetReader()
        with pytest.raises(AssetResolutionError, match="traversal"):
            reader.read(str(skill_dir), "../../etc/passwd")

    def test_path_traversal_exists_returns_false(self, skill_dir: Path) -> None:
        reader = FileSystemAssetReader()
        assert not reader.exists(str(skill_dir), "../../etc/passwd")


# ── MustacheStyleRenderer ────────────────────────────────────────────


class TestMustacheStyleRenderer:
    def test_render_basic(self) -> None:
        renderer = MustacheStyleRenderer()
        result = renderer.render("Hello {{name}}!", {"name": "World"})
        assert result == "Hello World!"

    def test_render_multiple_vars(self) -> None:
        renderer = MustacheStyleRenderer()
        result = renderer.render("{{a}} and {{b}}", {"a": "X", "b": "Y"})
        assert result == "X and Y"

    def test_render_missing_var_raises(self) -> None:
        renderer = MustacheStyleRenderer()
        with pytest.raises(TemplateRenderError, match="Unresolved"):
            renderer.render("Hello {{name}}!", {})

    def test_render_no_placeholders(self) -> None:
        renderer = MustacheStyleRenderer()
        result = renderer.render("No vars here.", {})
        assert result == "No vars here."


# ── FileSystemSkillLoader ────────────────────────────────────────────


class TestFileSystemSkillLoader:
    def test_load_valid_skill(self, skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        pkg = loader.load(str(skill_dir))
        assert pkg.manifest.name == "code-review"
        assert pkg.manifest.version == SkillVersion(1, 2, 0)
        assert "code reviewer" in pkg.instructions.entry_instruction
        assert len(pkg.manifest.assets) == 3

    def test_load_reads_fragments(self, skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        pkg = loader.load(str(skill_dir))
        assert pkg.instructions.get_fragment("style-guide") is not None
        assert "naming conventions" in (pkg.instructions.get_fragment("style-guide") or "")

    def test_load_missing_manifest_raises(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty"
        empty.mkdir()
        loader = FileSystemSkillLoader()
        with pytest.raises(ManifestLoadError, match="No manifest.json"):
            loader.load(str(empty))

    def test_load_invalid_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "bad"
        bad.mkdir()
        (bad / "manifest.json").write_text("not json!", encoding="utf-8")
        loader = FileSystemSkillLoader()
        with pytest.raises(ManifestLoadError, match="Cannot parse"):
            loader.load(str(bad))

    def test_load_missing_name_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "noname"
        bad.mkdir()
        (bad / "manifest.json").write_text(
            json.dumps({"version": "1.0.0"}), encoding="utf-8"
        )
        loader = FileSystemSkillLoader()
        with pytest.raises(ManifestLoadError, match="missing 'name'"):
            loader.load(str(bad))

    def test_load_missing_version_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "noversion"
        bad.mkdir()
        (bad / "manifest.json").write_text(
            json.dumps({"name": "test"}), encoding="utf-8"
        )
        loader = FileSystemSkillLoader()
        with pytest.raises(ManifestLoadError, match="missing 'version'"):
            loader.load(str(bad))

    def test_load_invalid_version_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "badver"
        bad.mkdir()
        (bad / "manifest.json").write_text(
            json.dumps({"name": "test", "version": "nope"}), encoding="utf-8"
        )
        loader = FileSystemSkillLoader()
        with pytest.raises(ManifestLoadError, match="Invalid version"):
            loader.load(str(bad))

    def test_load_invalid_asset_kind_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "badkind"
        bad.mkdir()
        (bad / "manifest.json").write_text(
            json.dumps({
                "name": "test",
                "version": "1.0.0",
                "assets": [{"name": "x", "kind": "bogus", "path": "x.md"}],
            }),
            encoding="utf-8",
        )
        loader = FileSystemSkillLoader()
        with pytest.raises(ManifestLoadError, match="Invalid asset kind"):
            loader.load(str(bad))

    def test_load_minimal(self, minimal_skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        pkg = loader.load(str(minimal_skill_dir))
        assert pkg.manifest.name == "minimal"
        assert pkg.instructions.entry_instruction == "Do something."

    def test_read_manifest_only(self, skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        manifest = loader.read_manifest(str(skill_dir))
        assert manifest.name == "code-review"
        assert manifest.version == SkillVersion(1, 2, 0)

    def test_load_dependencies(self, skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        pkg = loader.load(str(skill_dir))
        assert len(pkg.manifest.dependencies) == 1
        assert pkg.manifest.dependencies[0].skill_name == "base-reviewer"
        assert pkg.manifest.dependencies[0].version_constraint == ">=1.0.0"

    def test_load_metadata(self, skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        pkg = loader.load(str(skill_dir))
        assert pkg.manifest.metadata.author == "ElectriPy Team"
        assert pkg.manifest.metadata.license == "MIT"
        assert "code_generation" in pkg.manifest.metadata.capabilities

    def test_load_variables(self, skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        pkg = loader.load(str(skill_dir))
        assert "code" in pkg.manifest.variables
        assert "reviewer" in pkg.manifest.variables


# ── DefaultSkillValidator ────────────────────────────────────────────


class TestDefaultSkillValidator:
    def test_valid_skill(self, skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        manifest = loader.read_manifest(str(skill_dir))
        validator = DefaultSkillValidator()
        result = validator.validate(manifest, str(skill_dir))
        assert result.valid

    def test_missing_entry_instruction(self, tmp_path: Path) -> None:
        manifest = SkillManifest(
            name="test",
            version=SkillVersion(1, 0, 0),
            entry_instruction="",
        )
        validator = DefaultSkillValidator()
        result = validator.validate(manifest, str(tmp_path))
        assert not result.valid
        codes = [d.code for d in result.errors]
        assert "MISSING_ENTRY_INSTRUCTION" in codes

    def test_entry_instruction_file_not_found(self, tmp_path: Path) -> None:
        manifest = SkillManifest(
            name="test",
            version=SkillVersion(1, 0, 0),
            entry_instruction="nonexistent.md",
        )
        validator = DefaultSkillValidator()
        result = validator.validate(manifest, str(tmp_path))
        assert not result.valid
        codes = [d.code for d in result.errors]
        assert "ENTRY_INSTRUCTION_NOT_FOUND" in codes

    def test_duplicate_asset_names(self, tmp_path: Path) -> None:
        (tmp_path / "main.md").write_text("entry", encoding="utf-8")
        (tmp_path / "a.md").write_text("asset", encoding="utf-8")
        manifest = SkillManifest(
            name="test",
            version=SkillVersion(1, 0, 0),
            entry_instruction="main.md",
            assets=(
                SkillAsset(name="dup", kind=AssetKind.DATA, relative_path="a.md"),
                SkillAsset(name="dup", kind=AssetKind.DATA, relative_path="a.md"),
            ),
        )
        validator = DefaultSkillValidator()
        result = validator.validate(manifest, str(tmp_path))
        assert not result.valid
        codes = [d.code for d in result.errors]
        assert "DUPLICATE_ASSET_NAME" in codes

    def test_asset_file_not_found(self, tmp_path: Path) -> None:
        (tmp_path / "main.md").write_text("entry", encoding="utf-8")
        manifest = SkillManifest(
            name="test",
            version=SkillVersion(1, 0, 0),
            entry_instruction="main.md",
            assets=(
                SkillAsset(name="ghost", kind=AssetKind.DATA, relative_path="ghost.dat"),
            ),
        )
        validator = DefaultSkillValidator()
        result = validator.validate(manifest, str(tmp_path))
        assert not result.valid
        codes = [d.code for d in result.errors]
        assert "ASSET_NOT_FOUND" in codes

    def test_missing_asset_path(self, tmp_path: Path) -> None:
        (tmp_path / "main.md").write_text("entry", encoding="utf-8")
        manifest = SkillManifest(
            name="test",
            version=SkillVersion(1, 0, 0),
            entry_instruction="main.md",
            assets=(
                SkillAsset(name="nopath", kind=AssetKind.DATA, relative_path=""),
            ),
        )
        validator = DefaultSkillValidator()
        result = validator.validate(manifest, str(tmp_path))
        assert not result.valid
        codes = [d.code for d in result.errors]
        assert "MISSING_ASSET_PATH" in codes

    def test_empty_dependency_name(self, tmp_path: Path) -> None:
        from electripy.ai.skills.domain import SkillDependency

        (tmp_path / "main.md").write_text("entry", encoding="utf-8")
        manifest = SkillManifest(
            name="test",
            version=SkillVersion(1, 0, 0),
            entry_instruction="main.md",
            dependencies=(SkillDependency(skill_name=""),),
        )
        validator = DefaultSkillValidator()
        result = validator.validate(manifest, str(tmp_path))
        assert not result.valid
        codes = [d.code for d in result.errors]
        assert "EMPTY_DEPENDENCY_NAME" in codes

    def test_missing_description_is_warning(self, minimal_skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        manifest = loader.read_manifest(str(minimal_skill_dir))
        validator = DefaultSkillValidator()
        result = validator.validate(manifest, str(minimal_skill_dir))
        # Missing description is a warning, not an error.
        assert result.valid
        codes = [d.code for d in result.warnings]
        assert "MISSING_DESCRIPTION" in codes

    def test_missing_name(self, tmp_path: Path) -> None:
        (tmp_path / "main.md").write_text("entry", encoding="utf-8")
        manifest = SkillManifest(
            name="",
            version=SkillVersion(1, 0, 0),
            entry_instruction="main.md",
        )
        validator = DefaultSkillValidator()
        result = validator.validate(manifest, str(tmp_path))
        assert not result.valid
        codes = [d.code for d in result.errors]
        assert "MISSING_NAME" in codes


# ── DefaultSkillResolver ─────────────────────────────────────────────


class TestDefaultSkillResolver:
    def test_resolve_replaces_variables(self, skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        pkg = loader.load(str(skill_dir))
        resolver = DefaultSkillResolver()

        ctx = SkillExecutionContext(
            variables=(
                ("code", "def add(a, b): return a + b"),
                ("reviewer", "Alice"),
                ("findings", "- Looks good\n- Consider type hints"),
            ),
        )
        result = resolver.resolve(pkg, ctx)
        assert "def add(a, b)" in result.instructions.entry_instruction
        assert result.unresolved_variables == ()

    def test_resolve_renders_templates(self, skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        pkg = loader.load(str(skill_dir))
        resolver = DefaultSkillResolver()

        ctx = SkillExecutionContext(
            variables=(
                ("code", "x = 1"),
                ("reviewer", "Bob"),
                ("findings", "All clear."),
            ),
        )
        result = resolver.resolve(pkg, ctx)
        templates = dict(result.rendered_templates)
        assert "Bob" in templates["report-template"]
        assert "All clear." in templates["report-template"]

    def test_resolve_reports_unresolved_vars(self, skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        pkg = loader.load(str(skill_dir))
        resolver = DefaultSkillResolver()

        ctx = SkillExecutionContext(variables=())
        result = resolver.resolve(pkg, ctx)
        assert "code" in result.unresolved_variables
        assert "reviewer" in result.unresolved_variables

    def test_resolve_empty_context(self, minimal_skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        pkg = loader.load(str(minimal_skill_dir))
        resolver = DefaultSkillResolver()

        result = resolver.resolve(pkg, SkillExecutionContext())
        assert result.instructions.entry_instruction == "Do something."
        assert result.unresolved_variables == ()

    def test_resolve_fragments(self, skill_dir: Path) -> None:
        loader = FileSystemSkillLoader()
        pkg = loader.load(str(skill_dir))
        resolver = DefaultSkillResolver()

        ctx = SkillExecutionContext(
            variables=(("code", "test"),("reviewer", "X"),("findings", "Y")),
        )
        result = resolver.resolve(pkg, ctx)
        # The style guide fragment has no variables, should pass through.
        frag = result.instructions.get_fragment("style-guide")
        assert frag is not None
        assert "naming conventions" in frag


# ── InMemorySkillRegistry ────────────────────────────────────────────


class TestInMemorySkillRegistry:
    def test_register_and_get(self) -> None:
        registry = InMemorySkillRegistry()
        pkg = SkillPackage(
            manifest=SkillManifest(name="test", version=SkillVersion(1, 0, 0)),
            root_path="/tmp",
        )
        registry.register(pkg)
        assert registry.get("test") is pkg

    def test_get_missing_returns_none(self) -> None:
        registry = InMemorySkillRegistry()
        assert registry.get("nonexistent") is None

    def test_list_names(self) -> None:
        registry = InMemorySkillRegistry()
        for name in ("beta", "alpha", "gamma"):
            pkg = SkillPackage(
                manifest=SkillManifest(name=name, version=SkillVersion(1, 0, 0)),
                root_path="/tmp",
            )
            registry.register(pkg)
        assert registry.list_names() == ["alpha", "beta", "gamma"]
