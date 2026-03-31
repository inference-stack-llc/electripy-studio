"""Adapter implementations for the skills packaging system.

Provides concrete implementations of the ports defined in
:mod:`electripy.ai.skills.ports`:

- **FileSystemAssetReader** — reads assets from the local filesystem.
- **FileSystemSkillLoader** — loads a skill from a directory with a
  ``manifest.json``.
- **DefaultSkillValidator** — validates manifests and asset references.
- **DefaultSkillResolver** — resolves instructions and renders templates.
- **MustacheStyleRenderer** — lightweight ``{{var}}`` template renderer.
- **InMemorySkillRegistry** — in-memory registry for testing.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from electripy.core.logging import get_logger

from .domain import (
    AssetKind,
    SkillAsset,
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
from .errors import (
    AssetResolutionError,
    ManifestLoadError,
    TemplateRenderError,
)

__all__ = [
    "DefaultSkillResolver",
    "DefaultSkillValidator",
    "FileSystemAssetReader",
    "FileSystemSkillLoader",
    "InMemorySkillRegistry",
    "MustacheStyleRenderer",
]

logger = get_logger(__name__)

_VAR_PATTERN = re.compile(r"\{\{(\w+)\}\}")

MANIFEST_FILENAME = "manifest.json"


# ── Asset reader ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class FileSystemAssetReader:
    """Reads skill asset files from the local filesystem."""

    encoding: str = "utf-8"

    def read(self, base_path: str, relative_path: str) -> str:
        """Read an asset file and return its text content.

        Raises:
            AssetResolutionError: If the file cannot be read.
        """
        full = Path(base_path) / relative_path
        # Prevent path traversal.
        try:
            full.resolve().relative_to(Path(base_path).resolve())
        except ValueError:
            raise AssetResolutionError(f"Path traversal detected: {relative_path!r}") from None
        if not full.is_file():
            raise AssetResolutionError(f"Asset not found: {relative_path!r} in {base_path}")
        try:
            return full.read_text(encoding=self.encoding)
        except OSError as exc:
            raise AssetResolutionError(f"Cannot read asset {relative_path!r}: {exc}") from exc

    def exists(self, base_path: str, relative_path: str) -> bool:
        """Check whether an asset file exists."""
        full = Path(base_path) / relative_path
        try:
            full.resolve().relative_to(Path(base_path).resolve())
        except ValueError:
            return False
        return full.is_file()


# ── Template renderer ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class MustacheStyleRenderer:
    """Lightweight ``{{variable}}`` template renderer.

    Consistent with the ElectriPy prompt_engine rendering convention.
    """

    def render(self, template: str, variables: dict[str, str]) -> str:
        """Render a template by replacing ``{{var}}`` placeholders.

        Raises:
            TemplateRenderError: If a placeholder has no matching variable.
        """

        def _replace(match: re.Match[str]) -> str:
            var = match.group(1)
            if var not in variables:
                raise TemplateRenderError(f"Unresolved template variable: {var!r}")
            return variables[var]

        return _VAR_PATTERN.sub(_replace, template)


# ── Manifest loading ─────────────────────────────────────────────────


def _parse_manifest(data: dict[str, Any], source: str) -> SkillManifest:
    """Parse a raw dict into a SkillManifest.

    Raises:
        ManifestLoadError: On missing/invalid fields.
    """
    name = data.get("name")
    if not name or not isinstance(name, str):
        raise ManifestLoadError(f"Manifest in {source} is missing 'name'")

    version_str = data.get("version")
    if not version_str or not isinstance(version_str, str):
        raise ManifestLoadError(f"Manifest in {source} is missing 'version'")
    try:
        version = SkillVersion.parse(version_str)
    except ValueError as exc:
        raise ManifestLoadError(f"Invalid version in {source}: {exc}") from exc

    # Assets
    assets: list[SkillAsset] = []
    for raw_asset in data.get("assets", []):
        if not isinstance(raw_asset, dict):
            raise ManifestLoadError(f"Invalid asset entry in {source}: {raw_asset!r}")
        asset_name = raw_asset.get("name", "")
        kind_str = raw_asset.get("kind", "data")
        try:
            kind = AssetKind(kind_str)
        except ValueError:
            raise ManifestLoadError(f"Invalid asset kind {kind_str!r} in {source}") from None
        assets.append(
            SkillAsset(
                name=asset_name,
                kind=kind,
                relative_path=raw_asset.get("path", ""),
                description=raw_asset.get("description", ""),
            )
        )

    # Dependencies
    deps: list[SkillDependency] = []
    for raw_dep in data.get("dependencies", []):
        if isinstance(raw_dep, str):
            deps.append(SkillDependency(skill_name=raw_dep))
        elif isinstance(raw_dep, dict):
            deps.append(
                SkillDependency(
                    skill_name=raw_dep.get("name", ""),
                    version_constraint=raw_dep.get("version", ""),
                )
            )

    # Metadata
    raw_meta = data.get("metadata", {})
    metadata = SkillMetadata(
        author=raw_meta.get("author", ""),
        license=raw_meta.get("license", ""),
        homepage=raw_meta.get("homepage", ""),
        capabilities=tuple(raw_meta.get("capabilities", [])),
        tags=tuple(raw_meta.get("tags", [])),
        min_python=raw_meta.get("min_python", ""),
        extra=tuple(
            (k, str(v))
            for k, v in raw_meta.items()
            if k not in {"author", "license", "homepage", "capabilities", "tags", "min_python"}
        ),
    )

    return SkillManifest(
        name=name,
        version=version,
        description=data.get("description", ""),
        entry_instruction=data.get("entry_instruction", ""),
        assets=tuple(assets),
        dependencies=tuple(deps),
        metadata=metadata,
        variables=tuple(data.get("variables", [])),
    )


# ── Skill loader ─────────────────────────────────────────────────────


@dataclass(slots=True)
class FileSystemSkillLoader:
    """Loads a skill package from a directory on disk.

    Expects a ``manifest.json`` at the root of the skill directory,
    and resolves the entry instruction and fragment assets.

    Attributes:
        asset_reader: Reader for individual asset files.
    """

    asset_reader: FileSystemAssetReader = field(default_factory=FileSystemAssetReader)

    def load(self, source: str) -> SkillPackage:
        """Load a skill from a directory path.

        Args:
            source: Path to the skill root directory.

        Returns:
            A fully loaded skill package.

        Raises:
            ManifestLoadError: If the manifest is invalid or missing.
            AssetResolutionError: If instruction files cannot be read.
        """
        root = Path(source).resolve()
        manifest_path = root / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ManifestLoadError(f"No {MANIFEST_FILENAME} found in {source}")

        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ManifestLoadError(f"Cannot parse {MANIFEST_FILENAME} in {source}: {exc}") from exc

        manifest = _parse_manifest(raw, source)
        instructions = self._load_instructions(manifest, str(root))

        logger.debug(
            "Loaded skill %s@%s from %s",
            manifest.name,
            manifest.version,
            source,
        )
        return SkillPackage(
            manifest=manifest,
            root_path=str(root),
            instructions=instructions,
        )

    def _load_instructions(
        self,
        manifest: SkillManifest,
        root: str,
    ) -> SkillInstructionSet:
        """Resolve entry instruction and instruction-kind assets."""
        entry = ""
        if manifest.entry_instruction:
            entry = self.asset_reader.read(root, manifest.entry_instruction)

        fragments: list[tuple[str, str]] = []
        for asset in manifest.assets:
            if asset.kind == AssetKind.INSTRUCTION and asset.relative_path:
                # Skip the entry instruction if also listed as an asset.
                if asset.relative_path == manifest.entry_instruction:
                    continue
                content = self.asset_reader.read(root, asset.relative_path)
                fragments.append((asset.name, content))

        return SkillInstructionSet(
            entry_instruction=entry,
            fragments=tuple(fragments),
        )

    def read_manifest(self, source: str) -> SkillManifest:
        """Read only the manifest from a skill directory.

        Args:
            source: Path to the skill root directory.

        Returns:
            The parsed skill manifest (without loading assets).

        Raises:
            ManifestLoadError: If the manifest is invalid or missing.
        """
        root = Path(source).resolve()
        manifest_path = root / MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise ManifestLoadError(f"No {MANIFEST_FILENAME} found in {source}")
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ManifestLoadError(f"Cannot parse {MANIFEST_FILENAME} in {source}: {exc}") from exc
        return _parse_manifest(raw, source)


# ── Skill validator ──────────────────────────────────────────────────


@dataclass(slots=True)
class DefaultSkillValidator:
    """Validates a skill manifest and its on-disk assets.

    Checks:
      - Required manifest fields (name, version, entry_instruction).
      - All declared asset files exist.
      - No duplicate asset names.
      - Entry instruction file exists.
      - Dependency names are non-empty.
      - Version is valid semver.
    """

    asset_reader: FileSystemAssetReader = field(default_factory=FileSystemAssetReader)

    def validate(
        self,
        manifest: SkillManifest,
        root_path: str,
    ) -> SkillValidationResult:
        """Validate a skill and return diagnostics."""
        diagnostics: list[ValidationDiagnostic] = []

        # Name
        if not manifest.name:
            diagnostics.append(
                ValidationDiagnostic(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_NAME",
                    message="Skill name is required.",
                    path="manifest.name",
                )
            )

        # Description
        if not manifest.description:
            diagnostics.append(
                ValidationDiagnostic(
                    severity=ValidationSeverity.WARNING,
                    code="MISSING_DESCRIPTION",
                    message="Skill description is recommended.",
                    path="manifest.description",
                )
            )

        # Entry instruction
        if not manifest.entry_instruction:
            diagnostics.append(
                ValidationDiagnostic(
                    severity=ValidationSeverity.ERROR,
                    code="MISSING_ENTRY_INSTRUCTION",
                    message="entry_instruction is required.",
                    path="manifest.entry_instruction",
                )
            )
        elif not self.asset_reader.exists(root_path, manifest.entry_instruction):
            diagnostics.append(
                ValidationDiagnostic(
                    severity=ValidationSeverity.ERROR,
                    code="ENTRY_INSTRUCTION_NOT_FOUND",
                    message=f"Entry instruction file not found: {manifest.entry_instruction!r}.",
                    path=manifest.entry_instruction,
                )
            )

        # Asset existence and uniqueness.
        seen_names: set[str] = set()
        for asset in manifest.assets:
            if asset.name in seen_names:
                diagnostics.append(
                    ValidationDiagnostic(
                        severity=ValidationSeverity.ERROR,
                        code="DUPLICATE_ASSET_NAME",
                        message=f"Duplicate asset name: {asset.name!r}.",
                        path=f"assets[{asset.name}]",
                    )
                )
            seen_names.add(asset.name)

            if not asset.relative_path:
                diagnostics.append(
                    ValidationDiagnostic(
                        severity=ValidationSeverity.ERROR,
                        code="MISSING_ASSET_PATH",
                        message=f"Asset {asset.name!r} has no path.",
                        path=f"assets[{asset.name}].path",
                    )
                )
            elif not self.asset_reader.exists(root_path, asset.relative_path):
                diagnostics.append(
                    ValidationDiagnostic(
                        severity=ValidationSeverity.ERROR,
                        code="ASSET_NOT_FOUND",
                        message=(
                            f"Asset file not found: {asset.relative_path!r} "
                            f"(asset {asset.name!r})."
                        ),
                        path=asset.relative_path,
                    )
                )

        # Dependencies
        for dep in manifest.dependencies:
            if not dep.skill_name:
                diagnostics.append(
                    ValidationDiagnostic(
                        severity=ValidationSeverity.ERROR,
                        code="EMPTY_DEPENDENCY_NAME",
                        message="Dependency has an empty skill_name.",
                        path="dependencies",
                    )
                )

        has_errors = any(d.severity == ValidationSeverity.ERROR for d in diagnostics)
        return SkillValidationResult(
            valid=not has_errors,
            diagnostics=tuple(diagnostics),
        )


# ── Skill resolver ───────────────────────────────────────────────────


@dataclass(slots=True)
class DefaultSkillResolver:
    """Resolves a skill's instructions and renders templates.

    Applies ``{{variable}}`` substitution on instruction content and
    template assets using the execution context.

    Attributes:
        renderer: Template renderer.
        asset_reader: Reader for template asset files.
    """

    renderer: MustacheStyleRenderer = field(default_factory=MustacheStyleRenderer)
    asset_reader: FileSystemAssetReader = field(default_factory=FileSystemAssetReader)

    def resolve(
        self,
        package: SkillPackage,
        context: SkillExecutionContext,
    ) -> SkillResolverResult:
        """Resolve instructions and render templates.

        Template variables are substituted using values from
        *context.variables*.  Unresolved variables are collected
        and reported (not raised) so callers can decide policy.
        """
        variables = dict(context.variables)

        # Render entry instruction.
        entry, entry_missing = self._safe_render(package.instructions.entry_instruction, variables)

        # Render fragments.
        rendered_fragments: list[tuple[str, str]] = []
        fragment_missing: list[str] = []
        for name, content in package.instructions.fragments:
            rendered, missing = self._safe_render(content, variables)
            rendered_fragments.append((name, rendered))
            fragment_missing.extend(missing)

        instructions = SkillInstructionSet(
            entry_instruction=entry,
            fragments=tuple(rendered_fragments),
        )

        # Render template assets.
        rendered_templates: list[tuple[str, str]] = []
        template_missing: list[str] = []
        for asset in package.manifest.assets:
            if asset.kind == AssetKind.TEMPLATE and asset.relative_path:
                raw = self.asset_reader.read(package.root_path, asset.relative_path)
                rendered, missing = self._safe_render(raw, variables)
                rendered_templates.append((asset.name, rendered))
                template_missing.extend(missing)

        all_missing = sorted(set(entry_missing + fragment_missing + template_missing))

        return SkillResolverResult(
            instructions=instructions,
            rendered_templates=tuple(rendered_templates),
            unresolved_variables=tuple(all_missing),
        )

    def _safe_render(
        self,
        template: str,
        variables: dict[str, str],
    ) -> tuple[str, list[str]]:
        """Render a template, collecting unresolved variable names."""
        missing: list[str] = []

        def _replace(match: re.Match[str]) -> str:
            var = match.group(1)
            if var in variables:
                return variables[var]
            missing.append(var)
            return match.group(0)  # Leave unresolved placeholders intact.

        rendered = _VAR_PATTERN.sub(_replace, template)
        return rendered, missing


# ── In-memory registry ───────────────────────────────────────────────


@dataclass(slots=True)
class InMemorySkillRegistry:
    """In-memory skill registry for testing and development."""

    _packages: dict[str, SkillPackage] = field(default_factory=dict)

    def register(self, package: SkillPackage) -> None:
        """Register a skill package by name."""
        self._packages[package.manifest.name] = package

    def get(self, name: str) -> SkillPackage | None:
        """Retrieve a registered skill by name."""
        return self._packages.get(name)

    def list_names(self) -> list[str]:
        """Return names of all registered skills."""
        return sorted(self._packages)
