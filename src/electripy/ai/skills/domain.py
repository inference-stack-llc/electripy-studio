"""Domain models for the skills packaging system.

All value objects are frozen dataclasses for immutability and
hashability.  Mutable containers use ``tuple`` instead of ``list``.

A *skill* is a versioned, portable unit of reusable AI behaviour —
instructions, assets, templates, dependencies and metadata bundled
into a directory with a ``manifest.json`` at the root.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

__all__ = [
    "AssetKind",
    "SkillAsset",
    "SkillCapability",
    "SkillDependency",
    "SkillExecutionContext",
    "SkillInstructionSet",
    "SkillManifest",
    "SkillMetadata",
    "SkillPackage",
    "SkillResolverResult",
    "SkillValidationResult",
    "SkillVersion",
    "ValidationDiagnostic",
    "ValidationSeverity",
]


# ── Enumerations ─────────────────────────────────────────────────────


class AssetKind(StrEnum):
    """Classification of a skill asset."""

    INSTRUCTION = "instruction"
    TEMPLATE = "template"
    CONFIG = "config"
    DATA = "data"
    EXAMPLE = "example"


class SkillCapability(StrEnum):
    """Well-known capability tags for skill profiles."""

    CHAT = "chat"
    EXTRACTION = "extraction"
    CLASSIFICATION = "classification"
    SUMMARIZATION = "summarization"
    CODE_GENERATION = "code_generation"
    TOOL_USE = "tool_use"
    RAG = "rag"
    REASONING = "reasoning"
    MULTIMODAL = "multimodal"


class ValidationSeverity(StrEnum):
    """Severity of a validation diagnostic."""

    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# ── Version ──────────────────────────────────────────────────────────

_SEMVER_RE = re.compile(
    r"^(?P<major>0|[1-9]\d*)\.(?P<minor>0|[1-9]\d*)\.(?P<patch>0|[1-9]\d*)"
    r"(?:-(?P<pre>[0-9A-Za-z\-.]+))?(?:\+(?P<build>[0-9A-Za-z\-.]+))?$"
)


@dataclass(frozen=True, slots=True)
class SkillVersion:
    """Semantic version for a skill.

    Attributes:
        major: Major version number.
        minor: Minor version number.
        patch: Patch version number.
        prerelease: Optional prerelease label (e.g. ``"beta.1"``).
        build: Optional build metadata.
    """

    major: int
    minor: int
    patch: int
    prerelease: str = ""
    build: str = ""

    # ── factories ────────────────────────────────────────────────────

    @classmethod
    def parse(cls, version_str: str) -> SkillVersion:
        """Parse a semantic version string.

        Args:
            version_str: A version string such as ``"1.2.3-beta.1"``.

        Returns:
            A parsed :class:`SkillVersion`.

        Raises:
            ValueError: If the string is not valid semver.
        """
        m = _SEMVER_RE.match(version_str.strip())
        if m is None:
            raise ValueError(f"Invalid semantic version: {version_str!r}")
        return cls(
            major=int(m.group("major")),
            minor=int(m.group("minor")),
            patch=int(m.group("patch")),
            prerelease=m.group("pre") or "",
            build=m.group("build") or "",
        )

    # ── display ──────────────────────────────────────────────────────

    def __str__(self) -> str:
        base = f"{self.major}.{self.minor}.{self.patch}"
        if self.prerelease:
            base += f"-{self.prerelease}"
        if self.build:
            base += f"+{self.build}"
        return base

    # ── comparison (ignores build metadata per semver spec) ──────────

    def _cmp_tuple(self) -> tuple[int, int, int, bool, str]:
        # Prerelease versions have *lower* precedence than release.
        # A release (empty prerelease) sorts *after* any prerelease.
        has_pre = bool(self.prerelease)
        return (self.major, self.minor, self.patch, not has_pre, self.prerelease)

    def __lt__(self, other: object) -> bool:
        if not isinstance(other, SkillVersion):
            return NotImplemented
        return self._cmp_tuple() < other._cmp_tuple()

    def __le__(self, other: object) -> bool:
        if not isinstance(other, SkillVersion):
            return NotImplemented
        return self._cmp_tuple() <= other._cmp_tuple()

    def __gt__(self, other: object) -> bool:
        if not isinstance(other, SkillVersion):
            return NotImplemented
        return self._cmp_tuple() > other._cmp_tuple()

    def __ge__(self, other: object) -> bool:
        if not isinstance(other, SkillVersion):
            return NotImplemented
        return self._cmp_tuple() >= other._cmp_tuple()


# ── Assets ───────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillAsset:
    """A single asset within a skill package.

    Attributes:
        name: Unique asset name within the skill (e.g. ``"main.md"``).
        kind: Classification of the asset.
        relative_path: Path relative to the skill root directory.
        description: Optional human-readable description.
    """

    name: str
    kind: AssetKind
    relative_path: str
    description: str = ""


@dataclass(frozen=True, slots=True)
class SkillInstructionSet:
    """Resolved instruction content for a skill.

    Attributes:
        entry_instruction: The primary instruction content.
        fragments: Named supplementary instruction fragments.
    """

    entry_instruction: str
    fragments: tuple[tuple[str, str], ...] = ()

    @property
    def full_text(self) -> str:
        """Concatenate entry instruction and all fragments."""
        parts = [self.entry_instruction]
        for _name, content in self.fragments:
            parts.append(content)
        return "\n\n".join(parts)

    def get_fragment(self, name: str) -> str | None:
        """Return a fragment by name, or ``None``."""
        for fname, content in self.fragments:
            if fname == name:
                return content
        return None


# ── Dependencies ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillDependency:
    """A dependency on another skill.

    Attributes:
        skill_name: Name of the required skill.
        version_constraint: Optional semver constraint (e.g. ``">=1.0.0"``).
    """

    skill_name: str
    version_constraint: str = ""


# ── Metadata ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillMetadata:
    """Metadata describing a skill's purpose and provenance.

    Attributes:
        author: Skill author or organisation.
        license: License identifier (e.g. ``"MIT"``).
        homepage: URL to documentation or repository.
        capabilities: Capability tags for discovery.
        tags: Arbitrary tags for filtering.
        min_python: Minimum Python version required (e.g. ``"3.11"``).
        extra: Arbitrary key-value metadata.
    """

    author: str = ""
    license: str = ""
    homepage: str = ""
    capabilities: tuple[str, ...] = ()
    tags: tuple[str, ...] = ()
    min_python: str = ""
    extra: tuple[tuple[str, str], ...] = ()


# ── Manifest ─────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillManifest:
    """Versioned manifest for a skill package.

    The manifest is the declarative definition of a skill — it lists
    all assets, dependencies, capability tags, and an entry
    instruction reference that a loader resolves.

    Attributes:
        name: Unique skill name (e.g. ``"code-review"``).
        version: Semantic version.
        description: Human-readable overview.
        entry_instruction: Relative path to the primary instruction file.
        assets: Declared assets.
        dependencies: Other skills this skill depends on.
        metadata: Extra metadata.
        variables: Variable names expected by templates.
    """

    name: str
    version: SkillVersion
    description: str = ""
    entry_instruction: str = ""
    assets: tuple[SkillAsset, ...] = ()
    dependencies: tuple[SkillDependency, ...] = ()
    metadata: SkillMetadata = field(default_factory=SkillMetadata)
    variables: tuple[str, ...] = ()


# ── Validation ───────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ValidationDiagnostic:
    """A single diagnostic from skill validation.

    Attributes:
        severity: Severity level.
        code: Machine-readable diagnostic code.
        message: Human-readable explanation.
        path: File or field path that caused the diagnostic.
    """

    severity: ValidationSeverity
    code: str
    message: str
    path: str = ""


@dataclass(frozen=True, slots=True)
class SkillValidationResult:
    """Result of validating a skill manifest and its assets.

    Attributes:
        valid: ``True`` when there are no error-level diagnostics.
        diagnostics: All diagnostics produced during validation.
    """

    valid: bool
    diagnostics: tuple[ValidationDiagnostic, ...] = ()

    @property
    def errors(self) -> tuple[ValidationDiagnostic, ...]:
        """Return only error-level diagnostics."""
        return tuple(d for d in self.diagnostics if d.severity == ValidationSeverity.ERROR)

    @property
    def warnings(self) -> tuple[ValidationDiagnostic, ...]:
        """Return only warning-level diagnostics."""
        return tuple(d for d in self.diagnostics if d.severity == ValidationSeverity.WARNING)


# ── Execution context ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillExecutionContext:
    """Context provided when preparing a skill for use.

    Attributes:
        variables: Template variable values.
        tags: Arbitrary context tags.
        metadata: Arbitrary runtime metadata.
    """

    variables: tuple[tuple[str, str], ...] = ()
    tags: tuple[str, ...] = ()
    metadata: tuple[tuple[str, str], ...] = ()

    def get_variable(self, name: str) -> str | None:
        """Look up a variable value by name."""
        for k, v in self.variables:
            if k == name:
                return v
        return None


# ── Full package ─────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillPackage:
    """A fully loaded skill with manifest and resolved content.

    Attributes:
        manifest: The skill manifest.
        root_path: Absolute path to the skill directory.
        instructions: Resolved instruction set.
        loaded_at: When the package was loaded.
    """

    manifest: SkillManifest
    root_path: str
    instructions: SkillInstructionSet = field(
        default_factory=lambda: SkillInstructionSet(entry_instruction="")
    )
    loaded_at: datetime = field(default_factory=lambda: datetime.now(tz=UTC))


# ── Resolver result ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SkillResolverResult:
    """Result of resolving a skill's assets and instructions.

    Attributes:
        instructions: The resolved instruction set.
        rendered_templates: Rendered template outputs (name → content).
        unresolved_variables: Variable names that could not be resolved.
    """

    instructions: SkillInstructionSet
    rendered_templates: tuple[tuple[str, str], ...] = ()
    unresolved_variables: tuple[str, ...] = ()
