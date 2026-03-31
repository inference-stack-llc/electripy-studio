"""Skill service — the primary orchestration entry-point.

The :class:`SkillService` wires loader, validator, resolver, registry,
and observer ports into a single facade.  Convenience functions
(``load_skill``, ``validate_skill``, etc.) provide ergonomic access
for common use-cases.

Example::

    from electripy.ai.skills import load_skill, validate_skill

    result = validate_skill("./skills/code-review")
    if result.valid:
        pkg = load_skill("./skills/code-review")
        print(pkg.instructions.entry_instruction)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from electripy.core.logging import get_logger

from .adapters import (
    DefaultSkillResolver,
    DefaultSkillValidator,
    FileSystemSkillLoader,
    InMemorySkillRegistry,
)
from .domain import (
    SkillExecutionContext,
    SkillManifest,
    SkillPackage,
    SkillResolverResult,
    SkillValidationResult,
)
from .errors import SkillNotFoundError, SkillValidationError
from .ports import (
    SkillLoaderPort,
    SkillObserverPort,
    SkillRegistryPort,
    SkillResolverPort,
    SkillValidatorPort,
)

__all__ = [
    "SkillService",
    "get_entry_instructions",
    "list_skills",
    "load_skill",
    "read_skill_manifest",
    "resolve_skill",
    "validate_skill",
]

logger = get_logger(__name__)


@dataclass(slots=True)
class SkillService:
    """Primary orchestrator for skill loading, validation, and resolution.

    Attributes:
        loader: Loads skill packages from disk.
        validator: Validates manifests and assets.
        resolver: Resolves instructions and renders templates.
        registry: Stores and retrieves loaded skills.
        observer: Optional lifecycle observer for instrumentation.
    """

    loader: SkillLoaderPort = field(default_factory=FileSystemSkillLoader)
    validator: SkillValidatorPort = field(default_factory=DefaultSkillValidator)
    resolver: SkillResolverPort = field(default_factory=DefaultSkillResolver)
    registry: SkillRegistryPort = field(default_factory=InMemorySkillRegistry)
    observer: SkillObserverPort | None = None

    # ── public API ───────────────────────────────────────────────────

    def load(self, source: str) -> SkillPackage:
        """Load a skill from disk and register it.

        Args:
            source: Path to the skill root directory.

        Returns:
            The loaded skill package.
        """
        package = self.loader.load(source)
        self.registry.register(package)
        if self.observer is not None:
            self.observer.on_load(package)
        logger.info(
            "Loaded skill %s@%s",
            package.manifest.name,
            package.manifest.version,
        )
        return package

    def validate(
        self,
        source: str,
        *,
        fail_on_error: bool = False,
    ) -> SkillValidationResult:
        """Validate a skill directory.

        Args:
            source: Path to the skill root directory.
            fail_on_error: If ``True``, raise on error-level diagnostics.

        Returns:
            Validation result with diagnostics.

        Raises:
            SkillValidationError: If *fail_on_error* is ``True`` and
                validation produced error-level diagnostics.
        """
        # Load just the manifest for validation.
        loader = self.loader
        if hasattr(loader, "read_manifest"):
            manifest = loader.read_manifest(source)
        else:
            manifest = loader.load(source).manifest

        root_path = str(Path(source).resolve())
        result = self.validator.validate(manifest, root_path)

        if self.observer is not None:
            self.observer.on_validate(manifest, result)

        if fail_on_error and not result.valid:
            messages = "; ".join(d.message for d in result.errors)
            raise SkillValidationError(f"Skill validation failed: {messages}")

        logger.debug(
            "Validated skill %s: valid=%s diagnostics=%d",
            manifest.name,
            result.valid,
            len(result.diagnostics),
        )
        return result

    def resolve(
        self,
        package: SkillPackage,
        context: SkillExecutionContext | None = None,
    ) -> SkillResolverResult:
        """Resolve a skill's instructions and templates.

        Args:
            package: A loaded skill package.
            context: Execution context (defaults to empty context).

        Returns:
            Resolved instructions and rendered templates.
        """
        ctx = context or SkillExecutionContext()
        result = self.resolver.resolve(package, ctx)
        if self.observer is not None:
            self.observer.on_resolve(package, result)
        return result

    def list_skills_in_directory(self, directory: str) -> list[SkillManifest]:
        """Discover skills in a directory.

        Scans immediate subdirectories of *directory* for
        ``manifest.json`` files and returns their manifests.

        Args:
            directory: Parent directory containing skill directories.

        Returns:
            List of skill manifests found.
        """
        results: list[SkillManifest] = []
        parent = Path(directory)
        if not parent.is_dir():
            return results

        for child in sorted(parent.iterdir()):
            if not child.is_dir():
                continue
            manifest_path = child / "manifest.json"
            if manifest_path.is_file():
                try:
                    if hasattr(self.loader, "read_manifest"):
                        manifest = self.loader.read_manifest(str(child))
                    else:
                        manifest = self.loader.load(str(child)).manifest
                    results.append(manifest)
                except Exception:
                    logger.warning("Skipping invalid skill in %s", child)
        return results

    def get_registered(self, name: str) -> SkillPackage:
        """Retrieve a registered skill by name.

        Args:
            name: Skill name.

        Returns:
            The registered skill package.

        Raises:
            SkillNotFoundError: If no skill with that name is registered.
        """
        package = self.registry.get(name)
        if package is None:
            raise SkillNotFoundError(f"Skill not found: {name!r}")
        return package

    def read_manifest(self, source: str) -> SkillManifest:
        """Read only the manifest from a skill directory.

        Args:
            source: Path to the skill root directory.

        Returns:
            The parsed skill manifest.
        """
        if hasattr(self.loader, "read_manifest"):
            return self.loader.read_manifest(source)  # type: ignore[no-any-return]
        return self.loader.load(source).manifest

    def get_entry_instructions(self, package: SkillPackage) -> str:
        """Return the full resolved entry instruction text.

        Args:
            package: A loaded skill package.

        Returns:
            The entry instruction content.
        """
        return package.instructions.entry_instruction


# ── Convenience functions ────────────────────────────────────────────


def _default_service() -> SkillService:
    return SkillService()


def load_skill(source: str) -> SkillPackage:
    """Load a skill package from a directory.

    Args:
        source: Path to the skill root directory.

    Returns:
        The loaded skill package.
    """
    return _default_service().load(source)


def validate_skill(
    source: str,
    *,
    fail_on_error: bool = False,
) -> SkillValidationResult:
    """Validate a skill directory.

    Args:
        source: Path to the skill root directory.
        fail_on_error: If ``True``, raise on error diagnostics.

    Returns:
        Validation result with diagnostics.
    """
    return _default_service().validate(source, fail_on_error=fail_on_error)


def resolve_skill(
    package: SkillPackage,
    context: SkillExecutionContext | None = None,
) -> SkillResolverResult:
    """Resolve a skill's instructions and render templates.

    Args:
        package: A loaded skill package.
        context: Optional execution context.

    Returns:
        Resolved instructions and rendered templates.
    """
    return _default_service().resolve(package, context)


def list_skills(directory: str) -> list[SkillManifest]:
    """Discover skills in a directory.

    Args:
        directory: Parent directory containing skill directories.

    Returns:
        List of skill manifests found.
    """
    return _default_service().list_skills_in_directory(directory)


def read_skill_manifest(source: str) -> SkillManifest:
    """Read only the manifest from a skill directory.

    Args:
        source: Path to the skill root directory.

    Returns:
        The parsed manifest.
    """
    return _default_service().read_manifest(source)


def get_entry_instructions(package: SkillPackage) -> str:
    """Return the entry instruction content from a loaded package.

    Args:
        package: A loaded skill package.

    Returns:
        The entry instruction text.
    """
    return package.instructions.entry_instruction
