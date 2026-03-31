"""Ports (Protocols) for the skills packaging system.

All ports are runtime-checkable Protocols so adapters can be
substituted without inheritance coupling.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from .domain import (
    SkillExecutionContext,
    SkillManifest,
    SkillPackage,
    SkillResolverResult,
    SkillValidationResult,
)

__all__ = [
    "AssetReaderPort",
    "SkillLoaderPort",
    "SkillObserverPort",
    "SkillRegistryPort",
    "SkillResolverPort",
    "SkillValidatorPort",
    "TemplateRendererPort",
]


@runtime_checkable
class SkillLoaderPort(Protocol):
    """Loads skill packages from a source."""

    def load(self, source: str) -> SkillPackage:
        """Load a skill from the given source path.

        Args:
            source: Path to the skill root directory.

        Returns:
            A fully loaded skill package.
        """
        ...


@runtime_checkable
class AssetReaderPort(Protocol):
    """Reads individual asset files."""

    def read(self, base_path: str, relative_path: str) -> str:
        """Read an asset file and return its text content.

        Args:
            base_path: Root directory of the skill.
            relative_path: Path relative to the skill root.

        Returns:
            The text content of the asset.
        """
        ...

    def exists(self, base_path: str, relative_path: str) -> bool:
        """Check whether an asset file exists.

        Args:
            base_path: Root directory of the skill.
            relative_path: Path relative to the skill root.

        Returns:
            ``True`` if the file exists.
        """
        ...


@runtime_checkable
class SkillValidatorPort(Protocol):
    """Validates a skill manifest and its assets."""

    def validate(
        self,
        manifest: SkillManifest,
        root_path: str,
    ) -> SkillValidationResult:
        """Validate a skill and return diagnostics.

        Args:
            manifest: The skill manifest.
            root_path: Path to the skill root directory.

        Returns:
            Validation result with diagnostics.
        """
        ...


@runtime_checkable
class SkillResolverPort(Protocol):
    """Resolves a skill's instructions and templates."""

    def resolve(
        self,
        package: SkillPackage,
        context: SkillExecutionContext,
    ) -> SkillResolverResult:
        """Resolve instructions and render templates.

        Args:
            package: A loaded skill package.
            context: Execution context with variables and metadata.

        Returns:
            Resolved instructions and rendered templates.
        """
        ...


@runtime_checkable
class TemplateRendererPort(Protocol):
    """Renders template strings by substituting variables."""

    def render(self, template: str, variables: dict[str, str]) -> str:
        """Render a template string.

        Args:
            template: Template text with ``{{variable}}`` placeholders.
            variables: Mapping of variable names to values.

        Returns:
            The rendered string.
        """
        ...


@runtime_checkable
class SkillRegistryPort(Protocol):
    """Stores and retrieves skill packages by name."""

    def register(self, package: SkillPackage) -> None:
        """Register a skill package."""
        ...

    def get(self, name: str) -> SkillPackage | None:
        """Retrieve a registered skill by name."""
        ...

    def list_names(self) -> list[str]:
        """Return names of all registered skills."""
        ...


@runtime_checkable
class SkillObserverPort(Protocol):
    """Receives notifications about skill lifecycle events."""

    def on_load(self, package: SkillPackage) -> None:
        """Called after a skill package is loaded."""
        ...

    def on_validate(
        self,
        manifest: SkillManifest,
        result: SkillValidationResult,
    ) -> None:
        """Called after a skill is validated."""
        ...

    def on_resolve(
        self,
        package: SkillPackage,
        result: SkillResolverResult,
    ) -> None:
        """Called after a skill is resolved."""
        ...
