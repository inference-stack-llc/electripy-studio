"""Skills packaging system for AI applications.

Purpose:
  - Bundle AI instructions, metadata, assets, and execution hints into
    versioned, portable skill units.
  - Load, validate, resolve, and compose skills from disk.
  - Provide a structured way to package reusable AI behaviour.

Guarantees:
  - Skills are versioned using semantic versioning.
  - Manifests are validated before use.
  - Template rendering follows the ``{{variable}}`` convention.
  - All domain models are immutable frozen dataclasses.

Usage:
  Basic example::

    from electripy.ai.skills import load_skill, validate_skill

    result = validate_skill("./skills/code-review")
    if result.valid:
        pkg = load_skill("./skills/code-review")
        print(pkg.instructions.entry_instruction)
"""

from __future__ import annotations

from .adapters import (
    DefaultSkillResolver,
    DefaultSkillValidator,
    FileSystemAssetReader,
    FileSystemSkillLoader,
    InMemorySkillRegistry,
    MustacheStyleRenderer,
)
from .domain import (
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
from .errors import (
    AssetResolutionError,
    ManifestLoadError,
    SkillError,
    SkillNotFoundError,
    SkillValidationError,
    TemplateRenderError,
)
from .ports import (
    AssetReaderPort,
    SkillLoaderPort,
    SkillObserverPort,
    SkillRegistryPort,
    SkillResolverPort,
    SkillValidatorPort,
    TemplateRendererPort,
)
from .services import (
    SkillService,
    get_entry_instructions,
    list_skills,
    load_skill,
    read_skill_manifest,
    resolve_skill,
    validate_skill,
)

__all__ = [
    # Domain — enumerations
    "AssetKind",
    "SkillCapability",
    "ValidationSeverity",
    # Domain — value objects
    "SkillVersion",
    "SkillAsset",
    "SkillInstructionSet",
    "SkillDependency",
    "SkillMetadata",
    "SkillManifest",
    "ValidationDiagnostic",
    "SkillValidationResult",
    "SkillExecutionContext",
    "SkillPackage",
    "SkillResolverResult",
    # Errors
    "SkillError",
    "ManifestLoadError",
    "AssetResolutionError",
    "SkillValidationError",
    "SkillNotFoundError",
    "TemplateRenderError",
    # Ports
    "SkillLoaderPort",
    "AssetReaderPort",
    "SkillValidatorPort",
    "SkillResolverPort",
    "TemplateRendererPort",
    "SkillRegistryPort",
    "SkillObserverPort",
    # Adapters
    "FileSystemSkillLoader",
    "FileSystemAssetReader",
    "DefaultSkillValidator",
    "DefaultSkillResolver",
    "MustacheStyleRenderer",
    "InMemorySkillRegistry",
    # Services
    "SkillService",
    "load_skill",
    "validate_skill",
    "resolve_skill",
    "list_skills",
    "read_skill_manifest",
    "get_entry_instructions",
]
