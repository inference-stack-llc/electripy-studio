"""Exception hierarchy for the skills packaging system.

All exceptions extend :class:`SkillError` which derives from
:class:`ElectriPyError`, keeping the error hierarchy consistent
with the rest of the ElectriPy codebase.
"""

from __future__ import annotations

from electripy.core.errors import ElectriPyError

__all__ = [
    "AssetResolutionError",
    "ManifestLoadError",
    "SkillError",
    "SkillNotFoundError",
    "SkillValidationError",
    "TemplateRenderError",
]


class SkillError(ElectriPyError):
    """Base exception for all skill packaging failures."""


class ManifestLoadError(SkillError):
    """Raised when a skill manifest cannot be loaded or parsed."""


class AssetResolutionError(SkillError):
    """Raised when a skill asset file cannot be found or read."""


class SkillValidationError(SkillError):
    """Raised when a skill fails validation checks."""


class SkillNotFoundError(SkillError):
    """Raised when a requested skill cannot be located."""


class TemplateRenderError(SkillError):
    """Raised when template variable substitution fails."""
