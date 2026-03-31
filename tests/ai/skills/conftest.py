"""Shared fixtures for skills tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture()
def skill_dir(tmp_path: Path) -> Path:
    """Create a minimal valid skill directory on disk."""
    root = tmp_path / "code-review"
    root.mkdir()

    instructions = root / "instructions"
    instructions.mkdir()
    (instructions / "main.md").write_text(
        "You are a code reviewer. Review the following code:\n\n{{code}}\n",
        encoding="utf-8",
    )
    (instructions / "style.md").write_text(
        "Focus on naming conventions and readability.",
        encoding="utf-8",
    )

    templates = root / "templates"
    templates.mkdir()
    (templates / "report.md").write_text(
        "# Review Report\n\nReviewer: {{reviewer}}\n\n{{findings}}",
        encoding="utf-8",
    )

    assets = root / "assets"
    assets.mkdir()
    (assets / "config.json").write_text(
        json.dumps({"max_issues": 10}),
        encoding="utf-8",
    )

    manifest = {
        "name": "code-review",
        "version": "1.2.0",
        "description": "Automated code review skill",
        "entry_instruction": "instructions/main.md",
        "variables": ["code", "reviewer", "findings"],
        "assets": [
            {
                "name": "style-guide",
                "kind": "instruction",
                "path": "instructions/style.md",
                "description": "Style guidelines",
            },
            {
                "name": "report-template",
                "kind": "template",
                "path": "templates/report.md",
            },
            {
                "name": "config",
                "kind": "config",
                "path": "assets/config.json",
            },
        ],
        "dependencies": [
            {"name": "base-reviewer", "version": ">=1.0.0"},
        ],
        "metadata": {
            "author": "ElectriPy Team",
            "license": "MIT",
            "capabilities": ["code_generation"],
            "tags": ["review", "quality"],
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    return root


@pytest.fixture()
def minimal_skill_dir(tmp_path: Path) -> Path:
    """Create a bare-minimum valid skill directory."""
    root = tmp_path / "minimal"
    root.mkdir()
    (root / "main.md").write_text("Do something.", encoding="utf-8")
    manifest = {
        "name": "minimal",
        "version": "0.1.0",
        "entry_instruction": "main.md",
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    return root


@pytest.fixture()
def skills_parent(skill_dir: Path, minimal_skill_dir: Path) -> Path:
    """Return the parent directory containing multiple skill dirs."""
    return skill_dir.parent
