"""Tests for the ``electripy demo`` CLI subcommands."""

from typer.testing import CliRunner

from electripy.cli.app import app

runner = CliRunner()


def test_demo_help_lists_policy_collab() -> None:
    result = runner.invoke(app, ["demo", "--help"])
    assert result.exit_code == 0
    assert "policy-collab" in result.stdout


def test_demo_policy_collab_default() -> None:
    result = runner.invoke(app, ["demo", "policy-collab"])
    assert result.exit_code == 0
    assert "Pipeline Summary" in result.stdout
    assert "LLM Response" in result.stdout
    assert "Agent Transcript" in result.stdout
    assert "verified" in result.stdout.lower() or "completed" in result.stdout.lower()


def test_demo_policy_collab_custom_prompt() -> None:
    result = runner.invoke(
        app,
        ["demo", "policy-collab", "--prompt", "Check user@corp.io secrets"],
    )
    assert result.exit_code == 0
    assert "Pipeline Summary" in result.stdout


def test_demo_policy_collab_max_hops() -> None:
    result = runner.invoke(
        app,
        ["demo", "policy-collab", "--max-hops", "4"],
    )
    assert result.exit_code == 0
    assert "Hops" in result.stdout
