"""Tests for CLI module."""

from typer.testing import CliRunner

from electripy.cli.app import app

runner = CliRunner()


def test_cli_doctor() -> None:
    """Test electripy doctor command."""
    result = runner.invoke(app, ["doctor"])
    assert result.exit_code == 0
    assert "ElectriPy Doctor" in result.stdout
    assert "Python Version" in result.stdout


def test_cli_version() -> None:
    """Test electripy version command."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "version" in result.stdout.lower()


def test_cli_help() -> None:
    """Test CLI help output."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "ElectriPy" in result.stdout
    assert "doctor" in result.stdout
