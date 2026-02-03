"""Main CLI application using Typer."""

import sys
from importlib.metadata import version as get_version

import typer
from rich.console import Console
from rich.table import Table

from electripy.core.config import Config
from electripy.core.logging import setup_logging

app = typer.Typer(
    name="electripy",
    help="ElectriPy - Production-minded Python components and recipes",
    no_args_is_help=True,
)

console = Console()


@app.command()
def doctor() -> None:
    """Check ElectriPy installation and environment health.
    
    Verifies that ElectriPy is properly installed and all dependencies
    are available.
    """
    console.print("\n[bold cyan]ElectriPy Doctor[/bold cyan]")
    console.print("=" * 50)
    
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Check", style="cyan", width=30)
    table.add_column("Status", width=15)
    table.add_column("Details")
    
    # Check Python version
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    py_status = "✓ OK" if sys.version_info >= (3, 11) else "✗ FAIL"
    py_style = "green" if sys.version_info >= (3, 11) else "red"
    table.add_row(
        "Python Version",
        f"[{py_style}]{py_status}[/{py_style}]",
        py_version,
    )
    
    # Check ElectriPy version
    try:
        electripy_version = get_version("electripy")
        table.add_row(
            "ElectriPy Version",
            "[green]✓ OK[/green]",
            electripy_version,
        )
    except Exception as e:
        table.add_row(
            "ElectriPy Version",
            "[red]✗ FAIL[/red]",
            str(e),
        )
    
    # Check configuration
    try:
        config = Config.from_env()
        table.add_row(
            "Configuration",
            "[green]✓ OK[/green]",
            f"Log level: {config.log_level}",
        )
    except Exception as e:
        table.add_row(
            "Configuration",
            "[red]✗ FAIL[/red]",
            str(e),
        )
    
    # Check dependencies
    dependencies = ["typer", "rich"]
    for dep in dependencies:
        try:
            dep_version = get_version(dep)
            table.add_row(
                f"{dep} package",
                "[green]✓ OK[/green]",
                dep_version,
            )
        except Exception:
            table.add_row(
                f"{dep} package",
                "[red]✗ FAIL[/red]",
                "Not installed",
            )
    
    console.print(table)
    console.print("\n[bold green]✓[/bold green] ElectriPy is ready to use!\n")


@app.command()
def version() -> None:
    """Show ElectriPy version."""
    try:
        ver = get_version("electripy")
        console.print(f"ElectriPy version: [cyan]{ver}[/cyan]")
    except Exception:
        console.print("[red]Could not determine version[/red]")


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
) -> None:
    """ElectriPy CLI main entry point."""
    if verbose:
        setup_logging(level="DEBUG")
    else:
        setup_logging(level="INFO")


if __name__ == "__main__":
    app()
