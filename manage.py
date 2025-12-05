#!/usr/bin/env python
"""Management CLI for xai_chem_review project."""
import subprocess
import sys

import rich_click as click

# Known token for MCP server connection
JUPYTER_TOKEN = "xai-chem-review-mcp-token"
JUPYTER_PORT = 8888


@click.group()
@click.version_option(version="0.1.0")
def cli():
    """Management commands for the xai_chem_review project."""
    pass


@cli.command("jupyter-lab")
@click.option(
    "--port",
    "-p",
    default=JUPYTER_PORT,
    type=int,
    help=f"Port to run JupyterLab on (default: {JUPYTER_PORT})",
)
@click.option(
    "--token",
    "-t",
    default=JUPYTER_TOKEN,
    help="Authentication token for JupyterLab",
)
@click.option(
    "--no-browser",
    is_flag=True,
    default=False,
    help="Don't open browser automatically",
)
def jupyter_lab(port: int, token: str, no_browser: bool):
    """Start JupyterLab server with MCP-compatible configuration.

    Starts JupyterLab with a known token that can be used by the
    Jupyter MCP server for AI-assisted notebook editing.
    """
    click.echo(f"Starting JupyterLab on port {port}...")
    click.echo(f"Token: {token}")
    click.echo(f"URL: http://localhost:{port}/?token={token}")
    click.echo("-" * 50)

    cmd = [
        sys.executable, "-m", "jupyter", "lab",
        f"--port={port}",
        f"--IdentityProvider.token={token}",
        "--ip=0.0.0.0",
    ]

    if no_browser:
        cmd.append("--no-browser")

    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        click.echo("\nJupyterLab server stopped.")
    except subprocess.CalledProcessError as e:
        click.echo(f"Error starting JupyterLab: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    cli()
