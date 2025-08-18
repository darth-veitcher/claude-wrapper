"""Simple CLI wrapper for Claude."""

import asyncio

import typer
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel

from claude_wrapper.core import ClaudeClient
from claude_wrapper.core.exceptions import ClaudeWrapperError
from claude_wrapper.utils.config import Config

app = typer.Typer(
    name="claude-wrapper",
    help="A simple CLI wrapper for Claude.",
    add_completion=True,
    rich_markup_mode="rich",
)

console = Console()
config = Config()


def get_client() -> ClaudeClient:
    """Get configured Claude client."""
    return ClaudeClient(
        claude_path=config.claude_path,
        timeout=config.timeout,
        retry_attempts=config.retry_attempts,
    )


@app.command()
def chat(
    message: str = typer.Argument(..., help="Message to send to Claude"),
    stream: bool = typer.Option(False, "--stream", help="Stream the response"),
) -> None:
    """Send a message to Claude and get a response."""

    async def _chat() -> None:
        try:
            client = get_client()

            if stream:
                # Streaming response
                full_response = []
                async for chunk in client.stream_chat(message):
                    console.print(chunk, end="")
                    full_response.append(chunk)
                console.print()  # New line after streaming
            else:
                # Regular response
                with console.status("[bold cyan]Thinking...", spinner="dots"):
                    response = await client.chat(message)

                # Display response in a nice panel
                console.print(Panel(Markdown(response), title="Claude's Response", expand=False))

        except ClaudeWrapperError as e:
            console.print(f"[red]Error: {e.message}[/red]")
            raise typer.Exit(1) from e
        except Exception as e:
            console.print(f"[red]Unexpected error: {str(e)}[/red]")
            raise typer.Exit(1) from e

    asyncio.run(_chat())


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", "--host", "-h", help="Host to bind to"),
    port: int = typer.Option(8000, "--port", "-p", help="Port to bind to"),
    reload: bool = typer.Option(False, "--reload", help="Enable auto-reload for development"),
) -> None:
    """Start the OpenAI-compatible API server."""
    console.print(f"[cyan]Starting API server on {host}:{port}...[/cyan]")

    import uvicorn

    uvicorn.run(
        "claude_wrapper.api.server:app",
        host=host,
        port=port,
        reload=reload,
    )


@app.command()
def version() -> None:
    """Show version information."""
    from claude_wrapper import __version__

    console.print(f"[cyan]Claude Wrapper[/cyan] v{__version__}")

    # Check Claude CLI version
    try:
        client = get_client()
        asyncio.run(client.check_auth())
        console.print("[green]✓[/green] Claude CLI is installed and authenticated")
    except Exception as e:
        console.print(f"[red]✗[/red] Claude CLI issue: {str(e)}")


if __name__ == "__main__":
    app()
