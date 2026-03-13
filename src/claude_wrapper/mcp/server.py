"""MCP server exposing Claude Wrapper capabilities via the Model Context Protocol.

Supports both stdio and streamable HTTP transports.
"""

from __future__ import annotations

from typing import Annotated

from mcp.server import FastMCP

from claude_wrapper.core.client import ClaudeClient
from claude_wrapper.core.exceptions import ClaudeWrapperError
from claude_wrapper.utils.config import get_config

# ---------------------------------------------------------------------------
# Server instance (module-level so it can be imported and used directly)
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="claude-wrapper",
    instructions=(
        "A Model Context Protocol server that provides access to Claude via the "
        "Claude CLI.  Use the `chat` tool to have a conversation and the "
        "`complete` tool for raw text completion."
    ),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_client() -> ClaudeClient:
    """Return a configured :class:`ClaudeClient` instance."""
    config = get_config()
    return ClaudeClient(
        claude_path=config.claude_path,
        timeout=config.timeout,
        retry_attempts=config.retry_attempts,
    )


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def chat(
    message: Annotated[str, "The message to send to Claude"],
    system_prompt: Annotated[str | None, "Optional system prompt to guide the response"] = None,
) -> str:
    """Send a message to Claude and return the response.

    Uses the Claude CLI under the hood.  Supports an optional *system_prompt*
    that will be prepended to the conversation.
    """
    client = _get_client()
    try:
        if system_prompt:
            combined = f"{system_prompt}\n\nUser: {message}"
            return await client.chat(combined)
        return await client.chat(message)
    except ClaudeWrapperError as exc:
        raise RuntimeError(exc.message) from exc


@mcp.tool()
async def complete(
    prompt: Annotated[str, "The text prompt to complete"],
    max_tokens: Annotated[int | None, "Maximum number of tokens to generate"] = None,
    temperature: Annotated[float | None, "Sampling temperature (0–2)"] = None,
) -> str:
    """Complete a text prompt using Claude.

    Returns the generated completion text.
    """
    client = _get_client()
    try:
        return await client.complete(
            prompt=prompt,
            _max_tokens=max_tokens,
            _temperature=temperature,
        )
    except ClaudeWrapperError as exc:
        raise RuntimeError(exc.message) from exc


@mcp.tool()
async def list_models() -> list[dict[str, str]]:
    """Return the list of Claude models available through this server."""
    return [
        {"id": "opus", "owned_by": "anthropic"},
        {"id": "sonnet", "owned_by": "anthropic"},
        {"id": "haiku", "owned_by": "anthropic"},
    ]


# ---------------------------------------------------------------------------
# Factory helper (used by CLI & tests)
# ---------------------------------------------------------------------------


def create_mcp_server() -> FastMCP:
    """Create and return a configured :class:`FastMCP` instance."""
    return mcp
