"""MCP server exposing Claude Wrapper capabilities via the Model Context Protocol.

Supports both stdio and streamable HTTP transports.  The HTTP transport can
optionally be protected with an API key – pass it as
``Authorization: Bearer <key>`` or set ``CLAUDE_WRAPPER_API_KEY``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Annotated

from mcp.server import FastMCP
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.routing import Mount

from claude_wrapper.core.client import ClaudeClient
from claude_wrapper.core.exceptions import ClaudeWrapperError
from claude_wrapper.utils.config import get_config

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Each entry includes the model id, owner, and the context/output limits that
# Claude Code exposes.  Short aliases (opus/sonnet/haiku) are kept alongside
# the canonical versioned names so existing configurations keep working.
_AVAILABLE_MODELS: list[dict] = [
    # ── Short aliases (passed directly to `claude --model`) ───────────────
    {"id": "opus",   "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "sonnet", "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "haiku",  "owned_by": "anthropic", "context_window":   200_000, "max_output_tokens":  8_192},
    # ── Haiku ─────────────────────────────────────────────────────────────
    {"id": "claude-haiku-4-5",          "owned_by": "anthropic", "context_window": 200_000, "max_output_tokens":  8_192},
    {"id": "claude-haiku-4-5-20251001", "owned_by": "anthropic", "context_window": 200_000, "max_output_tokens":  8_192},
    # ── Sonnet ────────────────────────────────────────────────────────────
    {"id": "claude-sonnet-4-0",          "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "claude-sonnet-4-20250514",   "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "claude-sonnet-4-5",          "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "claude-sonnet-4-5-20250929", "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "claude-sonnet-4-6",          "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    # ── Opus ──────────────────────────────────────────────────────────────
    {"id": "claude-opus-4-0",          "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "claude-opus-4-20250514",   "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "claude-opus-4-1",          "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "claude-opus-4-1-20250805", "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "claude-opus-4-5",          "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "claude-opus-4-5-20251101", "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "claude-opus-4-6",          "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
    {"id": "claude-opus-4-7",          "owned_by": "anthropic", "context_window": 1_000_000, "max_output_tokens": 16_384},
]

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

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
# API-key middleware (HTTP transport only)
# ---------------------------------------------------------------------------


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Starlette middleware that enforces a Bearer API key on every request.

    Requests that carry a valid ``Authorization: Bearer <key>`` header are
    passed through unchanged.  All other requests receive a ``401`` response.
    """

    def __init__(self, app: Starlette, api_key: str) -> None:
        super().__init__(app)
        self._api_key = api_key

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        authorization = request.headers.get("Authorization", "")
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() != "bearer" or token != self._api_key:
            return JSONResponse(
                status_code=401,
                content={"error": "Unauthorized – valid Bearer token required"},
            )
        return await call_next(request)


def get_http_app(api_key: str | None = None) -> Starlette:
    """Return the Starlette ASGI app for the streamable-HTTP transport.

    When *api_key* is provided every incoming HTTP request must supply a
    matching ``Authorization: Bearer <api_key>`` header.  When omitted the
    endpoint is unauthenticated.

    Args:
        api_key: Secret token that clients must present as a Bearer token.
                 If *None* (or empty) no authentication is enforced.

    Returns:
        A :class:`starlette.applications.Starlette` ASGI app ready to be
        served by uvicorn or any other ASGI server.
    """
    inner: Starlette = mcp.streamable_http_app()

    if not api_key:
        return inner

    # Wrap the FastMCP Starlette app in a top-level app that enforces the key.
    return Starlette(
        routes=[Mount("/", app=inner)],
        middleware=[Middleware(APIKeyMiddleware, api_key=api_key)],
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
    return _AVAILABLE_MODELS


# ---------------------------------------------------------------------------
# Factory helper (used by CLI & tests)
# ---------------------------------------------------------------------------


def create_mcp_server() -> FastMCP:
    """Create and return a configured :class:`FastMCP` instance."""
    return mcp
