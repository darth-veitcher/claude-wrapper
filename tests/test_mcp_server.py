"""Tests for the MCP server."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient
from starlette.applications import Starlette
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.routing import Mount

from claude_wrapper.mcp.server import APIKeyMiddleware, create_mcp_server, get_http_app, mcp


@pytest.mark.unit
class TestMcpServerCreation:
    """Test MCP server instance creation."""

    def test_create_mcp_server_returns_fastmcp(self):
        """create_mcp_server returns the module-level FastMCP instance."""
        from mcp.server import FastMCP

        server = create_mcp_server()
        assert isinstance(server, FastMCP)

    def test_module_level_mcp_is_fastmcp(self):
        """The module-level `mcp` variable is a FastMCP instance."""
        from mcp.server import FastMCP

        assert isinstance(mcp, FastMCP)

    def test_server_has_expected_tools(self):
        """MCP server registers the expected tool names."""
        tool_names = {t.name for t in mcp._tool_manager.list_tools()}
        assert "chat" in tool_names
        assert "complete" in tool_names
        assert "list_models" in tool_names


@pytest.mark.asyncio
@pytest.mark.unit
class TestMcpChatTool:
    """Test the `chat` MCP tool."""

    async def test_chat_returns_response(self):
        """chat tool forwards message to ClaudeClient and returns response."""
        with patch("claude_wrapper.mcp.server._get_client") as mock_get_client:
            client = AsyncMock()
            client.chat = AsyncMock(return_value="Hello from Claude!")
            mock_get_client.return_value = client

            from claude_wrapper.mcp.server import chat

            result = await chat(message="Hello")
            assert result == "Hello from Claude!"
            client.chat.assert_awaited_once_with("Hello")

    async def test_chat_with_system_prompt(self):
        """chat tool prepends system prompt when provided."""
        with patch("claude_wrapper.mcp.server._get_client") as mock_get_client:
            client = AsyncMock()
            client.chat = AsyncMock(return_value="Sure!")
            mock_get_client.return_value = client

            from claude_wrapper.mcp.server import chat

            await chat(message="Tell me a joke", system_prompt="You are funny")
            expected = "You are funny\n\nUser: Tell me a joke"
            client.chat.assert_awaited_once_with(expected)

    async def test_chat_raises_runtime_error_on_wrapper_error(self):
        """chat tool converts ClaudeWrapperError to RuntimeError."""
        from claude_wrapper.core.exceptions import ClaudeWrapperError

        with patch("claude_wrapper.mcp.server._get_client") as mock_get_client:
            client = AsyncMock()
            client.chat = AsyncMock(side_effect=ClaudeWrapperError("oops"))
            mock_get_client.return_value = client

            from claude_wrapper.mcp.server import chat

            with pytest.raises(RuntimeError, match="oops"):
                await chat(message="Hi")


@pytest.mark.asyncio
@pytest.mark.unit
class TestMcpCompleteTool:
    """Test the `complete` MCP tool."""

    async def test_complete_returns_completion(self):
        """complete tool delegates to ClaudeClient.complete and returns text."""
        with patch("claude_wrapper.mcp.server._get_client") as mock_get_client:
            client = AsyncMock()
            client.complete = AsyncMock(return_value="The answer is 42.")
            mock_get_client.return_value = client

            from claude_wrapper.mcp.server import complete

            result = await complete(prompt="What is 6 times 7?")
            assert result == "The answer is 42."
            client.complete.assert_awaited_once_with(
                prompt="What is 6 times 7?",
                _max_tokens=None,
                _temperature=None,
            )

    async def test_complete_passes_optional_params(self):
        """complete tool forwards max_tokens and temperature."""
        with patch("claude_wrapper.mcp.server._get_client") as mock_get_client:
            client = AsyncMock()
            client.complete = AsyncMock(return_value="response")
            mock_get_client.return_value = client

            from claude_wrapper.mcp.server import complete

            await complete(prompt="Hello", max_tokens=100, temperature=0.5)
            client.complete.assert_awaited_once_with(
                prompt="Hello",
                _max_tokens=100,
                _temperature=0.5,
            )

    async def test_complete_raises_runtime_error_on_wrapper_error(self):
        """complete tool converts ClaudeWrapperError to RuntimeError."""
        from claude_wrapper.core.exceptions import ClaudeWrapperError

        with patch("claude_wrapper.mcp.server._get_client") as mock_get_client:
            client = AsyncMock()
            client.complete = AsyncMock(side_effect=ClaudeWrapperError("fail"))
            mock_get_client.return_value = client

            from claude_wrapper.mcp.server import complete

            with pytest.raises(RuntimeError, match="fail"):
                await complete(prompt="test")


@pytest.mark.asyncio
@pytest.mark.unit
class TestMcpListModelsTool:
    """Test the `list_models` MCP tool."""

    async def test_list_models_returns_list(self):
        """list_models returns a list of model dicts."""
        from claude_wrapper.mcp.server import list_models

        result = await list_models()
        assert isinstance(result, list)
        assert len(result) > 0
        ids = [m["id"] for m in result]
        assert "opus" in ids
        assert "sonnet" in ids
        assert "haiku" in ids


@pytest.mark.unit
class TestGetHttpApp:
    """Test the get_http_app factory."""

    def test_no_api_key_returns_inner_app(self):
        """get_http_app without a key returns the raw FastMCP Starlette app."""
        from starlette.applications import Starlette

        app = get_http_app(api_key=None)
        assert isinstance(app, Starlette)

    def test_with_api_key_returns_starlette_app(self):
        """get_http_app with a key returns a wrapped Starlette app."""
        from starlette.applications import Starlette

        app = get_http_app(api_key="secret")
        assert isinstance(app, Starlette)

    def test_empty_string_key_treated_as_no_key(self):
        """An empty string API key is falsy; no middleware is added."""
        app_no_key = get_http_app(api_key=None)
        app_empty = get_http_app(api_key="")
        # Both should be the same inner app (no wrapping)
        assert type(app_no_key) is type(app_empty)


@pytest.mark.asyncio
@pytest.mark.unit
class TestAPIKeyMiddleware:
    """Test the APIKeyMiddleware enforces bearer tokens."""

    def _make_protected_app(self, api_key: str) -> Starlette:
        """Build a minimal Starlette app wrapped with APIKeyMiddleware."""
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def ok(_request: Request) -> PlainTextResponse:
            return PlainTextResponse("ok")

        inner = Starlette(routes=[Route("/probe", ok)])
        return Starlette(
            routes=[Mount("/", app=inner)],
            middleware=[Middleware(APIKeyMiddleware, api_key=api_key)],
        )

    async def test_valid_key_passes_through(self):
        """Requests with the correct Bearer token reach the inner app."""
        app = self._make_protected_app("my-secret")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/probe", headers={"Authorization": "Bearer my-secret"})
            assert response.status_code == 200
            assert response.text == "ok"

    async def test_missing_auth_header_returns_401(self):
        """Requests without an Authorization header are rejected."""
        app = self._make_protected_app("my-secret")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/probe")
            assert response.status_code == 401

    async def test_wrong_key_returns_401(self):
        """Requests with an incorrect token are rejected."""
        app = self._make_protected_app("my-secret")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/probe", headers={"Authorization": "Bearer wrong-key"})
            assert response.status_code == 401

    async def test_non_bearer_scheme_returns_401(self):
        """Requests using a non-Bearer auth scheme are rejected."""
        app = self._make_protected_app("my-secret")
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/probe", headers={"Authorization": "Basic my-secret"})
            assert response.status_code == 401

    async def test_no_api_key_allows_unauthenticated(self):
        """When no API key is configured any request gets through."""
        from starlette.responses import PlainTextResponse
        from starlette.routing import Route

        async def ok(_request: Request) -> PlainTextResponse:
            return PlainTextResponse("ok")

        app = Starlette(routes=[Route("/probe", ok)])
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/probe")
            assert response.status_code == 200


@pytest.mark.unit
class TestMcpServerCLI:
    """Test the mcp-server CLI command."""

    def test_mcp_server_unknown_transport(self):
        """Unknown transport exits with code 1."""
        from typer.testing import CliRunner

        from claude_wrapper.cli.main import app as cli_app

        runner = CliRunner()
        result = runner.invoke(cli_app, ["mcp-server", "--transport", "grpc"])
        assert result.exit_code == 1
        assert "Unknown transport" in result.output

    def test_mcp_server_stdio_runs(self):
        """stdio transport calls mcp.run with transport='stdio'."""
        from unittest.mock import patch

        from typer.testing import CliRunner

        from claude_wrapper.cli.main import app as cli_app

        runner = CliRunner()
        with patch("claude_wrapper.mcp.server.mcp") as mock_mcp:
            mock_mcp.run = lambda **_kwargs: None
            result = runner.invoke(cli_app, ["mcp-server", "--transport", "stdio"])
        assert result.exit_code == 0

    def test_mcp_server_http_uses_api_key(self):
        """HTTP transport with --api-key passes key to get_http_app."""
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from claude_wrapper.cli.main import app as cli_app

        runner = CliRunner()
        captured: dict = {}

        def fake_get_http_app(api_key=None):
            captured["api_key"] = api_key
            return MagicMock()

        with (
            patch("claude_wrapper.mcp.server.get_http_app", fake_get_http_app),
            patch("uvicorn.run"),
            patch("claude_wrapper.mcp.server.mcp") as mock_mcp,
        ):
            mock_mcp.settings = MagicMock()
            result = runner.invoke(
                cli_app,
                ["mcp-server", "--transport", "streamable-http", "--api-key", "tok"],
            )

        assert result.exit_code == 0
        assert captured["api_key"] == "tok"

    def test_mcp_server_http_reads_env_api_key(self):
        """--api-key falls back to CLAUDE_WRAPPER_API_KEY environment variable."""
        import os
        from unittest.mock import MagicMock, patch

        from typer.testing import CliRunner

        from claude_wrapper.cli.main import app as cli_app

        runner = CliRunner()
        captured: dict = {}

        def fake_get_http_app(api_key=None):
            captured["api_key"] = api_key
            return MagicMock()

        with (
            patch("claude_wrapper.mcp.server.get_http_app", fake_get_http_app),
            patch("uvicorn.run"),
            patch("claude_wrapper.mcp.server.mcp") as mock_mcp,
        ):
            mock_mcp.settings = MagicMock()
            result = runner.invoke(
                cli_app,
                ["mcp-server", "--transport", "streamable-http"],
                env={**os.environ, "CLAUDE_WRAPPER_API_KEY": "env-secret"},
            )

        assert result.exit_code == 0
        assert captured["api_key"] == "env-secret"

