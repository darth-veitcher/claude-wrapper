"""Tests for the MCP server."""

from unittest.mock import AsyncMock, patch

import pytest

from claude_wrapper.mcp.server import create_mcp_server, mcp


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
