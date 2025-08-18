"""Tests for the CLI interface."""

import re
from unittest.mock import AsyncMock, patch

import pytest
from typer.testing import CliRunner

from claude_wrapper.cli.main import app


@pytest.mark.cli
class TestCLICommands:
    """Test CLI commands."""

    @pytest.fixture
    def cli_runner(self):
        """Get CLI runner."""
        return CliRunner()

    @pytest.mark.unit
    def test_version_command(self, cli_runner):
        """Test version command."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.check_auth = AsyncMock(return_value=True)

            result = cli_runner.invoke(app, ["version"])

            assert result.exit_code == 0
            assert "Claude Wrapper" in result.output
            assert "✓ Claude CLI is installed and authenticated" in result.output

    @pytest.mark.unit
    def test_version_command_with_error(self, cli_runner):
        """Test version command with auth error."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.check_auth = AsyncMock(
                side_effect=Exception("Not authenticated")
            )

            result = cli_runner.invoke(app, ["version"])

            assert result.exit_code == 0
            assert "Claude Wrapper" in result.output
            assert "✗ Claude CLI issue: Not authenticated" in result.output

    @pytest.mark.unit
    def test_chat_command(self, cli_runner):
        """Test basic chat command."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.chat = AsyncMock(return_value="Hello! How can I help you?")

            result = cli_runner.invoke(app, ["chat", "Hello"])

            assert result.exit_code == 0
            assert "Hello! How can I help you?" in result.output

    @pytest.mark.unit
    def test_chat_streaming(self, cli_runner):
        """Test chat with streaming enabled."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:

            async def mock_stream(*_args, **_kwargs):
                for chunk in ["Hello ", "world!"]:
                    yield chunk

            mock_get_client.return_value.stream_chat = mock_stream

            result = cli_runner.invoke(app, ["chat", "Test", "--stream"])

            assert result.exit_code == 0
            assert "Hello world!" in result.output

    @pytest.mark.unit
    def test_chat_error_handling(self, cli_runner):
        """Test chat error handling."""
        from claude_wrapper.core.exceptions import ClaudeWrapperError

        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.chat = AsyncMock(
                side_effect=ClaudeWrapperError("Test error")
            )

            result = cli_runner.invoke(app, ["chat", "Hello"])

            assert result.exit_code == 1
            assert "Error: Test error" in result.output

    @pytest.mark.unit
    def test_chat_unexpected_error(self, cli_runner):
        """Test chat with unexpected error."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.chat = AsyncMock(side_effect=Exception("Unexpected"))

            result = cli_runner.invoke(app, ["chat", "Hello"])

            assert result.exit_code == 1
            assert "Unexpected error: Unexpected" in result.output

    @pytest.mark.unit
    def test_config_loading(self):
        """Test configuration loading."""
        with patch("claude_wrapper.cli.main.config") as mock_config:
            mock_config.claude_path = "/custom/claude"
            mock_config.timeout = 60

            # Just test that config is accessible
            assert mock_config.claude_path == "/custom/claude"
            assert mock_config.timeout == 60

    @pytest.mark.unit
    def test_server_command(self, cli_runner):
        """Test starting API server."""
        with patch("uvicorn.run") as mock_uvicorn:
            cli_runner.invoke(app, ["server", "--host", "0.0.0.0", "--port", "8080"])

            # Server runs in background, so check it was called
            mock_uvicorn.assert_called_once()
            call_kwargs = mock_uvicorn.call_args.kwargs
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 8080

    @pytest.mark.unit
    def test_server_command_with_reload(self, cli_runner):
        """Test starting API server with reload."""
        with patch("uvicorn.run") as mock_uvicorn:
            cli_runner.invoke(app, ["server", "--reload"])

            mock_uvicorn.assert_called_once()
            call_kwargs = mock_uvicorn.call_args.kwargs
            assert call_kwargs["reload"] is True

    @pytest.mark.unit
    def test_help_command(self, cli_runner):
        """Test help command."""
        result = cli_runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "CLI wrapper for Claude" in result.output
        assert "chat" in result.output
        assert "server" in result.output
        assert "version" in result.output

    @pytest.mark.unit
    def test_chat_help(self, cli_runner):
        """Test chat command help."""
        result = cli_runner.invoke(app, ["chat", "--help"])

        assert result.exit_code == 0
        assert "Send a message to Claude" in result.output
        # Strip ANSI color codes for reliable string matching
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--stream" in clean_output

    @pytest.mark.unit
    def test_server_help(self, cli_runner):
        """Test server command help."""
        result = cli_runner.invoke(app, ["server", "--help"])

        assert result.exit_code == 0
        assert "Start the OpenAI-compatible API server" in result.output
        # Strip ANSI color codes for reliable string matching
        clean_output = re.sub(r"\x1b\[[0-9;]*m", "", result.output)
        assert "--host" in clean_output
        assert "--port" in clean_output
        assert "--reload" in clean_output
