"""Tests for Typer CLI interface."""

from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from typer.testing import CliRunner

from claude_wrapper.cli.main import app
from claude_wrapper.core.session import Session, Message
from datetime import datetime


def create_test_session(session_id: str = "test-session", **kwargs):
    """Create a test session with required timestamps."""
    now = datetime.now()
    
    # Convert dict messages to Message objects if provided
    if 'messages' in kwargs:
        messages = []
        for msg in kwargs['messages']:
            if isinstance(msg, dict):
                messages.append(Message(role=msg['role'], content=msg['content']))
            else:
                messages.append(msg)
        kwargs['messages'] = messages
    
    return Session(
        id=session_id,
        created_at=now,
        updated_at=now,
        **kwargs
    )


@pytest.fixture
def cli_runner():
    """Create a Typer test runner."""
    return CliRunner()


class TestCLI:
    """Test suite for CLI commands."""
    
    @pytest.mark.unit
    def test_cli_help(self, cli_runner):
        """Test CLI help command."""
        result = cli_runner.invoke(app, ["--help"])
        
        assert result.exit_code == 0
        assert "claude-wrapper" in result.output or "Claude" in result.output
        assert "chat" in result.output
        assert "complete" in result.output
        assert "sessions" in result.output
    
    @pytest.mark.unit
    def test_chat_command(self, cli_runner):
        """Test basic chat command."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.chat = AsyncMock()
            mock_get_client.return_value.chat.return_value = ("Test response", create_test_session("test-123"))
            
            result = cli_runner.invoke(app, ["chat", "Hello Claude"])
            
            assert result.exit_code == 0
            assert "Test response" in result.output
    
    @pytest.mark.unit
    def test_chat_with_session(self, cli_runner):
        """Test chat with session ID."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_session = create_test_session("existing-session")
            mock_get_client.return_value.session_manager.get_session = AsyncMock(return_value=mock_session)
            mock_get_client.return_value.chat = AsyncMock()
            mock_get_client.return_value.chat.return_value = ("Response", mock_session)
            
            result = cli_runner.invoke(app, [
                "chat", "Continue", "--session", "existing-session"
            ])
            
            assert result.exit_code == 0
            mock_get_client.return_value.session_manager.get_session.assert_called_once_with("existing-session")
    
    @pytest.mark.unit
    def test_chat_with_parameters(self, cli_runner):
        """Test chat with various parameters."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.chat = AsyncMock()
            mock_get_client.return_value.chat.return_value = ("Response", create_test_session())
            
            result = cli_runner.invoke(app, [
                "chat", "Test",
                "--max-tokens", "500",
                "--temperature", "0.5",
                "--system", "You are helpful"
            ])
            
            assert result.exit_code == 0
            
            # Verify parameters were passed
            mock_get_client.return_value.chat.assert_called_once()
            call_kwargs = mock_get_client.return_value.chat.call_args.kwargs
            assert call_kwargs["max_tokens"] == 500
            assert call_kwargs["temperature"] == 0.5
            assert call_kwargs["system_prompt"] == "You are helpful"
    
    @pytest.mark.unit
    def test_chat_streaming(self, cli_runner):
        """Test chat with streaming enabled."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            async def mock_stream(*args, **kwargs):
                for chunk in ["Hello ", "world!"]:
                    yield chunk
            
            mock_get_client.return_value.stream_chat = mock_stream
            
            result = cli_runner.invoke(app, ["chat", "Test", "--stream"])
            
            assert result.exit_code == 0
            # Output should contain streamed content
            assert "Hello world!" in result.output or "Hello" in result.output
    
    @pytest.mark.unit
    def test_complete_command(self, cli_runner):
        """Test complete command."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.complete = AsyncMock()
            mock_get_client.return_value.complete.return_value = "Completed text"
            
            result = cli_runner.invoke(app, ["complete", "Start of"])
            
            assert result.exit_code == 0
            assert "Completed text" in result.output
    
    @pytest.mark.unit
    def test_sessions_list(self, cli_runner):
        """Test listing sessions."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.session_manager.list_sessions = AsyncMock()
            mock_get_client.return_value.session_manager.list_sessions.return_value = [
                create_test_session("session-1", metadata={"user": "alice"}),
                create_test_session("session-2", metadata={"user": "bob"}),
            ]
            
            result = cli_runner.invoke(app, ["sessions"])
            
            assert result.exit_code == 0
            assert "session-..." in result.output  # Session IDs are truncated to 8 chars + "..."
    
    @pytest.mark.unit
    def test_session_detail(self, cli_runner):
        """Test viewing session details."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_session = create_test_session(
                "test-session",
                messages=[
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there!"},
                ],
                metadata={"test": "value"}
            )
            mock_get_client.return_value.session_manager.get_session = AsyncMock(return_value=mock_session)
            
            result = cli_runner.invoke(app, ["session", "test-session"])
            
            assert result.exit_code == 0
            assert "test-session" in result.output
            assert "Hello" in result.output
            assert "Hi there!" in result.output
    
    @pytest.mark.unit
    def test_session_not_found(self, cli_runner):
        """Test viewing non-existent session."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.session_manager.get_session = AsyncMock(return_value=None)
            
            result = cli_runner.invoke(app, ["session", "nonexistent"])
            
            assert result.exit_code != 0
            assert "not found" in result.output.lower()
    
    @pytest.mark.unit
    def test_cleanup_command(self, cli_runner):
        """Test cleanup old sessions."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.session_manager.cleanup_old_sessions = AsyncMock(return_value=5)
            
            result = cli_runner.invoke(app, ["cleanup", "--days", "7", "--force"])
            
            assert result.exit_code == 0
            assert "5" in result.output
            assert "deleted" in result.output.lower()
    
    @pytest.mark.unit
    def test_delete_session(self, cli_runner):
        """Test deleting specific session."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.session_manager.delete_session = AsyncMock(return_value=True)
            
            result = cli_runner.invoke(app, ["delete-session", "test-123", "--force"])
            
            assert result.exit_code == 0
            assert "deleted" in result.output.lower()
    
    @pytest.mark.unit
    def test_tokens_command(self, cli_runner):
        """Test token counting command."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.count_tokens = AsyncMock(return_value={"tokens": 5})
            
            result = cli_runner.invoke(app, ["tokens", "This is a test message"])
            
            assert result.exit_code == 0
            assert "tokens" in result.output.lower()
            # Should show a number
            assert any(char.isdigit() for char in result.output)
    
    @pytest.mark.unit
    def test_tokens_from_file(self, cli_runner, temp_dir):
        """Test counting tokens from file."""
        test_file = temp_dir / "test.txt"
        test_file.write_text("This is test content from a file.")
        
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.count_tokens = AsyncMock(return_value={"tokens": 8})
            
            result = cli_runner.invoke(app, ["tokens", "--file", str(test_file)])
            
            assert result.exit_code == 0
            assert "tokens" in result.output.lower()
    
    @pytest.mark.unit
    def test_config_show(self, cli_runner):
        """Test showing configuration."""
        with patch("claude_wrapper.cli.main.config") as mock_config:
            mock_config.to_yaml.return_value = "claude_path: /usr/bin/claude\ntimeout: 120\napi_key: null"
            
            result = cli_runner.invoke(app, ["configure", "--show"])
            
            assert result.exit_code == 0
            assert "claude_path" in result.output
            assert "/usr/bin/claude" in result.output
    
    @pytest.mark.unit
    def test_config_set(self, cli_runner):
        """Test setting configuration values."""
        with patch("claude_wrapper.cli.main.config") as mock_config:
            result = cli_runner.invoke(app, [
                "configure",
                "--claude-path", "/custom/claude",
                "--timeout", "60"
            ])
            
            assert result.exit_code == 0
            assert "updated" in result.output.lower()
    
    @pytest.mark.unit
    def test_server_command(self, cli_runner):
        """Test starting API server."""
        with patch("uvicorn.run") as mock_uvicorn:
            result = cli_runner.invoke(app, [
                "server",
                "--host", "0.0.0.0",
                "--port", "8080"
            ])
            
            # Server runs in background, so check it was called
            mock_uvicorn.assert_called_once()
            call_kwargs = mock_uvicorn.call_args.kwargs
            assert call_kwargs["host"] == "0.0.0.0"
            assert call_kwargs["port"] == 8080
    
    @pytest.mark.unit
    def test_version_command(self, cli_runner):
        """Test version command."""
        result = cli_runner.invoke(app, ["version"])
        
        assert result.exit_code == 0
        assert "0.1.0" in result.output  # Or whatever version is set
    
    @pytest.mark.unit
    def test_error_handling(self, cli_runner):
        """Test error handling in CLI."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            from claude_wrapper.core.exceptions import AuthenticationError
            mock_get_client.return_value.chat = AsyncMock()
            mock_get_client.return_value.chat.side_effect = AuthenticationError("Not authenticated")
            
            result = cli_runner.invoke(app, ["chat", "Test"])
            
            assert result.exit_code != 0
            assert "error" in result.output.lower() or "authenticated" in result.output.lower()
    
    @pytest.mark.unit
    def test_chat_output_formatting(self, cli_runner):
        """Test that chat output is properly formatted."""
        with patch("claude_wrapper.cli.main.get_client") as mock_get_client:
            mock_get_client.return_value.chat = AsyncMock()
            mock_get_client.return_value.chat.return_value = (
                "```python\nprint('Hello')\n```",
                create_test_session("test")
            )
            
            result = cli_runner.invoke(app, ["chat", "Show me Python"])
            
            assert result.exit_code == 0
            # Should preserve code formatting
            assert "print('Hello')" in result.output