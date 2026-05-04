"""Tests for the Claude client."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest

from claude_wrapper.core.client import ClaudeClient
from claude_wrapper.core.exceptions import (
    ClaudeAuthError,
    ClaudeExecutionError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
)


class TestClaudeClient:
    """Test suite for ClaudeClient."""

    @pytest.mark.unit
    async def test_client_initialization(self):
        """Test ClaudeClient initialization with default and custom parameters."""
        with patch("shutil.which", return_value="/usr/bin/claude"):
            # Test default initialization
            client = ClaudeClient()
            assert client.claude_path == "claude"
            assert client.timeout == 120.0
            assert client.retry_attempts == 3

            # Test custom initialization
            client = ClaudeClient(
                claude_path="/custom/claude",
                timeout=60.0,
                retry_attempts=5,
            )
            assert client.claude_path == "/custom/claude"
            assert client.timeout == 60.0
            assert client.retry_attempts == 5

    @pytest.mark.unit
    async def test_check_auth_success(self):
        """Test successful authentication check."""
        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec") as mock_proc,
        ):
            # Mock successful subprocess
            process = AsyncMock()
            process.communicate = AsyncMock(return_value=(b"Claude CLI v1.0", b""))
            process.returncode = 0
            mock_proc.return_value = process

            client = ClaudeClient()
            result = await client.check_auth()
            assert result is True

    @pytest.mark.unit
    async def test_check_auth_not_found(self):
        """Test authentication check when Claude CLI is not found."""
        with patch("shutil.which", return_value=None):
            client = ClaudeClient()

            with pytest.raises(ClaudeNotFoundError):
                await client.check_auth()

    @pytest.mark.unit
    async def test_check_auth_not_authenticated(self):
        """Test authentication check when Claude CLI is not authenticated."""
        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec") as mock_proc,
        ):
            # Mock failed authentication
            process = AsyncMock()
            process.communicate = AsyncMock(return_value=(b"", b"Error: not authenticated"))
            process.returncode = 1
            mock_proc.return_value = process

            client = ClaudeClient()

            with pytest.raises(ClaudeAuthError):
                await client.check_auth()

    @pytest.mark.unit
    async def test_check_auth_timeout(self):
        """Test authentication check timeout."""
        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec") as mock_proc,
        ):
            # Mock timeout
            process = AsyncMock()
            process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_proc.return_value = process

            client = ClaudeClient()

            with pytest.raises(ClaudeTimeoutError):
                await client.check_auth()

    @pytest.mark.unit
    async def test_chat_success(self):
        """Test successful chat interaction."""
        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec") as mock_proc,
        ):
            # Mock successful chat
            process = AsyncMock()
            process.communicate = AsyncMock(return_value=(b"Hello! How can I help you?", b""))
            process.returncode = 0
            mock_proc.return_value = process

            client = ClaudeClient()
            # Skip auth check for this test
            client._claude_available = True

            result = await client.chat("Hello")
            assert result == "Hello! How can I help you?"

    @pytest.mark.unit
    async def test_chat_with_retry(self):
        """Test chat with retry on failure."""
        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec") as mock_proc,
        ):
            # Mock failure then success
            failed_process = AsyncMock()
            failed_process.communicate = AsyncMock(return_value=(b"", b"Error"))
            failed_process.returncode = 1

            success_process = AsyncMock()
            success_process.communicate = AsyncMock(return_value=(b"Success", b""))
            success_process.returncode = 0

            mock_proc.side_effect = [failed_process, success_process]

            client = ClaudeClient(retry_attempts=2)
            client._claude_available = True

            result = await client.chat("Hello")
            assert result == "Success"
            assert mock_proc.call_count == 2

    @pytest.mark.unit
    async def test_chat_timeout(self):
        """Test chat timeout."""
        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec") as mock_proc,
        ):
            # Mock timeout
            process = AsyncMock()
            process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
            mock_proc.return_value = process

            client = ClaudeClient(retry_attempts=1)
            client._claude_available = True

            with pytest.raises(ClaudeTimeoutError):
                await client.chat("Hello")

    @pytest.mark.unit
    async def test_chat_execution_error(self):
        """Test chat execution error after all retries."""
        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec") as mock_proc,
        ):
            # Mock consistent failure
            process = AsyncMock()
            process.communicate = AsyncMock(return_value=(b"", b"Command failed"))
            process.returncode = 1
            mock_proc.return_value = process

            client = ClaudeClient(retry_attempts=2)
            client._claude_available = True

            with pytest.raises(ClaudeExecutionError):
                await client.chat("Hello")

    @pytest.mark.unit
    async def test_stream_chat_real_streaming(self):
        """Test streaming chat via stream-json output format."""
        from unittest.mock import MagicMock

        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec") as mock_proc,
        ):
            # Build fake NDJSON lines the process would emit
            lines = [
                b'{"type":"system","subtype":"init","session_id":"s1"}\n',
                b'{"type":"assistant","message":{"content":[{"type":"text","text":"Hello"}]}}\n',
                b'{"type":"assistant","message":{"content":[{"type":"text","text":"Hello world"}]}}\n',
                b'{"type":"assistant","message":{"content":[{"type":"text","text":"Hello world test"}]}}\n',
                b'{"type":"result","subtype":"success","result":"Hello world test"}\n',
            ]

            # AsyncIterator over the fake lines
            async def _aiter_lines():
                for line in lines:
                    yield line

            stdout_mock = MagicMock()
            stdout_mock.__aiter__ = lambda self: _aiter_lines()

            process = AsyncMock()
            process.stdout = stdout_mock
            process.wait = AsyncMock(return_value=0)
            mock_proc.return_value = process

            client = ClaudeClient()
            client._claude_available = True

            chunks = []
            async for chunk in client.stream_chat("Test"):
                chunks.append(chunk)

            result = "".join(chunks)
            assert "Hello" in result
            assert "world" in result
            assert "test" in result

    @pytest.mark.unit
    async def test_stream_chat_fallback(self):
        """Test streaming chat falls back to word-by-word on subprocess error."""
        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeClient()
            client._claude_available = True

            # Make subprocess creation raise so the fallback path is exercised
            with (
                patch("asyncio.create_subprocess_exec", side_effect=OSError("no binary")),
                patch.object(client, "chat", return_value="Hello world test"),
            ):
                chunks = []
                async for chunk in client.stream_chat("Test"):
                    chunks.append(chunk)

                result = "".join(chunks)
                assert "Hello" in result
                assert "world" in result
                assert "test" in result

    @pytest.mark.unit
    async def test_complete(self):
        """Test completion method."""
        with patch("shutil.which", return_value="/usr/bin/claude"):
            client = ClaudeClient()

            # Mock the chat method since complete delegates to it
            with patch.object(client, "chat", return_value="Completed text"):
                result = await client.complete("Complete this:")
                assert result == "Completed text"

    @pytest.mark.unit
    async def test_count_tokens(self):
        """Test token counting (approximation)."""
        client = ClaudeClient()

        result = await client.count_tokens("Hello world test")
        assert "tokens" in result
        assert "characters" in result
        assert result["characters"] == 16  # "Hello world test"
        assert result["tokens"] == 4  # Approximation: 16 chars / 4
        assert "approximation" in result["note"]

    @pytest.mark.unit
    async def test_execute_claude_with_input(self):
        """Test _execute_claude with input text."""
        with (
            patch("shutil.which", return_value="/usr/bin/claude"),
            patch("asyncio.create_subprocess_exec") as mock_proc,
        ):
            # Mock successful execution
            process = AsyncMock()
            process.communicate = AsyncMock(return_value=(b"Response with input", b""))
            process.returncode = 0
            mock_proc.return_value = process

            client = ClaudeClient()
            client._claude_available = True

            result = await client._execute_claude(["-p"], input_text="Test input")
            assert result == "Response with input"

            # Verify communicate was called with input
            process.communicate.assert_called_once_with(b"Test input")
