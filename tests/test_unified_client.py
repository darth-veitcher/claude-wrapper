"""Tests for unified client."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from claude_wrapper.core.exceptions import (
    ClaudeAuthError,
    ClaudeExecutionError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
    ClaudeWrapperError,
)
from claude_wrapper.core.unified_client import (
    ClientMode,
    RecursionError,
    UnifiedClaudeClient,
    _call_stack,
    _recursion_depth,
    _thread_local,
    create_api_client,
    create_claude_client,
    create_claude_wrapper_client,
)


class TestUnifiedClaudeClient:
    """Test UnifiedClaudeClient functionality."""

    def test_init_defaults(self):
        """Test initialization with defaults."""
        client = UnifiedClaudeClient()
        assert client.mode == ClientMode.CLAUDE_CLI
        assert client.claude_path == "claude"
        assert client.claude_wrapper_path == "claude-wrapper"
        assert client.timeout == 120.0
        assert client.retry_attempts == 3
        assert client.max_recursion_depth == 10
        assert client.enable_recursion_tracking is True
        assert client._claude_available is None
        assert client._http_client is None

    def test_init_api_mode(self):
        """Test initialization in API mode."""
        client = UnifiedClaudeClient(
            mode=ClientMode.API,
            api_base_url="https://api.example.com",
            api_key="test-key",
        )
        assert client.mode == ClientMode.API
        assert client._http_client is not None

    def test_init_custom_values(self):
        """Test initialization with custom values."""
        client = UnifiedClaudeClient(
            mode=ClientMode.CLAUDE_WRAPPER_CLI,
            claude_path="/custom/claude",
            claude_wrapper_path="/custom/wrapper",
            timeout=30.0,
            retry_attempts=5,
            max_recursion_depth=20,
            enable_recursion_tracking=False,
        )
        assert client.mode == ClientMode.CLAUDE_WRAPPER_CLI
        assert client.claude_path == "/custom/claude"
        assert client.claude_wrapper_path == "/custom/wrapper"
        assert client.timeout == 30.0
        assert client.retry_attempts == 5
        assert client.max_recursion_depth == 20
        assert client.enable_recursion_tracking is False

    async def test_context_manager_without_http_client(self):
        """Test context manager when not in API mode."""
        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI)
        async with client as c:
            assert c is client
            assert c._http_client is None

    @patch("claude_wrapper.core.unified_client.ClaudeHTTPClient")
    async def test_context_manager_with_http_client(self, mock_http_client_class):
        """Test context manager in API mode."""
        mock_http_client = AsyncMock()
        mock_http_client_class.return_value = mock_http_client

        async with UnifiedClaudeClient(mode=ClientMode.API) as client:
            assert client._http_client is mock_http_client
            mock_http_client.__aenter__.assert_called_once()

        mock_http_client.__aexit__.assert_called_once()

    def test_check_recursion_depth_disabled(self):
        """Test recursion checking when disabled."""
        client = UnifiedClaudeClient(enable_recursion_tracking=False)
        # Should not raise even with high depth
        for _ in range(20):
            client._check_recursion_depth("test")

    def test_check_recursion_depth_basic(self):
        """Test basic recursion depth checking."""
        client = UnifiedClaudeClient(max_recursion_depth=3)

        # Clear any existing context
        _recursion_depth.set(0)
        _call_stack.set([])

        # First few calls should work
        client._check_recursion_depth("op1")
        assert _recursion_depth.get() == 1

        client._check_recursion_depth("op2")
        assert _recursion_depth.get() == 2

        client._check_recursion_depth("op3")
        assert _recursion_depth.get() == 3

        # Should raise at max depth (which is 3, so next call will exceed)
        with pytest.raises(RecursionError, match="Maximum recursion depth"):
            client._check_recursion_depth("op4")

    def test_check_recursion_depth_loop_detection(self):
        """Test detection of immediate recursion loops."""
        client = UnifiedClaudeClient()

        # Set up a potential loop
        _recursion_depth.set(2)
        _call_stack.set(["claude-cli:op1", "claude-cli:op1"])

        with pytest.raises(RecursionError, match="infinite recursion loop"):
            client._check_recursion_depth("op1")

    def test_check_recursion_depth_no_context(self):
        """Test recursion checking with no existing context."""
        client = UnifiedClaudeClient()

        # Clear context to ensure we test the LookupError path
        # Context variables will handle LookupError internally and use defaults
        # We need to ensure clean state
        try:
            _recursion_depth.set(0)
            _call_stack.set([])
        except LookupError:
            pass

        # Should handle gracefully and initialize
        client._check_recursion_depth("test")

        # Verify it worked and set the context
        assert _recursion_depth.get() == 1
        assert len(_call_stack.get()) == 1

    def test_check_subprocess_recursion_disabled(self):
        """Test subprocess recursion checking when disabled."""
        client = UnifiedClaudeClient(enable_recursion_tracking=False)
        for _ in range(20):
            client._check_subprocess_recursion("test")

    def test_check_subprocess_recursion_basic(self):
        """Test basic subprocess recursion checking."""
        client = UnifiedClaudeClient(max_recursion_depth=3)

        # Reset thread local
        if hasattr(_thread_local, "depth"):
            _thread_local.depth = 0
            _thread_local.stack = []

        client._check_subprocess_recursion("op1")
        assert _thread_local.depth == 1

        client._check_subprocess_recursion("op2")
        assert _thread_local.depth == 2

        client._check_subprocess_recursion("op3")
        assert _thread_local.depth == 3

        with pytest.raises(RecursionError, match="Maximum recursion depth"):
            client._check_subprocess_recursion("op4")

    def test_cleanup_subprocess_recursion(self):
        """Test cleanup of subprocess recursion tracking."""
        client = UnifiedClaudeClient()

        # Set up some depth
        _thread_local.depth = 3
        _thread_local.stack = ["op1", "op2", "op3"]

        client._cleanup_subprocess_recursion()
        assert _thread_local.depth == 2
        assert len(_thread_local.stack) == 2

        client._cleanup_subprocess_recursion()
        client._cleanup_subprocess_recursion()
        assert _thread_local.depth == 0
        assert len(_thread_local.stack) == 0

    def test_cleanup_subprocess_recursion_uninitialized(self):
        """Test cleanup when thread local is not initialized."""
        client = UnifiedClaudeClient()

        # Clear thread local
        if hasattr(_thread_local, "depth"):
            delattr(_thread_local, "depth")
        if hasattr(_thread_local, "stack"):
            delattr(_thread_local, "stack")

        # Should handle gracefully
        client._cleanup_subprocess_recursion()

    @patch("shutil.which")
    @patch("asyncio.create_subprocess_exec")
    async def test_execute_claude_cli_success(self, mock_subprocess, mock_which):
        """Test successful Claude CLI execution."""
        mock_which.return_value = "/usr/bin/claude"

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Success\n", b""))
        mock_subprocess.return_value = mock_process

        client = UnifiedClaudeClient()
        client._claude_available = True  # Skip auth check

        result = await client._execute_claude_cli(["-p", "test"])
        assert result == "Success"

        mock_subprocess.assert_called_once()
        call_args = mock_subprocess.call_args[0]
        assert call_args == ("claude", "-p", "test")

    @patch("shutil.which")
    @patch("asyncio.create_subprocess_exec")
    async def test_execute_claude_cli_with_input(self, mock_subprocess, mock_which):
        """Test Claude CLI execution with input text."""
        mock_which.return_value = "/usr/bin/claude"

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Response", b""))
        mock_subprocess.return_value = mock_process

        client = UnifiedClaudeClient()
        client._claude_available = True

        result = await client._execute_claude_cli(["chat"], "Hello")
        assert result == "Response"

        mock_process.communicate.assert_called_once_with(b"Hello")

    @patch("shutil.which")
    @patch("asyncio.create_subprocess_exec")
    async def test_execute_claude_cli_retry_on_failure(self, mock_subprocess, mock_which):
        """Test Claude CLI retry on failure."""
        mock_which.return_value = "/usr/bin/claude"

        # First attempt fails, second succeeds
        mock_process_fail = AsyncMock()
        mock_process_fail.returncode = 1
        mock_process_fail.communicate = AsyncMock(return_value=(b"", b"Error"))

        mock_process_success = AsyncMock()
        mock_process_success.returncode = 0
        mock_process_success.communicate = AsyncMock(return_value=(b"Success", b""))

        mock_subprocess.side_effect = [mock_process_fail, mock_process_success]

        client = UnifiedClaudeClient(retry_attempts=2)
        client._claude_available = True

        result = await client._execute_claude_cli(["-p", "test"])
        assert result == "Success"
        assert mock_subprocess.call_count == 2

    @patch("shutil.which")
    @patch("asyncio.create_subprocess_exec")
    async def test_execute_claude_cli_timeout(self, mock_subprocess, mock_which):
        """Test Claude CLI timeout handling."""
        mock_which.return_value = "/usr/bin/claude"

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_subprocess.return_value = mock_process

        client = UnifiedClaudeClient(retry_attempts=1)
        client._claude_available = True

        with pytest.raises(ClaudeTimeoutError, match="timed out"):
            await client._execute_claude_cli(["-p", "test"])

    @patch("shutil.which")
    @patch("asyncio.create_subprocess_exec")
    async def test_execute_claude_cli_all_retries_fail(self, mock_subprocess, mock_which):
        """Test Claude CLI when all retries fail."""
        mock_which.return_value = "/usr/bin/claude"

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Persistent error"))
        mock_subprocess.return_value = mock_process

        client = UnifiedClaudeClient(retry_attempts=2)
        client._claude_available = True

        with pytest.raises(ClaudeExecutionError, match="Persistent error"):
            await client._execute_claude_cli(["-p", "test"])

        assert mock_subprocess.call_count == 2

    @patch("shutil.which")
    @patch("asyncio.create_subprocess_exec")
    async def test_execute_claude_wrapper_cli_success(self, mock_subprocess, mock_which):
        """Test successful claude-wrapper CLI execution."""
        mock_which.return_value = "/usr/bin/claude-wrapper"

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Response\n", b""))
        mock_subprocess.return_value = mock_process

        client = UnifiedClaudeClient()
        result = await client._execute_claude_wrapper_cli("chat", "Hello")
        assert result == "Response"

    @patch("shutil.which")
    async def test_execute_claude_wrapper_cli_not_found(self, mock_which):
        """Test claude-wrapper CLI not found."""
        mock_which.return_value = None

        client = UnifiedClaudeClient()
        with pytest.raises(ClaudeNotFoundError, match="not found"):
            await client._execute_claude_wrapper_cli("chat", "Hello")

    @patch("shutil.which")
    @patch("asyncio.create_subprocess_exec")
    async def test_execute_claude_wrapper_cli_with_stream(self, mock_subprocess, mock_which):
        """Test claude-wrapper CLI with streaming."""
        mock_which.return_value = "/usr/bin/claude-wrapper"

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Stream response", b""))
        mock_subprocess.return_value = mock_process

        client = UnifiedClaudeClient()
        result = await client._execute_claude_wrapper_cli("chat", "Hello", stream=True)
        assert result == "Stream response"

        call_args = mock_subprocess.call_args[0]
        assert "--stream" in call_args

    @patch("shutil.which")
    @patch("asyncio.create_subprocess_exec")
    async def test_check_claude_auth_success(self, mock_subprocess, mock_which):
        """Test successful Claude auth check."""
        mock_which.return_value = "/usr/bin/claude"

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"Claude v1.0", b""))
        mock_subprocess.return_value = mock_process

        client = UnifiedClaudeClient()
        result = await client._check_claude_auth()
        assert result is True
        assert client._claude_available is True

    @patch("shutil.which")
    async def test_check_claude_auth_not_found(self, mock_which):
        """Test Claude auth check when CLI not found."""
        mock_which.return_value = None

        client = UnifiedClaudeClient()
        with pytest.raises(ClaudeNotFoundError, match="not found"):
            await client._check_claude_auth()

    @patch("shutil.which")
    @patch("asyncio.create_subprocess_exec")
    async def test_check_claude_auth_not_authenticated(self, mock_subprocess, mock_which):
        """Test Claude auth check when not authenticated."""
        mock_which.return_value = "/usr/bin/claude"

        mock_process = AsyncMock()
        mock_process.returncode = 1
        mock_process.communicate = AsyncMock(return_value=(b"", b"Error: not authenticated"))
        mock_subprocess.return_value = mock_process

        client = UnifiedClaudeClient()
        with pytest.raises(ClaudeAuthError, match="not authenticated"):
            await client._check_claude_auth()

    @patch("shutil.which")
    @patch("asyncio.create_subprocess_exec")
    async def test_check_claude_auth_timeout(self, mock_subprocess, mock_which):
        """Test Claude auth check timeout."""
        mock_which.return_value = "/usr/bin/claude"

        mock_process = AsyncMock()
        mock_process.communicate = AsyncMock(side_effect=asyncio.TimeoutError())
        mock_subprocess.return_value = mock_process

        client = UnifiedClaudeClient()
        with pytest.raises(ClaudeTimeoutError, match="auth check timed out"):
            await client._check_claude_auth()

    @patch.object(UnifiedClaudeClient, "_execute_claude_cli")
    async def test_chat_claude_cli_mode(self, mock_execute):
        """Test chat in Claude CLI mode."""
        mock_execute.return_value = "CLI response"

        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI)

        # Reset context
        _recursion_depth.set(0)
        _call_stack.set([])

        result = await client.chat("Hello")
        assert result == "CLI response"
        mock_execute.assert_called_once_with(["-p", "Hello"])

    @patch.object(UnifiedClaudeClient, "_execute_claude_wrapper_cli")
    async def test_chat_wrapper_cli_mode(self, mock_execute):
        """Test chat in claude-wrapper CLI mode."""
        mock_execute.return_value = "Wrapper response"

        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_WRAPPER_CLI)

        # Reset context
        _recursion_depth.set(0)
        _call_stack.set([])

        result = await client.chat("Hello")
        assert result == "Wrapper response"
        mock_execute.assert_called_once_with("chat", "Hello")

    @patch("claude_wrapper.core.unified_client.ClaudeHTTPClient")
    async def test_chat_api_mode(self, mock_http_client_class):
        """Test chat in API mode."""
        mock_http_client = AsyncMock()
        mock_http_client.chat = AsyncMock(return_value="API response")
        mock_http_client_class.return_value = mock_http_client

        client = UnifiedClaudeClient(mode=ClientMode.API)

        # Reset context
        _recursion_depth.set(0)
        _call_stack.set([])

        result = await client.chat("Hello", temperature=0.7)
        assert result == "API response"
        mock_http_client.chat.assert_called_once_with("Hello", temperature=0.7)

    async def test_chat_api_mode_no_client(self):
        """Test chat in API mode without HTTP client."""
        client = UnifiedClaudeClient(mode=ClientMode.API)
        client._http_client = None  # Force no client

        # Reset context
        _recursion_depth.set(0)
        _call_stack.set([])

        with pytest.raises(ClaudeWrapperError, match="HTTP client not initialized"):
            await client.chat("Hello")

    async def test_chat_unsupported_mode(self):
        """Test chat with unsupported mode."""
        client = UnifiedClaudeClient()

        # Reset context
        _recursion_depth.set(0)
        _call_stack.set([])

        # Force invalid mode with proper structure
        invalid_mode = MagicMock()
        invalid_mode.value = "invalid"
        client.mode = invalid_mode

        with pytest.raises(ClaudeWrapperError, match="Unsupported client mode"):
            await client.chat("Hello")

    async def test_chat_recursion_limit(self):
        """Test chat recursion limit."""
        client = UnifiedClaudeClient(max_recursion_depth=2)

        # Set high recursion depth
        _recursion_depth.set(5)
        _call_stack.set(["op1", "op2", "op3", "op4", "op5"])

        with pytest.raises(RecursionError, match="Maximum recursion depth"):
            await client.chat("Hello")

    @patch.object(UnifiedClaudeClient, "_execute_claude_cli")
    async def test_stream_chat_claude_cli_mode(self, mock_execute):
        """Test stream chat in Claude CLI mode."""
        mock_execute.return_value = "Word1 Word2"

        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI)

        # Reset context
        _recursion_depth.set(0)
        _call_stack.set([])

        chunks = []
        async for chunk in client.stream_chat("Hello"):
            chunks.append(chunk)

        assert chunks == ["Word1", " ", "Word2"]

    @patch.object(UnifiedClaudeClient, "_execute_claude_wrapper_cli")
    async def test_stream_chat_wrapper_cli_mode(self, mock_execute):
        """Test stream chat in claude-wrapper CLI mode."""
        mock_execute.return_value = "Stream response"

        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_WRAPPER_CLI)

        # Reset context
        _recursion_depth.set(0)
        _call_stack.set([])

        chunks = []
        async for chunk in client.stream_chat("Hello"):
            chunks.append(chunk)

        # Should simulate streaming
        assert "Stream" in "".join(chunks)

    @patch("claude_wrapper.core.unified_client.ClaudeHTTPClient")
    async def test_stream_chat_api_mode(self, mock_http_client_class):
        """Test stream chat in API mode."""
        mock_http_client = AsyncMock()

        async def mock_stream(message, **kwargs):  # noqa: ARG001
            yield "Chunk1"
            yield "Chunk2"

        mock_http_client.stream_chat = mock_stream
        mock_http_client_class.return_value = mock_http_client

        client = UnifiedClaudeClient(mode=ClientMode.API)

        # Reset context
        _recursion_depth.set(0)
        _call_stack.set([])

        chunks = []
        async for chunk in client.stream_chat("Hello"):
            chunks.append(chunk)

        assert chunks == ["Chunk1", "Chunk2"]

    async def test_stream_chat_api_mode_no_client(self):
        """Test stream chat in API mode without HTTP client."""
        client = UnifiedClaudeClient(mode=ClientMode.API)
        client._http_client = None

        # Reset context
        _recursion_depth.set(0)
        _call_stack.set([])

        with pytest.raises(ClaudeWrapperError, match="HTTP client not initialized"):
            async for _ in client.stream_chat("Hello"):
                pass

    async def test_stream_chat_unsupported_mode(self):
        """Test stream chat with unsupported mode."""
        client = UnifiedClaudeClient()

        # Reset context
        _recursion_depth.set(0)
        _call_stack.set([])

        # Force invalid mode with proper structure
        invalid_mode = MagicMock()
        invalid_mode.value = "invalid"
        client.mode = invalid_mode

        with pytest.raises(ClaudeWrapperError, match="Unsupported client mode"):
            async for _ in client.stream_chat("Hello"):
                pass

    @patch.object(UnifiedClaudeClient, "_check_claude_auth")
    async def test_health_check_claude_cli_mode(self, mock_auth):
        """Test health check in Claude CLI mode."""
        mock_auth.return_value = True

        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI)
        result = await client.health_check()

        assert result is True
        mock_auth.assert_called_once()

    @patch("shutil.which")
    @patch("asyncio.create_subprocess_exec")
    async def test_health_check_wrapper_cli_mode(self, mock_subprocess, mock_which):
        """Test health check in claude-wrapper CLI mode."""
        mock_which.return_value = "/usr/bin/claude-wrapper"

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"v1.0", b""))
        mock_subprocess.return_value = mock_process

        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_WRAPPER_CLI)
        result = await client.health_check()

        assert result is True

    @patch("shutil.which")
    async def test_health_check_wrapper_cli_not_found(self, mock_which):
        """Test health check when wrapper CLI not found."""
        mock_which.return_value = None

        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_WRAPPER_CLI)

        with pytest.raises(ClaudeWrapperError, match="health check failed"):
            await client.health_check()

    @patch("claude_wrapper.core.unified_client.ClaudeHTTPClient")
    async def test_health_check_api_mode(self, mock_http_client_class):
        """Test health check in API mode."""
        mock_http_client = AsyncMock()
        mock_http_client.health_check = AsyncMock(return_value=True)
        mock_http_client_class.return_value = mock_http_client

        client = UnifiedClaudeClient(mode=ClientMode.API)
        result = await client.health_check()

        assert result is True
        mock_http_client.health_check.assert_called_once()

    async def test_health_check_api_mode_no_client(self):
        """Test health check in API mode without HTTP client."""
        client = UnifiedClaudeClient(mode=ClientMode.API)
        client._http_client = None

        with pytest.raises(ClaudeWrapperError, match="HTTP client not initialized"):
            await client.health_check()

    async def test_health_check_unsupported_mode(self):
        """Test health check with unsupported mode."""
        client = UnifiedClaudeClient()

        # Force invalid mode with proper structure
        invalid_mode = MagicMock()
        invalid_mode.value = "invalid"
        client.mode = invalid_mode

        with pytest.raises(ClaudeWrapperError, match="Unsupported client mode"):
            await client.health_check()

    def test_get_recursion_info_disabled(self):
        """Test getting recursion info when tracking disabled."""
        client = UnifiedClaudeClient(enable_recursion_tracking=False)
        info = client.get_recursion_info()

        assert info == {"tracking_enabled": False}

    def test_get_recursion_info_with_context(self):
        """Test getting recursion info with context set."""
        client = UnifiedClaudeClient(max_recursion_depth=10)

        _recursion_depth.set(3)
        _call_stack.set(["op1", "op2", "op3"])

        if hasattr(_thread_local, "depth"):
            _thread_local.depth = 2
            _thread_local.stack = ["thread_op1", "thread_op2"]

        info = client.get_recursion_info()

        assert info["tracking_enabled"] is True
        assert info["max_depth"] == 10
        assert info["current_depth"] == 3
        assert info["call_stack"] == ["op1", "op2", "op3"]
        assert info["context_depth"] == 3

    def test_get_recursion_info_no_context(self):
        """Test getting recursion info with no context."""
        client = UnifiedClaudeClient()

        # Reset context to clean state
        _recursion_depth.set(0)
        _call_stack.set([])

        # Clear thread local
        if hasattr(_thread_local, "depth"):
            delattr(_thread_local, "depth")
        if hasattr(_thread_local, "stack"):
            delattr(_thread_local, "stack")

        info = client.get_recursion_info()

        assert info["tracking_enabled"] is True
        assert info["current_depth"] == 0
        assert info["call_stack"] == []

    def test_get_recursion_info_thread_local_priority(self):
        """Test recursion info with thread local having higher depth."""
        client = UnifiedClaudeClient()

        _recursion_depth.set(2)
        _call_stack.set(["ctx_op1", "ctx_op2"])

        _thread_local.depth = 5
        _thread_local.stack = ["thread_op1", "thread_op2", "thread_op3", "thread_op4", "thread_op5"]

        info = client.get_recursion_info()

        assert info["current_depth"] == 5  # Max of context and thread
        assert info["context_depth"] == 2
        assert info["thread_depth"] == 5

    def test_create_claude_client(self):
        """Test create_claude_client factory function."""
        client = create_claude_client(timeout=60.0)
        assert client.mode == ClientMode.CLAUDE_CLI
        assert client.timeout == 60.0

    def test_create_claude_wrapper_client(self):
        """Test create_claude_wrapper_client factory function."""
        client = create_claude_wrapper_client(retry_attempts=5)
        assert client.mode == ClientMode.CLAUDE_WRAPPER_CLI
        assert client.retry_attempts == 5

    def test_create_api_client(self):
        """Test create_api_client factory function."""
        client = create_api_client(api_key="test-key")
        assert client.mode == ClientMode.API
        assert client._http_client is not None

    def test_client_mode_enum(self):
        """Test ClientMode enum values."""
        assert ClientMode.CLAUDE_CLI.value == "claude-cli"
        assert ClientMode.CLAUDE_WRAPPER_CLI.value == "claude-wrapper-cli"
        assert ClientMode.API.value == "api"

    def test_recursion_error_inheritance(self):
        """Test RecursionError inherits from ClaudeWrapperError."""
        error = RecursionError("test")
        assert isinstance(error, ClaudeWrapperError)
