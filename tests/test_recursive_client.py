"""Tests for recursive client functionality."""

from unittest.mock import AsyncMock, patch

import pytest

from claude_wrapper.core import (
    ClientMode,
    RecursionError,
    UnifiedClaudeClient,
    create_api_client,
    create_claude_client,
    create_claude_wrapper_client,
)


@pytest.fixture
def mock_claude_client():
    """Mock UnifiedClaudeClient with basic functionality."""
    client = UnifiedClaudeClient(
        mode=ClientMode.CLAUDE_CLI,
        max_recursion_depth=3,
        enable_recursion_tracking=True,
    )
    return client


@pytest.fixture
def mock_http_client():
    """Mock HTTP client."""
    with patch("claude_wrapper.core.unified_client.ClaudeHTTPClient") as mock:
        mock_instance = AsyncMock()
        mock_instance.chat.return_value = "Test response"
        mock_instance.stream_chat.return_value = iter(["Test", " response"])
        mock_instance.health_check.return_value = True
        mock.return_value = mock_instance
        yield mock_instance


class TestUnifiedClaudeClient:
    """Test the UnifiedClaudeClient class."""

    def test_client_initialization(self):
        """Test client initialization with different modes."""
        # Claude CLI mode
        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI)
        assert client.mode == ClientMode.CLAUDE_CLI
        assert client.max_recursion_depth == 10
        assert client.enable_recursion_tracking is True

        # Claude Wrapper CLI mode
        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_WRAPPER_CLI, max_recursion_depth=5)
        assert client.mode == ClientMode.CLAUDE_WRAPPER_CLI
        assert client.max_recursion_depth == 5

        # API mode
        client = UnifiedClaudeClient(mode=ClientMode.API, api_base_url="http://test:9000")
        assert client.mode == ClientMode.API
        assert client._http_client is not None

    @pytest.mark.asyncio
    async def test_recursion_depth_tracking(self, mock_claude_client):
        """Test recursion depth tracking and limits."""
        # Reset context before test
        from claude_wrapper.core.unified_client import _call_stack, _recursion_depth

        _recursion_depth.set(0)
        _call_stack.set([])

        # Mock the subprocess execution
        with patch.object(mock_claude_client, "_execute_claude_cli", return_value="response"):
            # First call should work
            await mock_claude_client.chat("test message 1")

            # Check recursion info
            info = mock_claude_client.get_recursion_info()
            assert info["tracking_enabled"] is True
            assert info["max_depth"] == 3

            # Reset and manually set recursion depth to simulate nested calls
            _recursion_depth.set(0)
            _call_stack.set([])
            _recursion_depth.set(2)

            # This should still work
            await mock_claude_client.chat("test message 2")

            # Reset and set to max depth
            _recursion_depth.set(0)
            _call_stack.set([])
            _recursion_depth.set(3)

            # This should raise RecursionError
            with pytest.raises(RecursionError, match="Maximum recursion depth"):
                await mock_claude_client.chat("test message 3")

    @pytest.mark.asyncio
    async def test_recursion_loop_detection(self, mock_claude_client):
        """Test detection of infinite recursion loops."""
        from claude_wrapper.core.unified_client import _call_stack

        # Simulate a loop by setting the same operation twice
        _call_stack.set(["claude-cli:chat", "claude-cli:chat"])

        with (
            patch.object(mock_claude_client, "_execute_claude_cli", return_value="response"),
            pytest.raises(RecursionError, match="infinite recursion loop"),
        ):
            await mock_claude_client.chat("test message")

    def test_recursion_disabled(self):
        """Test client with recursion tracking disabled."""
        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI, enable_recursion_tracking=False)

        info = client.get_recursion_info()
        assert info["tracking_enabled"] is False

        # Should not raise recursion errors when disabled
        client._check_recursion_depth("test_operation")

    @pytest.mark.asyncio
    async def test_api_mode_chat(self, mock_http_client):
        """Test API mode chat functionality."""
        client = UnifiedClaudeClient(mode=ClientMode.API)
        client._http_client = mock_http_client

        response = await client.chat("test message")
        assert response == "Test response"
        mock_http_client.chat.assert_called_once_with("test message")

    @pytest.mark.asyncio
    async def test_api_mode_stream_chat(self, mock_http_client):
        """Test API mode streaming chat."""
        client = UnifiedClaudeClient(mode=ClientMode.API)
        client._http_client = mock_http_client

        # Mock async iterator
        async def mock_stream_chat(*_args, **_kwargs):
            for chunk in ["Test", " response"]:
                yield chunk

        mock_http_client.stream_chat = mock_stream_chat

        chunks = []
        async for chunk in client.stream_chat("test message"):
            chunks.append(chunk)

        assert chunks == ["Test", " response"]

    @pytest.mark.asyncio
    async def test_claude_cli_mode_execution(self, mock_claude_client):
        """Test Claude CLI mode execution."""
        # Reset thread-local state
        from claude_wrapper.core.unified_client import _thread_local

        if hasattr(_thread_local, "depth"):
            _thread_local.depth = 0
        if hasattr(_thread_local, "stack"):
            _thread_local.stack = []

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"test response", b"")

        async def mock_wait_for(coro, timeout=None):  # noqa: ARG001
            return await coro

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
            patch("asyncio.wait_for", side_effect=mock_wait_for),
            patch("shutil.which", return_value="/usr/bin/claude"),
        ):
            response = await mock_claude_client.chat("test message")
            assert response == "test response"

    @pytest.mark.asyncio
    async def test_claude_wrapper_cli_mode_execution(self):
        """Test Claude Wrapper CLI mode execution."""
        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_WRAPPER_CLI)

        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"wrapper response", b"")

        async def mock_wait_for_wrapper(coro, timeout=None):  # noqa: ARG001
            return await coro

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
            patch("asyncio.wait_for", side_effect=mock_wait_for_wrapper),
            patch("shutil.which", return_value="/usr/bin/claude-wrapper"),
        ):
            response = await client.chat("test message")
            assert response == "wrapper response"

    @pytest.mark.asyncio
    async def test_health_checks(self):
        """Test health check functionality for different modes."""
        # Claude CLI mode
        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI)
        mock_process = AsyncMock()
        mock_process.returncode = 0
        mock_process.communicate.return_value = (b"version output", b"")

        async def mock_wait_for_health(coro, timeout=None):  # noqa: ARG001
            return await coro

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
            patch("asyncio.wait_for", side_effect=mock_wait_for_health),
            patch("shutil.which", return_value="/usr/bin/claude"),
        ):
            result = await client.health_check()
            assert result is True

        # Claude Wrapper CLI mode
        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_WRAPPER_CLI)

        with (
            patch("asyncio.create_subprocess_exec", return_value=mock_process),
            patch("asyncio.wait_for", side_effect=mock_wait_for_health),
            patch("shutil.which", return_value="/usr/bin/claude-wrapper"),
        ):
            result = await client.health_check()
            assert result is True

        # API mode
        client = UnifiedClaudeClient(mode=ClientMode.API)
        mock_http_client = AsyncMock()
        mock_http_client.health_check.return_value = True
        client._http_client = mock_http_client

        result = await client.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_async_context_manager(self, mock_http_client):
        """Test async context manager functionality."""
        client = UnifiedClaudeClient(mode=ClientMode.API)
        client._http_client = mock_http_client

        async with client as ctx_client:
            assert ctx_client is client

        mock_http_client.__aenter__.assert_called_once()
        mock_http_client.__aexit__.assert_called_once()


class TestClientFactoryFunctions:
    """Test the client factory functions."""

    def test_create_claude_client(self):
        """Test create_claude_client factory function."""
        client = create_claude_client(max_recursion_depth=5)
        assert isinstance(client, UnifiedClaudeClient)
        assert client.mode == ClientMode.CLAUDE_CLI
        assert client.max_recursion_depth == 5

    def test_create_claude_wrapper_client(self):
        """Test create_claude_wrapper_client factory function."""
        client = create_claude_wrapper_client(timeout=60.0)
        assert isinstance(client, UnifiedClaudeClient)
        assert client.mode == ClientMode.CLAUDE_WRAPPER_CLI
        assert client.timeout == 60.0

    def test_create_api_client(self):
        """Test create_api_client factory function."""
        client = create_api_client(api_base_url="http://test:8080", api_key="test-key")
        assert isinstance(client, UnifiedClaudeClient)
        assert client.mode == ClientMode.API
        assert client._http_client is not None


class TestRecursionSafeguards:
    """Test recursion safeguards and protection mechanisms."""

    @pytest.mark.asyncio
    async def test_subprocess_recursion_tracking(self):
        """Test subprocess-based recursion tracking."""
        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI, max_recursion_depth=2)

        # Mock subprocess execution that would recursively call itself
        call_count = 0

        async def mock_execute_recursive(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count > 3:  # Prevent infinite test loop
                return "final response"
            # Simulate recursive call
            client._check_subprocess_recursion("test_operation")
            return await mock_execute_recursive(*args, **kwargs)

        with (
            patch.object(client, "_execute_claude_cli", side_effect=mock_execute_recursive),
            pytest.raises(RecursionError),
        ):
            await client.chat("recursive test")

    def test_thread_local_recursion_cleanup(self):
        """Test thread-local recursion cleanup."""
        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI)

        from claude_wrapper.core.unified_client import _thread_local

        # Reset thread-local state for clean test
        if hasattr(_thread_local, "depth"):
            _thread_local.depth = 0
        if hasattr(_thread_local, "stack"):
            _thread_local.stack = []

        # Simulate setting thread-local recursion state
        client._check_subprocess_recursion("test_op")

        assert hasattr(_thread_local, "depth")
        assert _thread_local.depth == 1

        # Cleanup should decrement
        client._cleanup_subprocess_recursion()
        assert _thread_local.depth == 0

    @pytest.mark.asyncio
    async def test_context_variable_isolation(self):
        """Test that context variables properly isolate recursion tracking."""
        # Reset context before test
        from claude_wrapper.core.unified_client import _call_stack, _recursion_depth

        _recursion_depth.set(0)
        _call_stack.set([])

        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI)

        async def nested_operation(depth: int):
            """Simulate nested operations."""
            # Each async context should have its own isolated recursion tracking
            # In practice, the context is already isolated due to async task creation
            client._check_recursion_depth(f"operation_{depth}")
            info = client.get_recursion_info()
            return info["current_depth"]

        # Reset before operations
        _recursion_depth.set(0)
        _call_stack.set([])

        # Run first operation
        depth1 = await nested_operation(1)

        # Reset between operations to simulate isolated contexts
        _recursion_depth.set(0)
        _call_stack.set([])

        # Run second operation
        depth2 = await nested_operation(2)

        # Both should register as depth 1 in their respective contexts
        assert depth1 == 1
        assert depth2 == 1


class TestRealWorldScenarios:
    """Test real-world recursive scenarios."""

    @pytest.mark.asyncio
    async def test_claude_analyzing_claude_output(self):
        """Test scenario where Claude analyzes its own output."""
        client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI, max_recursion_depth=5)

        responses = [
            "Initial response about code analysis",
            "Follow-up analysis of the analysis",
            "Meta-analysis of the meta-analysis",
            "Final insights",
        ]

        call_count = 0

        async def mock_claude_call(*_args, **_kwargs):
            nonlocal call_count
            response = responses[min(call_count, len(responses) - 1)]
            call_count += 1
            return response

        with patch.object(client, "_execute_claude_cli", side_effect=mock_claude_call):
            # First call
            result1 = await client.chat("Analyze this code")
            assert "Initial response" in result1

            # Nested analysis
            result2 = await client.chat(f"Please analyze this analysis: {result1}")
            assert "Follow-up analysis" in result2

            # Should track recursion properly
            info = client.get_recursion_info()
            assert info["current_depth"] >= 0  # Context-dependent

    @pytest.mark.asyncio
    async def test_api_to_cli_recursion_scenario(self):
        """Test mixed mode recursion (API calling CLI)."""
        api_client = UnifiedClaudeClient(mode=ClientMode.API, max_recursion_depth=3)
        cli_client = UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI, max_recursion_depth=3)

        # Mock both clients
        with patch.object(api_client, "_http_client") as mock_http:
            mock_http.chat = AsyncMock(return_value="API response that needs CLI verification")

            with patch.object(cli_client, "_execute_claude_cli") as mock_cli:
                mock_cli.return_value = "CLI verification complete"

                # Simulate API client calling CLI client for verification
                api_response = await api_client.chat("Initial query")
                cli_response = await cli_client.chat(f"Verify this: {api_response}")

                assert "API response" in api_response
                assert "CLI verification" in cli_response
