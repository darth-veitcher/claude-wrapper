"""Tests for HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from claude_wrapper.core.exceptions import (
    ClaudeAuthError,
    ClaudeExecutionError,
    ClaudeTimeoutError,
    ClaudeWrapperError,
)
from claude_wrapper.core.http_client import ClaudeHTTPClient


@pytest.fixture
async def http_client():
    """Create an HTTP client instance."""
    client = ClaudeHTTPClient(
        base_url="http://localhost:8000",
        timeout=10.0,
        api_key="test-api-key",
    )
    yield client
    await client.close()


@pytest.fixture
async def http_client_no_auth():
    """Create an HTTP client without authentication."""
    client = ClaudeHTTPClient(
        base_url="http://localhost:8000/",  # Test trailing slash removal
        timeout=10.0,
    )
    yield client
    await client.close()


class TestClaudeHTTPClient:
    """Test ClaudeHTTPClient functionality."""

    async def test_init_with_defaults(self):
        """Test initialization with default values."""
        client = ClaudeHTTPClient()
        assert client.base_url == "http://localhost:8000"
        assert client.timeout == 120.0
        assert client.api_key is None
        assert client._client is None
        await client.close()

    async def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        client = ClaudeHTTPClient(
            base_url="https://api.example.com/",
            timeout=30.0,
            api_key="secret-key",
        )
        assert client.base_url == "https://api.example.com"
        assert client.timeout == 30.0
        assert client.api_key == "secret-key"
        await client.close()

    async def test_context_manager(self, http_client):  # noqa: ARG002
        """Test async context manager."""
        async with ClaudeHTTPClient() as client:
            assert client._client is not None
            assert isinstance(client._client, httpx.AsyncClient)
        # Client should be closed after context exit
        assert client._client is None

    async def test_ensure_client_creates_client(self, http_client):
        """Test that _ensure_client creates an HTTP client."""
        assert http_client._client is None
        client = await http_client._ensure_client()
        assert client is not None
        assert isinstance(client, httpx.AsyncClient)
        assert http_client._client is client

    async def test_ensure_client_with_auth(self, http_client):
        """Test client creation with authentication."""
        client = await http_client._ensure_client()
        assert "Authorization" in client.headers
        assert client.headers["Authorization"] == "Bearer test-api-key"

    async def test_ensure_client_without_auth(self, http_client_no_auth):
        """Test client creation without authentication."""
        client = await http_client_no_auth._ensure_client()
        assert "Authorization" not in client.headers
        assert "Content-Type" in client.headers

    async def test_ensure_client_reuses_existing(self, http_client):
        """Test that _ensure_client reuses existing client."""
        client1 = await http_client._ensure_client()
        client2 = await http_client._ensure_client()
        assert client1 is client2

    async def test_close(self, http_client):
        """Test closing the HTTP client."""
        await http_client._ensure_client()
        assert http_client._client is not None
        await http_client.close()
        assert http_client._client is None

    async def test_close_when_not_initialized(self, http_client):
        """Test closing when client is not initialized."""
        assert http_client._client is None
        await http_client.close()  # Should not raise
        assert http_client._client is None

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_chat_success(self, mock_client_class, http_client):
        """Test successful chat request."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Hello, I'm Claude!"}}]
        }
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        response = await http_client.chat("Hello")
        assert response == "Hello, I'm Claude!"

        mock_client.post.assert_called_once_with(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "Hello"}],
            },
        )

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_chat_with_custom_model(self, mock_client_class, http_client):
        """Test chat with custom model."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        response = await http_client.chat("Test", model="opus")
        assert response == "Response"

        mock_client.post.assert_called_once_with(
            "/v1/chat/completions",
            json={
                "model": "opus",
                "messages": [{"role": "user", "content": "Test"}],
            },
        )

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_chat_with_extra_kwargs(self, mock_client_class, http_client):
        """Test chat with additional parameters."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": [{"message": {"content": "Response"}}]}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        response = await http_client.chat("Test", temperature=0.7, max_tokens=100)
        assert response == "Response"

        mock_client.post.assert_called_once_with(
            "/v1/chat/completions",
            json={
                "model": "sonnet",
                "messages": [{"role": "user", "content": "Test"}],
                "temperature": 0.7,
                "max_tokens": 100,
            },
        )

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_chat_invalid_response(self, mock_client_class, http_client):
        """Test chat with invalid response format."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"invalid": "response"}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        with pytest.raises(ClaudeWrapperError, match="Unexpected API error"):
            await http_client.chat("Test")

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_chat_empty_choices(self, mock_client_class, http_client):
        """Test chat with empty choices array."""
        mock_response = MagicMock()
        mock_response.json.return_value = {"choices": []}
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        with pytest.raises(ClaudeWrapperError, match="Unexpected API error"):
            await http_client.chat("Test")

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_chat_timeout(self, mock_client_class, http_client):
        """Test chat request timeout."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client_class.return_value = mock_client

        with pytest.raises(ClaudeTimeoutError, match="API request timed out"):
            await http_client.chat("Test")

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_chat_auth_error(self, mock_client_class, http_client):
        """Test chat with authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        error = httpx.HTTPStatusError("Auth failed", request=None, response=mock_response)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=error)
        mock_client_class.return_value = mock_client

        with pytest.raises(ClaudeAuthError, match="API authentication failed"):
            await http_client.chat("Test")

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_chat_http_error(self, mock_client_class, http_client):
        """Test chat with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        error = httpx.HTTPStatusError("Server error", request=None, response=mock_response)

        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=error)
        mock_client_class.return_value = mock_client

        with pytest.raises(ClaudeExecutionError, match="API request failed"):
            await http_client.chat("Test")

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_chat_unexpected_error(self, mock_client_class, http_client):
        """Test chat with unexpected error."""
        mock_client = AsyncMock()
        mock_client.post = AsyncMock(side_effect=ValueError("Unexpected"))
        mock_client_class.return_value = mock_client

        with pytest.raises(ClaudeWrapperError, match="Unexpected API error"):
            await http_client.chat("Test")

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_stream_chat_success(self, mock_client_class, http_client):
        """Test successful streaming chat."""
        # Create mock response with SSE data
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield 'data: {"choices": [{"delta": {"content": "Hello"}}]}'
            yield 'data: {"choices": [{"delta": {"content": " world"}}]}'
            yield 'data: {"choices": [{"delta": {}}]}'  # No content
            yield "data: [DONE]"

        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_client.stream.return_value.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        chunks = []
        async for chunk in http_client.stream_chat("Test"):
            chunks.append(chunk)

        assert chunks == ["Hello", " world"]

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_stream_chat_malformed_json(self, mock_client_class, http_client):
        """Test streaming with malformed JSON lines."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield 'data: {"choices": [{"delta": {"content": "Valid"}}]}'
            yield "data: {malformed json"  # Invalid JSON
            yield 'data: {"choices": [{"delta": {"content": " data"}}]}'
            yield "data: [DONE]"

        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_client.stream.return_value.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        chunks = []
        async for chunk in http_client.stream_chat("Test"):
            chunks.append(chunk)

        assert chunks == ["Valid", " data"]  # Malformed line should be skipped

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_stream_chat_timeout(self, mock_client_class, http_client):
        """Test streaming timeout."""
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=httpx.TimeoutException("Timeout"))
        mock_client_class.return_value = mock_client

        with pytest.raises(ClaudeTimeoutError, match="Streaming API request timed out"):
            async for _ in http_client.stream_chat("Test"):
                pass

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_stream_chat_auth_error(self, mock_client_class, http_client):
        """Test streaming with authentication error."""
        mock_response = MagicMock()
        mock_response.status_code = 401
        error = httpx.HTTPStatusError("Auth failed", request=None, response=mock_response)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=error)
        mock_client_class.return_value = mock_client

        with pytest.raises(ClaudeAuthError, match="API authentication failed"):
            async for _ in http_client.stream_chat("Test"):
                pass

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_stream_chat_http_error(self, mock_client_class, http_client):
        """Test streaming with HTTP error."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        error = httpx.HTTPStatusError("Server error", request=None, response=mock_response)

        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=error)
        mock_client_class.return_value = mock_client

        with pytest.raises(ClaudeExecutionError, match="Streaming API request failed"):
            async for _ in http_client.stream_chat("Test"):
                pass

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_stream_chat_unexpected_error(self, mock_client_class, http_client):
        """Test streaming with unexpected error."""
        mock_client = AsyncMock()
        mock_client.stream = MagicMock(side_effect=ValueError("Unexpected"))
        mock_client_class.return_value = mock_client

        with pytest.raises(ClaudeWrapperError, match="Unexpected streaming API error"):
            async for _ in http_client.stream_chat("Test"):
                pass

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_health_check_success(self, mock_client_class, http_client):
        """Test successful health check."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()

        mock_client = AsyncMock()
        mock_client.get = AsyncMock(return_value=mock_response)
        mock_client_class.return_value = mock_client

        result = await http_client.health_check()
        assert result is True

        mock_client.get.assert_called_once_with("/")

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_health_check_failure(self, mock_client_class, http_client):
        """Test health check failure."""
        mock_client = AsyncMock()
        mock_client.get = AsyncMock(side_effect=httpx.ConnectError("Connection failed"))
        mock_client_class.return_value = mock_client

        with pytest.raises(ClaudeWrapperError, match="API server health check failed"):
            await http_client.health_check()

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_stream_chat_with_extra_kwargs(self, mock_client_class, http_client):
        """Test streaming with additional parameters."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield 'data: {"choices": [{"delta": {"content": "Response"}}]}'
            yield "data: [DONE]"

        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_stream = MagicMock()
        mock_stream.__aenter__ = AsyncMock(return_value=mock_response)
        mock_stream.__aexit__ = AsyncMock()
        mock_client.stream = MagicMock(return_value=mock_stream)
        mock_client_class.return_value = mock_client

        chunks = []
        async for chunk in http_client.stream_chat("Test", model="opus", temperature=0.5):
            chunks.append(chunk)

        assert chunks == ["Response"]

        mock_client.stream.assert_called_once_with(
            "POST",
            "/v1/chat/completions",
            json={
                "model": "opus",
                "messages": [{"role": "user", "content": "Test"}],
                "stream": True,
                "temperature": 0.5,
            },
        )

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_stream_chat_empty_content(self, mock_client_class, http_client):
        """Test streaming with empty content in delta."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield 'data: {"choices": [{"delta": {"content": "Test"}}]}'
            yield 'data: {"choices": [{"delta": {"content": ""}}]}'  # Empty content
            yield 'data: {"choices": [{"delta": {"content": null}}]}'  # Null content
            yield "data: [DONE]"

        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_client.stream.return_value.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        chunks = []
        async for chunk in http_client.stream_chat("Test"):
            chunks.append(chunk)

        assert chunks == ["Test"]  # Empty/null content should be skipped

    @patch("claude_wrapper.core.http_client.httpx.AsyncClient")
    async def test_stream_chat_non_data_lines(self, mock_client_class, http_client):
        """Test streaming with non-data lines."""
        mock_response = AsyncMock()
        mock_response.raise_for_status = MagicMock()

        async def mock_aiter_lines():
            yield ""  # Empty line
            yield "event: message"  # Event line
            yield 'data: {"choices": [{"delta": {"content": "Test"}}]}'
            yield "  "  # Whitespace only
            yield "data: [DONE]"

        mock_response.aiter_lines = mock_aiter_lines

        mock_client = AsyncMock()
        mock_client.stream = MagicMock()
        mock_client.stream.return_value.__aenter__ = AsyncMock(return_value=mock_response)
        mock_client.stream.return_value.__aexit__ = AsyncMock()
        mock_client_class.return_value = mock_client

        chunks = []
        async for chunk in http_client.stream_chat("Test"):
            chunks.append(chunk)

        assert chunks == ["Test"]  # Non-data lines should be ignored
