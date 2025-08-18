"""End-to-end integration tests for Claude Wrapper."""

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from claude_wrapper.api.server import app
from claude_wrapper.core import ClaudeClient


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_chat_workflow(self):
        """Test complete chat workflow through API."""
        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            # Mock Claude responses
            mock_client.chat = AsyncMock()
            mock_client.chat.return_value = "Hello! How can I help you?"

            client = TestClient(app)

            # Create chat request
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Hello Claude"}],
                    "model": "sonnet",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["content"] == "Hello! How can I help you?"

    @pytest.mark.asyncio
    async def test_streaming_workflow(self):
        """Test streaming from client through API."""
        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            # Mock streaming
            async def mock_stream(*_args, **_kwargs):
                for chunk in ["Hello ", "from ", "streaming ", "Claude!"]:
                    yield chunk

            mock_client.stream_chat = mock_stream

            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions",
                json={"messages": [{"role": "user", "content": "Stream test"}], "stream": True},
            )
            assert response.status_code == 200

            # Verify streaming response headers
            assert "text/event-stream" in response.headers.get("content-type", "")

    @pytest.mark.asyncio
    @pytest.mark.requires_claude
    async def test_real_claude_cli(self):
        """Test with real Claude CLI (if available)."""
        # This test requires Claude CLI to be installed and authenticated
        # Skip if not available
        client = ClaudeClient()

        try:
            authenticated = await client.check_auth()
            if not authenticated:
                pytest.skip("Claude CLI not authenticated")
        except Exception:
            pytest.skip("Claude CLI not available")

        # Simple test with real CLI
        response = await client.chat("What is 2+2? Answer with just the number.")
        assert "4" in response

    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error propagation through the stack."""
        from claude_wrapper.core.exceptions import ClaudeAuthError

        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            mock_client.chat = AsyncMock()
            mock_client.chat.side_effect = ClaudeAuthError("Not authenticated")

            client = TestClient(app)

            response = client.post(
                "/v1/chat/completions", json={"messages": [{"role": "user", "content": "Test"}]}
            )

            assert response.status_code == 500
            assert "Not authenticated" in response.json()["detail"]

    @pytest.mark.asyncio
    async def test_api_with_authentication(self):
        """Test API with authentication enabled."""
        # Enable API key authentication
        test_api_key = "test-secret-key"

        with patch("claude_wrapper.api.server.config") as mock_config:
            mock_config.api_key = test_api_key

            client = TestClient(app)

            # Request without API key should fail
            response = client.get("/v1/models")
            assert response.status_code == 401

            # Request with wrong API key should fail
            response = client.get("/v1/models", headers={"Authorization": "Bearer wrong-key"})
            assert response.status_code == 401

            # Request with correct API key should succeed
            response = client.get("/v1/models", headers={"Authorization": f"Bearer {test_api_key}"})
            assert response.status_code == 200

    @pytest.mark.asyncio
    async def test_openai_client_compatibility(self):
        """Test compatibility with OpenAI Python client."""
        # This simulates how the OpenAI client would interact with our API
        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            mock_client.chat = AsyncMock()
            mock_client.chat.return_value = "Compatible response"

            client = TestClient(app)

            # Simulate OpenAI client request format
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "sonnet",  # Our model name
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Hello"},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 150,
                },
            )

            assert response.status_code == 200

            # Verify OpenAI-compatible response format
            data = response.json()
            assert "id" in data
            assert data["object"] == "chat.completion"
            assert "created" in data
            assert "choices" in data
            assert len(data["choices"]) == 1
            assert "message" in data["choices"][0]
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["choices"][0]["message"]["content"] == "Compatible response"
            assert "usage" in data
