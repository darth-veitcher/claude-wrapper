"""Tests for the FastAPI server."""

from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from claude_wrapper.api.server import app


@pytest.mark.asyncio
@pytest.mark.api
class TestAPIEndpoints:
    """Test API endpoints."""

    async def test_root_endpoint(self):
        """Test root endpoint."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/")
            assert response.status_code == 200
            data = response.json()
            assert data["message"] == "Claude Wrapper API"
            assert "endpoints" in data

    async def test_health_endpoint(self):
        """Test health endpoint."""
        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            mock_client.check_auth = AsyncMock(return_value=True)

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.get("/health")
                assert response.status_code == 200
                data = response.json()
                assert data["status"] == "healthy"
                assert data["claude_cli"] == "authenticated"

    async def test_models_endpoint(self):
        """Test models listing endpoint."""
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/v1/models")
            assert response.status_code == 200
            data = response.json()
            assert "data" in data
            assert len(data["data"]) > 0
            # Check for our simplified model names
            model_ids = [model["id"] for model in data["data"]]
            assert "opus" in model_ids
            assert "sonnet" in model_ids
            assert "haiku" in model_ids

    async def test_chat_completions_endpoint(self):
        """Test chat completions endpoint."""
        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            mock_client.chat = AsyncMock(return_value="Test response from Claude")

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "sonnet",
                        "messages": [{"role": "user", "content": "Hello"}],
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["choices"][0]["message"]["content"] == "Test response from Claude"
                assert data["choices"][0]["message"]["role"] == "assistant"
                assert "usage" in data

    async def test_completions_endpoint(self):
        """Test completions endpoint."""
        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            mock_client.complete = AsyncMock(return_value="Completion response")

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/completions",
                    json={
                        "model": "sonnet",
                        "prompt": "Complete this sentence:",
                    },
                )

                assert response.status_code == 200
                data = response.json()
                assert data["choices"][0]["text"] == "Completion response"
                assert "usage" in data

    async def test_streaming_chat_completions(self):
        """Test streaming chat completions."""
        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            # Mock streaming response
            async def mock_stream(*_args, **_kwargs):
                for chunk in ["Hello ", "streaming ", "world!"]:
                    yield chunk

            mock_client.stream_chat = mock_stream

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "sonnet",
                        "messages": [{"role": "user", "content": "Stream test"}],
                        "stream": True,
                    },
                )

                assert response.status_code == 200
                # Verify it's a streaming response
                content_type = response.headers.get("content-type", "")
                assert "text/event-stream" in content_type

    async def test_api_authentication(self):
        """Test API key authentication."""
        test_api_key = "test-secret-key"

        with patch("claude_wrapper.api.server.config") as mock_config:
            mock_config.api_key = test_api_key

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                # Test without API key
                response = await client.get("/v1/models")
                assert response.status_code == 401

                # Test with wrong API key
                response = await client.get(
                    "/v1/models", headers={"Authorization": "Bearer wrong-key"}
                )
                assert response.status_code == 401

                # Test with correct API key
                response = await client.get(
                    "/v1/models", headers={"Authorization": f"Bearer {test_api_key}"}
                )
                assert response.status_code == 200

    async def test_error_handling(self):
        """Test error handling in API endpoints."""
        from claude_wrapper.core.exceptions import ClaudeWrapperError

        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            mock_client.chat = AsyncMock(side_effect=ClaudeWrapperError("Test error"))

            async with AsyncClient(
                transport=ASGITransport(app=app), base_url="http://test"
            ) as client:
                response = await client.post(
                    "/v1/chat/completions",
                    json={
                        "model": "sonnet",
                        "messages": [{"role": "user", "content": "Error test"}],
                    },
                )

                assert response.status_code == 500
                data = response.json()
                assert "Test error" in data["detail"]
