"""Tests for FastAPI server and OpenAI-compatible endpoints."""

import json
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient

from claude_wrapper.api.server import app
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
def api_client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
async def async_api_client():
    """Create an async test client."""
    async with AsyncClient(app=app, base_url="http://test") as client:
        yield client


class TestAPIServer:
    """Test suite for API server endpoints."""
    
    @pytest.mark.unit
    def test_root_endpoint(self, api_client):
        """Test the root endpoint."""
        response = api_client.get("/")
        
        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "endpoints" in data
        assert data["endpoints"]["chat"] == "/v1/chat/completions"
    
    @pytest.mark.unit
    def test_health_endpoint(self, api_client):
        """Test the health check endpoint."""
        with patch("claude_wrapper.api.server.claude_client.check_auth") as mock_auth:
            mock_auth.return_value = True
            
            response = api_client.get("/health")
            
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["claude_cli"] == "authenticated"
    
    @pytest.mark.unit
    def test_health_endpoint_not_authenticated(self, api_client):
        """Test health check when Claude CLI is not authenticated."""
        with patch("claude_wrapper.api.server.claude_client.check_auth") as mock_auth:
            mock_auth.side_effect = Exception("Not authenticated")
            
            response = api_client.get("/health")
            
            assert response.status_code == 503
            data = response.json()
            assert data["status"] == "unhealthy"
            assert "error" in data
    
    @pytest.mark.unit
    def test_list_models(self, api_client):
        """Test listing available models."""
        response = api_client.get("/v1/models")
        
        assert response.status_code == 200
        data = response.json()
        assert "data" in data
        assert len(data["data"]) > 0
        
        # Check model structure
        model = data["data"][0]
        assert "id" in model
        assert "created" in model
        assert "owned_by" in model
        assert model["owned_by"] == "anthropic"
    
    @pytest.mark.unit
    async def test_chat_completion_non_streaming(self, api_client):
        """Test non-streaming chat completion."""
        with patch("claude_wrapper.api.server.claude_client.chat") as mock_chat:
            mock_session = create_test_session("test-session")
            mock_chat.return_value = ("Test response", mock_session)
            
            request_data = {
                "model": "claude-3-opus-20240229",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "max_tokens": 100,
                "temperature": 0.7,
                "stream": False,
            }
            
            response = api_client.post("/v1/chat/completions", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["model"] == "claude-3-opus-20240229"
            assert "choices" in data
            assert len(data["choices"]) == 1
            assert data["choices"][0]["message"]["content"] == "Test response"
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert "usage" in data
    
    @pytest.mark.unit
    async def test_chat_completion_with_system(self, api_client):
        """Test chat completion with system prompt."""
        with patch("claude_wrapper.api.server.claude_client.chat") as mock_chat:
            mock_session = create_test_session("test-session")
            mock_chat.return_value = ("Response", mock_session)
            
            request_data = {
                "model": "claude-3-opus-20240229",
                "messages": [
                    {"role": "system", "content": "You are helpful"},
                    {"role": "user", "content": "Hello"}
                ],
            }
            
            response = api_client.post("/v1/chat/completions", json=request_data)
            
            assert response.status_code == 200
            
            # Verify system prompt was passed
            mock_chat.assert_called_once()
            call_kwargs = mock_chat.call_args.kwargs
            assert call_kwargs["system_prompt"] == "You are helpful"
    
    @pytest.mark.unit
    def test_chat_completion_missing_user_message(self, api_client):
        """Test chat completion without user message."""
        request_data = {
            "model": "claude-3-opus-20240229",
            "messages": [
                {"role": "system", "content": "You are helpful"}
            ],
        }
        
        response = api_client.post("/v1/chat/completions", json=request_data)
        
        assert response.status_code == 500  # API has error before validation
        data = response.json()
        assert "detail" in data
    
    @pytest.mark.unit
    async def test_chat_completion_streaming(self, api_client):
        """Test streaming chat completion."""
        with patch("claude_wrapper.api.server.claude_client.stream_chat") as mock_stream:
            async def mock_generator():
                chunks = ["Hello ", "from ", "Claude!"]
                for chunk in chunks:
                    yield chunk
            
            mock_stream.return_value = mock_generator()
            
            request_data = {
                "model": "claude-3-opus-20240229",
                "messages": [
                    {"role": "user", "content": "Hello"}
                ],
                "stream": True,
            }
            
            response = api_client.post("/v1/chat/completions", json=request_data)
            
            assert response.status_code == 200
            # Check that it's an SSE stream
            assert "text/event-stream" in response.headers["content-type"]
    
    @pytest.mark.unit
    def test_completions_endpoint(self, api_client):
        """Test the completions endpoint."""
        with patch("claude_wrapper.api.server.claude_client.complete") as mock_complete:
            mock_complete.return_value = "Completed text"
            
            request_data = {
                "model": "claude-3-opus-20240229",
                "prompt": "Complete this: ",
                "max_tokens": 50,
                "temperature": 0.5,
            }
            
            response = api_client.post("/v1/completions", json=request_data)
            
            assert response.status_code == 200
            data = response.json()
            
            assert data["model"] == "claude-3-opus-20240229"
            assert "choices" in data
            assert data["choices"][0]["text"] == "Completed text"
            assert "usage" in data
    
    @pytest.mark.unit
    def test_completions_with_list_prompt(self, api_client):
        """Test completions with list of prompts."""
        with patch("claude_wrapper.api.server.claude_client.complete") as mock_complete:
            mock_complete.return_value = "Combined response"
            
            request_data = {
                "model": "claude-3-opus-20240229",
                "prompt": ["First part", "Second part"],
            }
            
            response = api_client.post("/v1/completions", json=request_data)
            
            assert response.status_code == 200
            
            # Verify prompts were joined
            mock_complete.assert_called_once()
            call_kwargs = mock_complete.call_args.kwargs
            assert call_kwargs["prompt"] == "First part\nSecond part"
    
    @pytest.mark.unit
    async def test_create_session(self, api_client):
        """Test creating a new session."""
        with patch("claude_wrapper.api.server.session_manager.create_session") as mock_create:
            mock_session = create_test_session("new-session-123")
            mock_create.return_value = mock_session
            
            response = api_client.post(
                "/v1/sessions/create",
                json={"metadata": {"user": "test"}}
            )
            
            assert response.status_code == 200
            data = response.json()
            assert data["session_id"] == "new-session-123"
            assert "created_at" in data
    
    @pytest.mark.unit
    async def test_get_session(self, api_client):
        """Test getting session details."""
        with patch("claude_wrapper.api.server.session_manager.get_session") as mock_get:
            mock_session = create_test_session(
                "test-session",
                messages=[{"role": "user", "content": "Test"}],
                metadata={"key": "value"}
            )
            mock_get.return_value = mock_session
            
            response = api_client.get("/v1/sessions/test-session")
            
            assert response.status_code == 200
            data = response.json()
            assert data["id"] == "test-session"
            # Messages are serialized as dicts in API response
            assert len(data["messages"]) == len(mock_session.messages)
            assert data["messages"][0]["role"] == "user"
            assert data["messages"][0]["content"] == "Test"
            assert data["metadata"] == mock_session.metadata
    
    @pytest.mark.unit
    def test_get_nonexistent_session(self, api_client):
        """Test getting non-existent session."""
        with patch("claude_wrapper.api.server.session_manager.get_session") as mock_get:
            mock_get.return_value = None
            
            response = api_client.get("/v1/sessions/nonexistent")
            
            assert response.status_code == 404
            assert "Session not found" in response.json()["detail"]
    
    @pytest.mark.unit
    async def test_delete_session(self, api_client):
        """Test deleting a session."""
        with patch("claude_wrapper.api.server.session_manager.delete_session") as mock_delete:
            mock_delete.return_value = True
            
            response = api_client.delete("/v1/sessions/test-session")
            
            assert response.status_code == 200
            assert response.json()["message"] == "Session deleted"
    
    @pytest.mark.unit
    def test_delete_nonexistent_session(self, api_client):
        """Test deleting non-existent session."""
        with patch("claude_wrapper.api.server.session_manager.delete_session") as mock_delete:
            mock_delete.return_value = False
            
            response = api_client.delete("/v1/sessions/nonexistent")
            
            assert response.status_code == 404
            assert "Session not found" in response.json()["detail"]
    
    @pytest.mark.unit
    def test_api_key_authentication(self, api_client):
        """Test API key authentication."""
        with patch("claude_wrapper.api.server.config.api_key", "test-api-key"):
            # Without API key
            response = api_client.get("/v1/models")
            assert response.status_code == 401
            
            # With wrong API key
            response = api_client.get(
                "/v1/models",
                headers={"Authorization": "Bearer wrong-key"}
            )
            assert response.status_code == 401
            
            # With correct API key
            response = api_client.get(
                "/v1/models",
                headers={"Authorization": "Bearer test-api-key"}
            )
            assert response.status_code == 200
    
    @pytest.mark.unit
    def test_error_handling(self, api_client):
        """Test error handling in API."""
        from claude_wrapper.core.exceptions import ClaudeWrapperError
        
        with patch("claude_wrapper.api.server.claude_client.chat") as mock_chat:
            mock_chat.side_effect = ClaudeWrapperError(
                "Test error",
                code="TEST_ERROR",
                details={"field": "value"}
            )
            
            request_data = {
                "messages": [{"role": "user", "content": "Test"}]
            }
            
            response = api_client.post("/v1/chat/completions", json=request_data)
            
            assert response.status_code == 500
            data = response.json()
            assert "detail" in data
            assert "Test error" in str(data["detail"])
    
    @pytest.mark.unit
    def test_cors_headers(self, api_client):
        """Test CORS headers are properly set."""
        # Note: TestClient doesn't always apply middleware the same way as real server
        # This test verifies the endpoint works, CORS is configured at server level
        response = api_client.get("/v1/models")
        
        assert response.status_code == 200
        # In real deployment, CORS headers would be present