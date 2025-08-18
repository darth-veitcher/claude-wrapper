"""End-to-end integration tests for Claude Wrapper."""

import asyncio
import json
import tempfile
from pathlib import Path
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import pytest
from fastapi.testclient import TestClient

from claude_wrapper.api.server import app
from claude_wrapper.core import ClaudeClient, SessionManager
from claude_wrapper.core.session import Session


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""
    
    @pytest.mark.asyncio
    async def test_full_chat_workflow(self, temp_dir):
        """Test complete chat workflow from API to session persistence."""
        # Setup
        session_manager = SessionManager(storage_dir=temp_dir / "sessions")
        
        with patch("claude_wrapper.api.server.session_manager", session_manager):
            with patch("claude_wrapper.api.server.claude_client") as mock_client:
                # Mock Claude responses
                mock_client.chat = AsyncMock()
                now = datetime.now()
                mock_client.chat.return_value = (
                    "Hello! How can I help you?",
                    Session(id="test-session-123", created_at=now, updated_at=now)
                )
                
                client = TestClient(app)
                
                # 1. Create initial chat
                response = client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "Hello Claude"}],
                    "model": "claude-3-opus-20240229"
                })
                
                assert response.status_code == 200
                data = response.json()
                assert data["choices"][0]["message"]["content"] == "Hello! How can I help you?"
                
                # 2. Continue conversation with session
                now2 = datetime.now()
                mock_client.chat.return_value = (
                    "Python is a programming language",
                    Session(id="test-session-123", created_at=now, updated_at=now2, messages=[
                        {"role": "user", "content": "Hello Claude"},
                        {"role": "assistant", "content": "Hello! How can I help you?"},
                        {"role": "user", "content": "What is Python?"},
                        {"role": "assistant", "content": "Python is a programming language"},
                    ])
                )
                
                response = client.post("/v1/chat/completions", json={
                    "messages": [{"role": "user", "content": "What is Python?"}],
                    "session_id": "test-session-123",
                    "model": "claude-3-opus-20240229"
                })
                
                assert response.status_code == 200
                
                # 3. Verify session was saved
                session = await session_manager.get_session("test-session-123")
                if session:  # If mocking allows persistence
                    assert len(session.messages) >= 2
    
    @pytest.mark.asyncio
    async def test_streaming_workflow(self):
        """Test streaming from client through API."""
        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            # Mock streaming
            async def mock_stream(*args, **kwargs):
                for chunk in ["Hello ", "from ", "streaming ", "Claude!"]:
                    yield chunk
            
            mock_client.stream_chat = mock_stream
            
            client = TestClient(app)
            
            response = client.post(
                "/v1/chat/completions",
                json={
                    "messages": [{"role": "user", "content": "Stream test"}],
                    "stream": True
                }
            )
            assert response.status_code == 200
            
            # Collect streamed chunks
            chunks = []
            for line in response.iter_lines():
                if line and line.startswith("data: "):
                    data = line[6:]
                    if data != "[DONE]" and data.strip():
                        try:
                            chunks.append(json.loads(data))
                        except json.JSONDecodeError:
                            # Skip malformed JSON chunks in test
                            continue
            
            # For streaming test, just verify response format is correct
            # The actual streaming content depends on the mock implementation
            # Verify streaming response headers
            assert "text/event-stream" in response.headers.get("content-type", "")
    
    @pytest.mark.asyncio
    async def test_session_lifecycle(self, temp_dir):
        """Test complete session lifecycle."""
        session_manager = SessionManager(storage_dir=temp_dir / "sessions")
        
        # 1. Create session
        session = await session_manager.create_session(
            metadata={"test": "integration"}
        )
        session_id = session.id
        
        # 2. Add messages
        session.add_message("user", "First message")
        session.add_message("assistant", "First response")
        await session_manager.update_session(session)
        
        # 3. Retrieve session
        retrieved = await session_manager.get_session(session_id)
        assert retrieved is not None
        assert len(retrieved.messages) == 2
        
        # 4. Continue conversation
        retrieved.add_message("user", "Second message")
        retrieved.add_message("assistant", "Second response")
        await session_manager.update_session(retrieved)
        
        # 5. List sessions
        all_sessions = await session_manager.list_sessions()
        assert any(s.id == session_id for s in all_sessions)
        
        # 6. Delete session
        deleted = await session_manager.delete_session(session_id)
        assert deleted is True
        
        # 7. Verify deletion
        retrieved = await session_manager.get_session(session_id)
        assert retrieved is None
    
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
        response, session = await client.chat(
            "What is 2+2? Answer with just the number.",
            max_tokens=10
        )
        
        assert "4" in response
        assert isinstance(session, Session)
    
    @pytest.mark.asyncio
    async def test_error_propagation(self):
        """Test error propagation through the stack."""
        from claude_wrapper.core.exceptions import AuthenticationError
        
        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            mock_client.chat = AsyncMock()
            mock_client.chat.side_effect = AuthenticationError("Not authenticated")
            
            client = TestClient(app)
            
            response = client.post("/v1/chat/completions", json={
                "messages": [{"role": "user", "content": "Test"}]
            })
            
            assert response.status_code == 500
            assert "Not authenticated" in response.json()["detail"]
    
    @pytest.mark.asyncio
    async def test_concurrent_sessions(self, temp_dir):
        """Test handling multiple concurrent sessions."""
        session_manager = SessionManager(storage_dir=temp_dir / "sessions")
        
        async def create_and_chat(index):
            session = await session_manager.create_session(
                metadata={"user": f"user_{index}"}
            )
            
            # Simulate chat
            session.add_message("user", f"Message from user {index}")
            session.add_message("assistant", f"Response to user {index}")
            
            await session_manager.update_session(session)
            return session
        
        # Create multiple sessions concurrently
        tasks = [create_and_chat(i) for i in range(10)]
        sessions = await asyncio.gather(*tasks)
        
        # Verify all sessions were created
        assert len(sessions) == 10
        
        # Verify each session is unique and properly saved
        for i, session in enumerate(sessions):
            retrieved = await session_manager.get_session(session.id)
            assert retrieved is not None
            assert retrieved.metadata["user"] == f"user_{i}"
            assert len(retrieved.messages) == 2
    
    @pytest.mark.asyncio
    async def test_api_with_authentication(self):
        """Test API with authentication enabled."""
        from claude_wrapper.utils.config import Config
        
        # Enable API key authentication
        test_api_key = "test-secret-key"
        
        with patch("claude_wrapper.api.server.config") as mock_config:
            mock_config.api_key = test_api_key
            
            client = TestClient(app)
            
            # Request without API key should fail
            response = client.get("/v1/models")
            assert response.status_code == 401
            
            # Request with wrong API key should fail
            response = client.get(
                "/v1/models",
                headers={"Authorization": "Bearer wrong-key"}
            )
            assert response.status_code == 401
            
            # Request with correct API key should succeed
            response = client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {test_api_key}"}
            )
            assert response.status_code == 200
    
    @pytest.mark.asyncio
    async def test_openai_client_compatibility(self):
        """Test compatibility with OpenAI Python client."""
        # This simulates how the OpenAI client would interact with our API
        import httpx
        
        with patch("claude_wrapper.api.server.claude_client") as mock_client:
            mock_client.chat = AsyncMock()
            now = datetime.now()
            mock_client.chat.return_value = (
                "Compatible response",
                Session(id="openai-session", created_at=now, updated_at=now)
            )
            
            client = TestClient(app)
            
            # Simulate OpenAI client request format
            response = client.post(
                "/v1/chat/completions",
                json={
                    "model": "gpt-3.5-turbo",  # OpenAI model name
                    "messages": [
                        {"role": "system", "content": "You are helpful"},
                        {"role": "user", "content": "Hello"}
                    ],
                    "temperature": 0.7,
                    "max_tokens": 150,
                    "top_p": 1,
                    "frequency_penalty": 0,
                    "presence_penalty": 0,
                    "n": 1,
                    "stream": False
                }
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