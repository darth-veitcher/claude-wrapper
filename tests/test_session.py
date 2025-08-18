"""Tests for session management functionality."""

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
from freezegun import freeze_time

from claude_wrapper.core.exceptions import SessionNotFoundError
from claude_wrapper.core.session import Session, SessionManager, Message


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


class TestSession:
    """Test suite for Session class."""
    
    @pytest.mark.unit
    def test_session_creation_with_defaults(self):
        """Test creating a session with default values."""
        session = create_test_session()
        
        assert session.id is not None
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
        assert session.messages == []
        assert session.metadata == {}
    
    @pytest.mark.unit
    def test_session_creation_with_values(self):
        """Test creating a session with custom values."""
        now = datetime.now()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi"},
        ]
        metadata = {"user": "test", "project": "test-project"}
        
        session = Session(
            id="custom-id",
            created_at=now,
            updated_at=now,
            messages=messages,
            metadata=metadata,
        )
        
        assert session.id == "custom-id"
        assert session.created_at == now
        assert session.updated_at == now
        assert session.messages == messages
        assert session.metadata == metadata
    
    @pytest.mark.unit
    def test_session_to_dict(self, sample_session):
        """Test converting session to dictionary."""
        session_dict = sample_session.to_dict()
        
        assert session_dict["id"] == sample_session.id
        # Messages are serialized as dicts in to_dict() output
        assert len(session_dict["messages"]) == len(sample_session.messages)
        assert session_dict["messages"][0]["role"] == "user"
        assert session_dict["messages"][0]["content"] == "Hello"
        assert session_dict["metadata"] == sample_session.metadata
        assert "created_at" in session_dict
        assert "updated_at" in session_dict
    
    @pytest.mark.unit
    def test_session_from_dict(self):
        """Test creating session from dictionary."""
        data = {
            "id": "test-123",
            "created_at": "2024-01-01T12:00:00",
            "updated_at": "2024-01-01T13:00:00",
            "messages": [{"role": "user", "content": "Test"}],
            "metadata": {"key": "value"},
        }
        
        session = Session.from_dict(data)
        
        assert session.id == "test-123"
        assert isinstance(session.created_at, datetime)
        assert isinstance(session.updated_at, datetime)
        assert session.messages == data["messages"]
        assert session.metadata == data["metadata"]
    
    @pytest.mark.unit
    def test_session_add_message(self, sample_session):
        """Test adding messages to a session."""
        initial_count = len(sample_session.messages)
        initial_updated = sample_session.updated_at
        
        sample_session.add_message("user", "New question")
        
        assert len(sample_session.messages) == initial_count + 1
        assert sample_session.messages[-1]["role"] == "user"
        assert sample_session.messages[-1]["content"] == "New question"
        assert sample_session.updated_at > initial_updated
    
    @pytest.mark.unit
    def test_session_get_context(self, sample_session):
        """Test getting session context."""
        context = sample_session.get_context(max_messages=1)
        
        assert len(context) == 1
        assert context[0] == sample_session.messages[-1]
        
        # Test with more messages than available
        context = sample_session.get_context(max_messages=100)
        assert len(context) == len(sample_session.messages)
    
    @pytest.mark.unit
    def test_session_token_count(self, sample_session):
        """Test approximate token counting."""
        token_count = sample_session.get_token_count()
        
        # Should be a positive number
        assert token_count > 0
        
        # Add a message and verify count increases
        initial_count = token_count
        sample_session.add_message("user", "This is a longer message with more tokens")
        new_count = sample_session.get_token_count()
        
        assert new_count > initial_count


class TestSessionManager:
    """Test suite for SessionManager."""
    
    @pytest.mark.unit
    async def test_manager_initialization(self, temp_dir):
        """Test SessionManager initialization."""
        manager = SessionManager(session_dir=str(temp_dir / "sessions"))
        
        assert manager.session_dir == temp_dir / "sessions"
        assert manager.session_dir.exists()
    
    @pytest.mark.unit
    async def test_create_session(self, session_manager):
        """Test creating a new session."""
        metadata = {"test": "value"}
        session = await session_manager.create_session(metadata=metadata)
        
        assert isinstance(session, Session)
        assert session.metadata == metadata
        
        # Verify session file was created
        session_file = session_manager.session_dir / f"{session.id}.json"
        assert session_file.exists()
    
    @pytest.mark.unit
    async def test_get_existing_session(self, session_manager):
        """Test retrieving an existing session."""
        # Create a session
        created_session = await session_manager.create_session()
        
        # Retrieve it
        retrieved_session = await session_manager.get_session(created_session.id)
        
        assert retrieved_session is not None
        assert retrieved_session.id == created_session.id
        assert retrieved_session.created_at == created_session.created_at
    
    @pytest.mark.unit
    async def test_get_nonexistent_session(self, session_manager):
        """Test retrieving a non-existent session."""
        session = await session_manager.get_session("nonexistent-id")
        assert session is None
    
    @pytest.mark.unit
    async def test_save_session(self, session_manager, sample_session):
        """Test saving a session."""
        await session_manager.save_session(sample_session)
        
        # Verify file exists and contains correct data
        session_file = session_manager.session_dir / f"{sample_session.id}.json"
        assert session_file.exists()
        
        with open(session_file) as f:
            data = json.load(f)
        
        assert data["id"] == sample_session.id
        assert data["messages"] == sample_session.messages
    
    @pytest.mark.unit
    async def test_delete_session(self, session_manager):
        """Test deleting a session."""
        # Create a session
        session = await session_manager.create_session()
        session_file = session_manager.session_dir / f"{session.id}.json"
        assert session_file.exists()
        
        # Delete it
        result = await session_manager.delete_session(session.id)
        assert result is True
        assert not session_file.exists()
        
        # Try to delete non-existent session
        result = await session_manager.delete_session("nonexistent")
        assert result is False
    
    @pytest.mark.unit
    async def test_list_sessions(self, session_manager):
        """Test listing all sessions."""
        # Create multiple sessions
        sessions = []
        for i in range(3):
            session = await session_manager.create_session(
                metadata={"index": i}
            )
            sessions.append(session)
        
        # List all sessions
        all_sessions = await session_manager.list_sessions()
        
        assert len(all_sessions) >= 3
        session_ids = [s.id for s in all_sessions]
        for session in sessions:
            assert session.id in session_ids
    
    @pytest.mark.unit
    async def test_cleanup_old_sessions(self, session_manager):
        """Test cleaning up old sessions."""
        # Create sessions with different ages
        with freeze_time("2024-01-01"):
            old_session = await session_manager.create_session()
        
        with freeze_time("2024-01-10"):
            recent_session = await session_manager.create_session()
        
        # Cleanup sessions older than 5 days
        with freeze_time("2024-01-11"):
            deleted_count = await session_manager.cleanup_old_sessions(days=5)
        
        assert deleted_count == 1
        
        # Verify old session is gone, recent one remains
        assert await session_manager.get_session(old_session.id) is None
        assert await session_manager.get_session(recent_session.id) is not None
    
    @pytest.mark.unit
    async def test_get_or_create_session(self, session_manager):
        """Test get_or_create_session functionality."""
        # First call creates new session
        session1 = await session_manager.get_or_create_session(
            session_id=None,
            metadata={"first": True}
        )
        assert session1 is not None
        assert session1.metadata == {"first": True}
        
        # Second call with same ID returns existing
        session2 = await session_manager.get_or_create_session(
            session_id=session1.id
        )
        assert session2.id == session1.id
        
        # Call with non-existent ID creates new
        session3 = await session_manager.get_or_create_session(
            session_id="nonexistent",
            metadata={"new": True}
        )
        assert session3.id != session1.id
        assert session3.metadata == {"new": True}
    
    @pytest.mark.unit
    async def test_session_file_corruption_handling(self, session_manager):
        """Test handling of corrupted session files."""
        # Create a corrupted session file
        session_id = str(uuid.uuid4())
        session_file = session_manager.session_dir / f"{session_id}.json"
        session_file.write_text("invalid json {{{")
        
        # Should return None for corrupted file
        session = await session_manager.get_session(session_id)
        assert session is None
    
    @pytest.mark.unit
    async def test_concurrent_session_access(self, session_manager):
        """Test concurrent access to sessions."""
        import asyncio
        
        async def create_and_update_session(index):
            session = await session_manager.create_session(
                metadata={"index": index}
            )
            # Add some messages
            session.add_message("user", f"Message {index}")
            await session_manager.save_session(session)
            return session
        
        # Create and update sessions concurrently
        tasks = [create_and_update_session(i) for i in range(5)]
        sessions = await asyncio.gather(*tasks)
        
        # Verify all sessions were created and saved correctly
        assert len(sessions) == 5
        for i, session in enumerate(sessions):
            retrieved = await session_manager.get_session(session.id)
            assert retrieved is not None
            assert len(retrieved.messages) == 1
    
    @pytest.mark.unit
    async def test_session_metadata_search(self, session_manager):
        """Test searching sessions by metadata."""
        # Create sessions with different metadata
        await session_manager.create_session(metadata={"project": "A", "user": "alice"})
        await session_manager.create_session(metadata={"project": "B", "user": "bob"})
        await session_manager.create_session(metadata={"project": "A", "user": "bob"})
        
        # Search by metadata (would need to implement this method)
        sessions = await session_manager.list_sessions()
        project_a_sessions = [
            s for s in sessions 
            if s.metadata.get("project") == "A"
        ]
        
        assert len(project_a_sessions) == 2
    
    @pytest.mark.unit
    @pytest.mark.parametrize("max_messages", [0, 1, 5, 10, 100])
    async def test_session_context_limits(self, session_manager, max_messages):
        """Test session context with different message limits."""
        session = await session_manager.create_session()
        
        # Add many messages
        for i in range(20):
            session.add_message("user" if i % 2 == 0 else "assistant", f"Message {i}")
        
        context = session.get_context(max_messages=max_messages)
        
        if max_messages == 0:
            assert len(context) == 0
        elif max_messages >= 20:
            assert len(context) == 20
        else:
            assert len(context) == max_messages
            # Should get the most recent messages
            assert context[-1]["content"] == "Message 19"