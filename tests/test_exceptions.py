"""Tests for custom exceptions."""

import pytest

from claude_wrapper.core.exceptions import (
    AuthenticationError,
    ClaudeWrapperError,
    ConfigurationError,
    SessionNotFoundError,
    TimeoutError,
)


class TestExceptions:
    """Test suite for custom exceptions."""

    @pytest.mark.unit
    def test_base_exception(self):
        """Test ClaudeWrapperError base exception."""
        error = ClaudeWrapperError("Test error", code="TEST_ERROR")

        assert str(error) == "Test error"
        assert error.message == "Test error"
        assert error.code == "TEST_ERROR"
        assert error.details is None

        # With details
        error = ClaudeWrapperError("Test error", code="TEST", details={"key": "value"})
        assert error.details == {"key": "value"}

    @pytest.mark.unit
    def test_authentication_error(self):
        """Test AuthenticationError."""
        error = AuthenticationError("Not authenticated")

        assert isinstance(error, ClaudeWrapperError)
        assert str(error) == "Not authenticated"
        assert error.code == "AUTH_ERROR"

    @pytest.mark.unit
    def test_timeout_error(self):
        """Test TimeoutError."""
        error = TimeoutError("Request timed out after 30s")

        assert isinstance(error, ClaudeWrapperError)
        assert str(error) == "Request timed out after 30s"
        assert error.code == "TIMEOUT"

    @pytest.mark.unit
    def test_session_not_found_error(self):
        """Test SessionNotFoundError."""
        session_id = "test-123"
        error = SessionNotFoundError(session_id)

        assert isinstance(error, ClaudeWrapperError)
        assert str(error) == f"Session not found: {session_id}"
        assert error.code == "SESSION_NOT_FOUND"
        assert error.details == {"session_id": session_id}

    @pytest.mark.unit
    def test_configuration_error(self):
        """Test ConfigurationError."""
        error = ConfigurationError("Invalid configuration: missing API key")

        assert isinstance(error, ClaudeWrapperError)
        assert str(error) == "Invalid configuration: missing API key"
        assert error.code == "CONFIG_ERROR"

    @pytest.mark.unit
    def test_exception_inheritance(self):
        """Test that all custom exceptions inherit from base."""
        exceptions = [
            AuthenticationError("test"),
            TimeoutError("test"),
            SessionNotFoundError("test"),
            ConfigurationError("test"),
        ]

        for exc in exceptions:
            assert isinstance(exc, ClaudeWrapperError)
            assert isinstance(exc, Exception)

    @pytest.mark.unit
    def test_exception_with_cause(self):
        """Test exception chaining."""
        original = ValueError("Original error")

        try:
            raise ClaudeWrapperError("Wrapped error") from original
        except ClaudeWrapperError as e:
            assert e.__cause__ == original
            assert str(e) == "Wrapped error"

    @pytest.mark.unit
    def test_exception_serialization(self):
        """Test exception serialization for API responses."""
        error = ClaudeWrapperError(
            "Test error", code="TEST_CODE", details={"field": "value", "number": 42}
        )

        # Should be serializable
        import json

        error_dict = {
            "message": error.message,
            "code": error.code,
            "details": error.details,
        }
        serialized = json.dumps(error_dict)
        deserialized = json.loads(serialized)

        assert deserialized["message"] == "Test error"
        assert deserialized["code"] == "TEST_CODE"
        assert deserialized["details"] == {"field": "value", "number": 42}
