"""Tests for custom exceptions."""

import pytest

from claude_wrapper.core.exceptions import (
    ClaudeAuthError,
    ClaudeExecutionError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
    ClaudeWrapperError,
)


class TestExceptions:
    """Test suite for custom exceptions."""

    @pytest.mark.unit
    def test_base_exception(self):
        """Test ClaudeWrapperError base exception."""
        error = ClaudeWrapperError("Test error")

        assert str(error) == "Test error"
        assert error.message == "Test error"

        # Test default message
        error = ClaudeWrapperError()
        assert error.message == "Claude wrapper error occurred"

    @pytest.mark.unit
    def test_claude_not_found_error(self):
        """Test ClaudeNotFoundError."""
        error = ClaudeNotFoundError("/custom/path")

        assert isinstance(error, ClaudeWrapperError)
        assert "Claude CLI not found at '/custom/path'" in str(error)
        assert "Please install claude-cli" in str(error)

        # Test default path
        error = ClaudeNotFoundError()
        assert "Claude CLI not found at 'claude'" in str(error)

    @pytest.mark.unit
    def test_claude_auth_error(self):
        """Test ClaudeAuthError."""
        error = ClaudeAuthError("Custom auth message")

        assert isinstance(error, ClaudeWrapperError)
        assert "Custom auth message" in str(error)
        assert "Please run 'claude auth' first" in str(error)

        # Test default message
        error = ClaudeAuthError()
        assert "Claude is not authenticated" in str(error)

    @pytest.mark.unit
    def test_claude_timeout_error(self):
        """Test ClaudeTimeoutError."""
        error = ClaudeTimeoutError("Request timed out after 30s")

        assert isinstance(error, ClaudeWrapperError)
        assert str(error) == "Request timed out after 30s"

        # Test default message
        error = ClaudeTimeoutError()
        assert str(error) == "Command timed out"

    @pytest.mark.unit
    def test_claude_execution_error(self):
        """Test ClaudeExecutionError."""
        error = ClaudeExecutionError("Command failed with exit code 1")

        assert isinstance(error, ClaudeWrapperError)
        assert str(error) == "Command failed with exit code 1"

        # Test default message
        error = ClaudeExecutionError()
        assert str(error) == "Command execution failed"

    @pytest.mark.unit
    def test_exception_inheritance(self):
        """Test that all custom exceptions inherit from base."""
        exceptions = [
            ClaudeAuthError("test"),
            ClaudeTimeoutError("test"),
            ClaudeNotFoundError("test"),
            ClaudeExecutionError("test"),
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
    def test_exception_message_attribute(self):
        """Test that exceptions store the message attribute."""
        test_cases = [
            (ClaudeWrapperError("test message"), "test message"),
            (ClaudeAuthError("auth issue"), "auth issue. Please run 'claude auth' first."),
            (ClaudeTimeoutError("timeout occurred"), "timeout occurred"),
            (ClaudeExecutionError("exec failed"), "exec failed"),
            (
                ClaudeNotFoundError("custom"),
                "Claude CLI not found at 'custom'. Please install claude-cli.",
            ),
        ]

        for exception, expected_message in test_cases:
            assert exception.message == expected_message
            assert str(exception) == expected_message
