"""Claude Wrapper - A simple Python wrapper for Claude CLI.

This package provides a convenient interface to the Claude CLI tool,
enabling programmatic access to Claude's capabilities.
"""

__version__ = "0.1.0"
__author__ = "Claude Wrapper Contributors"

from claude_wrapper.core.client import ClaudeClient
from claude_wrapper.core.exceptions import (
    ClaudeAuthError,
    ClaudeExecutionError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
    ClaudeWrapperError,
)

__all__ = [
    "ClaudeClient",
    "ClaudeWrapperError",
    "ClaudeNotFoundError",
    "ClaudeAuthError",
    "ClaudeTimeoutError",
    "ClaudeExecutionError",
]
# Claude Wrapper - Test publishing pipeline
