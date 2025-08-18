"""Claude Wrapper - A simple Python wrapper for Claude CLI."""

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
