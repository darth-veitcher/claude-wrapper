"""Claude Wrapper - A simple Python wrapper for Claude CLI.

This package provides a convenient interface to the Claude CLI tool,
enabling programmatic access to Claude's capabilities with streaming support
and OpenAI-compatible API interface.

The version is dynamically determined from git tags during development
and from package metadata when installed.
"""

from claude_wrapper._version import __version__, get_version, get_version_info

__all__ = ["__version__", "get_version", "get_version_info"]

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
