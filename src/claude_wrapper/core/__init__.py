"""Core Claude wrapper functionality."""

from .client import ClaudeClient
from .exceptions import (
    ClaudeAuthError,
    ClaudeExecutionError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
    ClaudeWrapperError,
)
from .http_client import ClaudeHTTPClient
from .unified_client import (
    ClientMode,
    RecursionError,
    UnifiedClaudeClient,
    create_api_client,
    create_claude_client,
    create_claude_wrapper_client,
)

__all__ = [
    # Legacy client
    "ClaudeClient",
    # HTTP client for API mode
    "ClaudeHTTPClient",
    # Unified client with recursion support
    "UnifiedClaudeClient",
    "ClientMode",
    "RecursionError",
    # Client factory functions
    "create_claude_client",
    "create_claude_wrapper_client",
    "create_api_client",
    # Exceptions
    "ClaudeWrapperError",
    "ClaudeNotFoundError",
    "ClaudeAuthError",
    "ClaudeTimeoutError",
    "ClaudeExecutionError",
]
