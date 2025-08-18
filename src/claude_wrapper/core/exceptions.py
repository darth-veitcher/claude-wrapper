"""Exception classes for Claude wrapper."""


class ClaudeWrapperError(Exception):
    """Base exception for all Claude wrapper errors."""

    def __init__(self, message: str = "Claude wrapper error occurred"):
        self.message = message
        super().__init__(self.message)


class ClaudeNotFoundError(ClaudeWrapperError):
    """Raised when Claude CLI is not found."""

    def __init__(self, path: str = "claude"):
        super().__init__(f"Claude CLI not found at '{path}'. Please install claude-cli.")


class ClaudeAuthError(ClaudeWrapperError):
    """Raised when Claude is not authenticated."""

    def __init__(self, message: str = "Claude is not authenticated"):
        super().__init__(f"{message}. Please run 'claude auth' first.")


class ClaudeTimeoutError(ClaudeWrapperError):
    """Raised when a Claude command times out."""

    def __init__(self, message: str = "Command timed out"):
        super().__init__(message)


class ClaudeExecutionError(ClaudeWrapperError):
    """Raised when Claude command execution fails."""

    def __init__(self, message: str = "Command execution failed"):
        super().__init__(message)
