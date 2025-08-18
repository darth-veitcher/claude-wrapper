"""Unified client supporting CLI (Claude/Claude-wrapper) and API modes with recursion safeguards."""

import asyncio
import contextvars
import shutil
import threading
from collections.abc import AsyncIterator
from enum import Enum
from typing import Any

from .exceptions import (
    ClaudeAuthError,
    ClaudeExecutionError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
    ClaudeWrapperError,
)
from .http_client import ClaudeHTTPClient


class ClientMode(Enum):
    """Client operation modes."""

    CLAUDE_CLI = "claude-cli"  # Original Claude CLI
    CLAUDE_WRAPPER_CLI = "claude-wrapper-cli"  # Claude-wrapper CLI
    API = "api"  # Claude-wrapper HTTP API


class RecursionError(ClaudeWrapperError):
    """Raised when recursion limits are exceeded."""

    pass


# Context variables for tracking recursion depth
_recursion_depth: contextvars.ContextVar[int] = contextvars.ContextVar("recursion_depth", default=0)
_call_stack: contextvars.ContextVar[list[str]] = contextvars.ContextVar("call_stack")

# Thread-local storage for subprocess-based recursion tracking
_thread_local = threading.local()


class UnifiedClaudeClient:
    """Unified client supporting multiple modes with recursion safeguards."""

    def __init__(
        self,
        mode: ClientMode = ClientMode.CLAUDE_CLI,
        claude_path: str = "claude",
        claude_wrapper_path: str = "claude-wrapper",
        api_base_url: str = "http://localhost:8000",
        api_key: str | None = None,
        timeout: float = 120.0,
        retry_attempts: int = 3,
        max_recursion_depth: int = 10,
        enable_recursion_tracking: bool = True,
    ):
        """Initialize unified Claude client.

        Args:
            mode: Client operation mode
            claude_path: Path to Claude CLI executable
            claude_wrapper_path: Path to claude-wrapper CLI executable
            api_base_url: Base URL for API mode
            api_key: API key for authentication
            timeout: Request/command timeout in seconds
            retry_attempts: Number of retry attempts
            max_recursion_depth: Maximum allowed recursion depth
            enable_recursion_tracking: Enable recursion detection and limits
        """
        self.mode = mode
        self.claude_path = claude_path
        self.claude_wrapper_path = claude_wrapper_path
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self.max_recursion_depth = max_recursion_depth
        self.enable_recursion_tracking = enable_recursion_tracking

        self._claude_available: bool | None = None
        self._http_client: ClaudeHTTPClient | None = None

        # Initialize HTTP client for API mode
        if mode == ClientMode.API:
            self._http_client = ClaudeHTTPClient(
                base_url=api_base_url,
                timeout=timeout,
                api_key=api_key,
            )

    async def __aenter__(self):
        """Async context manager entry."""
        if self._http_client:
            await self._http_client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._http_client:
            await self._http_client.__aexit__(exc_type, exc_val, exc_tb)

    def _check_recursion_depth(self, operation: str) -> None:
        """Check and update recursion depth.

        Args:
            operation: Name of the operation being performed

        Raises:
            RecursionError: If recursion depth exceeds limits
        """
        if not self.enable_recursion_tracking:
            return

        # Get current context values
        try:
            current_depth = _recursion_depth.get()
        except LookupError:
            current_depth = 0

        try:
            call_stack = _call_stack.get().copy()
        except LookupError:
            call_stack = []

        # Check depth limit
        if current_depth >= self.max_recursion_depth:
            raise RecursionError(
                f"Maximum recursion depth ({self.max_recursion_depth}) exceeded. "
                f"Call stack: {' -> '.join(call_stack)}"
            )

        # Check for immediate loops
        if len(call_stack) >= 2 and call_stack[-1] == call_stack[-2]:
            raise RecursionError(f"Detected infinite recursion loop in operation: {operation}")

        # Update context
        new_depth = current_depth + 1
        new_stack = call_stack + [f"{self.mode.value}:{operation}"]

        _recursion_depth.set(new_depth)
        _call_stack.set(new_stack)

    def _check_subprocess_recursion(self, operation: str) -> None:
        """Check recursion for subprocess-based calls using thread-local storage."""
        if not self.enable_recursion_tracking:
            return

        if not hasattr(_thread_local, "depth"):
            _thread_local.depth = 0
            _thread_local.stack = []

        if _thread_local.depth >= self.max_recursion_depth:
            raise RecursionError(
                f"Maximum recursion depth ({self.max_recursion_depth}) exceeded. "
                f"Call stack: {' -> '.join(_thread_local.stack)}"
            )

        _thread_local.depth += 1
        _thread_local.stack.append(f"{self.mode.value}:{operation}")

    def _cleanup_subprocess_recursion(self) -> None:
        """Clean up subprocess recursion tracking."""
        if hasattr(_thread_local, "depth") and _thread_local.depth > 0:
            _thread_local.depth -= 1
            if _thread_local.stack:
                _thread_local.stack.pop()

    async def _execute_claude_cli(self, args: list[str], input_text: str | None = None) -> str:
        """Execute Claude CLI command.

        Args:
            args: Command arguments
            input_text: Optional input text

        Returns:
            Command output
        """
        if self._claude_available is None:
            await self._check_claude_auth()

        self._check_subprocess_recursion("claude_cli")

        try:
            for attempt in range(self.retry_attempts):
                try:
                    cmd = [self.claude_path] + args

                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdin=asyncio.subprocess.PIPE if input_text else None,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(input_text.encode() if input_text else None),
                        timeout=self.timeout,
                    )

                    if process.returncode != 0:
                        error_msg = stderr.decode() if stderr else "Unknown error"
                        if attempt < self.retry_attempts - 1:
                            await asyncio.sleep(2**attempt)
                            continue
                        raise ClaudeExecutionError(f"Claude CLI failed: {error_msg}")

                    return stdout.decode().strip()

                except asyncio.TimeoutError as e:
                    if attempt < self.retry_attempts - 1:
                        continue
                    raise ClaudeTimeoutError(f"Claude CLI timed out after {self.timeout}s") from e
                except Exception as e:
                    if isinstance(e, ClaudeWrapperError):
                        raise
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise ClaudeExecutionError(f"Failed to execute Claude CLI: {e}") from e

            raise ClaudeExecutionError("Unexpected execution path")
        finally:
            self._cleanup_subprocess_recursion()

    async def _execute_claude_wrapper_cli(
        self, command: str, message: str, stream: bool = False
    ) -> str:
        """Execute claude-wrapper CLI command.

        Args:
            command: CLI command (e.g., "chat")
            message: Message to send
            stream: Whether to use streaming

        Returns:
            Command output
        """
        if not shutil.which(self.claude_wrapper_path):
            raise ClaudeNotFoundError(f"Claude-wrapper CLI not found at {self.claude_wrapper_path}")

        self._check_subprocess_recursion("claude_wrapper_cli")

        try:
            for attempt in range(self.retry_attempts):
                try:
                    cmd = [self.claude_wrapper_path, command, message]
                    if stream:
                        cmd.append("--stream")

                    process = await asyncio.create_subprocess_exec(
                        *cmd,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )

                    stdout, stderr = await asyncio.wait_for(
                        process.communicate(),
                        timeout=self.timeout,
                    )

                    if process.returncode != 0:
                        error_msg = stderr.decode() if stderr else "Unknown error"
                        if attempt < self.retry_attempts - 1:
                            await asyncio.sleep(2**attempt)
                            continue
                        raise ClaudeExecutionError(f"Claude-wrapper CLI failed: {error_msg}")

                    return stdout.decode().strip()

                except asyncio.TimeoutError as e:
                    if attempt < self.retry_attempts - 1:
                        continue
                    raise ClaudeTimeoutError(
                        f"Claude-wrapper CLI timed out after {self.timeout}s"
                    ) from e
                except Exception as e:
                    if isinstance(e, ClaudeWrapperError):
                        raise
                    if attempt < self.retry_attempts - 1:
                        await asyncio.sleep(2**attempt)
                        continue
                    raise ClaudeExecutionError(f"Failed to execute claude-wrapper CLI: {e}") from e

            raise ClaudeExecutionError("Unexpected execution path")
        finally:
            self._cleanup_subprocess_recursion()

    async def _check_claude_auth(self) -> bool:
        """Check Claude CLI authentication."""
        if not shutil.which(self.claude_path):
            raise ClaudeNotFoundError(f"Claude CLI not found at {self.claude_path}")

        try:
            process = await asyncio.create_subprocess_exec(
                self.claude_path,
                "--version",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)

            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "Unknown error"
                if "not authenticated" in error_msg.lower() or "auth" in error_msg.lower():
                    raise ClaudeAuthError("Claude CLI is not authenticated")
                raise ClaudeExecutionError(f"Claude CLI check failed: {error_msg}")

            self._claude_available = True
            return True

        except asyncio.TimeoutError as e:
            raise ClaudeTimeoutError("Claude CLI auth check timed out") from e
        except Exception as e:
            if isinstance(e, ClaudeWrapperError):
                raise
            raise ClaudeExecutionError(f"Failed to check Claude CLI auth: {e}") from e

    async def chat(self, message: str, **kwargs: Any) -> str:
        """Send a chat message.

        Args:
            message: Message to send
            **kwargs: Additional parameters

        Returns:
            Response from Claude

        Raises:
            ClaudeWrapperError: On execution failure
            RecursionError: If recursion limits exceeded
        """
        self._check_recursion_depth("chat")

        if self.mode == ClientMode.CLAUDE_CLI:
            return await self._execute_claude_cli(["-p", message])

        elif self.mode == ClientMode.CLAUDE_WRAPPER_CLI:
            return await self._execute_claude_wrapper_cli("chat", message)

        elif self.mode == ClientMode.API:
            if not self._http_client:
                raise ClaudeWrapperError("HTTP client not initialized for API mode")
            return await self._http_client.chat(message, **kwargs)

        else:
            raise ClaudeWrapperError(f"Unsupported client mode: {self.mode}")

    async def stream_chat(self, message: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream a chat response.

        Args:
            message: Message to send
            **kwargs: Additional parameters

        Yields:
            Response chunks

        Raises:
            ClaudeWrapperError: On execution failure
            RecursionError: If recursion limits exceeded
        """
        self._check_recursion_depth("stream_chat")

        if self.mode == ClientMode.CLAUDE_CLI:
            # Simulate streaming with word-by-word output
            response = await self._execute_claude_cli(["-p", message])
            words = response.split()
            for i, word in enumerate(words):
                if i > 0:
                    yield " "
                yield word
                await asyncio.sleep(0.01)

        elif self.mode == ClientMode.CLAUDE_WRAPPER_CLI:
            # For CLI mode, we'll fall back to regular chat and simulate streaming
            response = await self._execute_claude_wrapper_cli("chat", message, stream=True)
            words = response.split()
            for i, word in enumerate(words):
                if i > 0:
                    yield " "
                yield word
                await asyncio.sleep(0.01)

        elif self.mode == ClientMode.API:
            if not self._http_client:
                raise ClaudeWrapperError("HTTP client not initialized for API mode")
            async for chunk in self._http_client.stream_chat(message, **kwargs):
                yield chunk

        else:
            raise ClaudeWrapperError(f"Unsupported client mode: {self.mode}")

    async def health_check(self) -> bool:
        """Check if the client can communicate successfully.

        Returns:
            True if healthy

        Raises:
            ClaudeWrapperError: On health check failure
        """
        if self.mode == ClientMode.CLAUDE_CLI:
            return await self._check_claude_auth()

        elif self.mode == ClientMode.CLAUDE_WRAPPER_CLI:
            try:
                if not shutil.which(self.claude_wrapper_path):
                    raise ClaudeNotFoundError(
                        f"Claude-wrapper CLI not found at {self.claude_wrapper_path}"
                    )

                process = await asyncio.create_subprocess_exec(
                    self.claude_wrapper_path,
                    "version",
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                )

                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=10.0)
                return process.returncode == 0

            except Exception as e:
                raise ClaudeWrapperError(f"Claude-wrapper CLI health check failed: {e}") from e

        elif self.mode == ClientMode.API:
            if not self._http_client:
                raise ClaudeWrapperError("HTTP client not initialized for API mode")
            return await self._http_client.health_check()

        else:
            raise ClaudeWrapperError(f"Unsupported client mode: {self.mode}")

    def get_recursion_info(self) -> dict[str, Any]:
        """Get current recursion information.

        Returns:
            Dictionary with recursion depth and call stack
        """
        if not self.enable_recursion_tracking:
            return {"tracking_enabled": False}

        try:
            depth = _recursion_depth.get()
        except LookupError:
            depth = 0

        try:
            stack = _call_stack.get()
        except LookupError:
            stack = []

        # Also check thread-local for subprocess calls
        thread_depth = getattr(_thread_local, "depth", 0)
        thread_stack = getattr(_thread_local, "stack", [])

        return {
            "tracking_enabled": True,
            "max_depth": self.max_recursion_depth,
            "current_depth": max(depth, thread_depth),
            "call_stack": stack if stack else thread_stack,
            "context_depth": depth,
            "thread_depth": thread_depth,
        }


# Convenience functions for easy client creation
def create_claude_client(**kwargs) -> UnifiedClaudeClient:
    """Create a client for Claude CLI."""
    return UnifiedClaudeClient(mode=ClientMode.CLAUDE_CLI, **kwargs)


def create_claude_wrapper_client(**kwargs) -> UnifiedClaudeClient:
    """Create a client for claude-wrapper CLI."""
    return UnifiedClaudeClient(mode=ClientMode.CLAUDE_WRAPPER_CLI, **kwargs)


def create_api_client(**kwargs) -> UnifiedClaudeClient:
    """Create a client for claude-wrapper API."""
    return UnifiedClaudeClient(mode=ClientMode.API, **kwargs)
