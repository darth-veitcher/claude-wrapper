"""Simple async client for Claude CLI wrapper."""

import asyncio
import json
import shutil
from collections.abc import AsyncIterator
from typing import Any

from .exceptions import (
    ClaudeAuthError,
    ClaudeExecutionError,
    ClaudeNotFoundError,
    ClaudeTimeoutError,
    ClaudeWrapperError,
)


class ClaudeClient:
    """Simple async client for interacting with Claude CLI."""

    def __init__(
        self,
        claude_path: str = "claude",
        timeout: float = 120.0,
        retry_attempts: int = 3,
    ):
        """Initialize Claude client.

        Args:
            claude_path: Path to Claude CLI executable
            timeout: Command timeout in seconds
            retry_attempts: Number of retry attempts for failed commands
        """
        self.claude_path = claude_path
        self.timeout = timeout
        self.retry_attempts = retry_attempts
        self._claude_available: bool | None = None

    async def check_auth(self) -> bool:
        """Check if Claude CLI is installed and authenticated.

        Returns:
            True if Claude is ready to use

        Raises:
            ClaudeNotFoundError: If Claude CLI is not found
            ClaudeAuthError: If Claude is not authenticated
        """
        # Check if Claude CLI exists
        if not shutil.which(self.claude_path):
            raise ClaudeNotFoundError(f"Claude CLI not found at {self.claude_path}")

        # Try a simple command to check auth
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
                    raise ClaudeAuthError(
                        "Claude CLI is not authenticated. Run 'claude auth' first."
                    )
                raise ClaudeExecutionError(f"Claude check failed: {error_msg}")

            self._claude_available = True
            return True

        except asyncio.TimeoutError as e:
            raise ClaudeTimeoutError("Claude auth check timed out") from e
        except Exception as e:
            if isinstance(e, ClaudeWrapperError):
                raise
            raise ClaudeExecutionError(f"Failed to check Claude auth: {str(e)}") from e

        # This should never be reached, but satisfies mypy
        return True

    async def _execute_claude(
        self,
        args: list[str],
        input_text: str | None = None,
    ) -> str:
        """Execute a Claude CLI command.

        Args:
            args: Command arguments
            input_text: Optional input to pipe to the command

        Returns:
            Command output

        Raises:
            ClaudeWrapperError: On execution failure
        """
        if self._claude_available is None:
            await self.check_auth()

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
                        await asyncio.sleep(2**attempt)  # Exponential backoff
                        continue
                    raise ClaudeExecutionError(f"Claude command failed: {error_msg}")

                return stdout.decode()

            except asyncio.TimeoutError as e:
                if attempt < self.retry_attempts - 1:
                    continue
                raise ClaudeTimeoutError(f"Claude command timed out after {self.timeout}s") from e
            except Exception as e:
                if isinstance(e, ClaudeWrapperError):
                    raise
                if attempt < self.retry_attempts - 1:
                    await asyncio.sleep(2**attempt)
                    continue
                raise ClaudeExecutionError(f"Failed to execute Claude command: {str(e)}") from e

        # This should never be reached, but satisfies mypy
        raise ClaudeExecutionError("Unexpected execution path")

    async def chat(self, message: str) -> str:
        """Send a message to Claude and get a response.

        Args:
            message: The message to send

        Returns:
            Claude's response
        """
        # Use -p flag for prompt
        response = await self._execute_claude(["-p", message])
        return response.strip()

    async def stream_chat(self, message: str) -> AsyncIterator[str]:
        """Stream a response from Claude using stream-json output format.

        Uses ``--output-format stream-json --verbose --include-partial-messages``
        for true token-by-token streaming.  Falls back to simulating streaming
        word-by-word (with a short delay) if the subprocess approach fails.

        Args:
            message: The message to send

        Yields:
            Response text chunks as they arrive
        """
        if self._claude_available is None:
            await self.check_auth()

        try:
            process = await asyncio.create_subprocess_exec(
                self.claude_path,
                "-p",
                message,
                "--output-format",
                "stream-json",
                "--verbose",
                "--include-partial-messages",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            assert process.stdout is not None
            last_text = ""

            async for raw_line in process.stdout:
                line = raw_line.decode().strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if event.get("type") != "assistant":
                    continue

                for block in event.get("message", {}).get("content", []):
                    if block.get("type") != "text":
                        continue
                    text: str = block["text"]
                    # Skip empty text blocks — Claude CLI occasionally emits
                    # content_block_start without any delta, and forwarding an
                    # empty chunk triggers Anthropic 400 errors upstream.
                    if not text:
                        continue
                    # Each partial event contains the cumulative text so far;
                    # yield only the delta since the previous event.
                    if text.startswith(last_text):
                        delta = text[len(last_text) :]
                        if delta:
                            yield delta
                    else:
                        # Non-cumulative event — yield the full block
                        yield text
                    last_text = text

            await asyncio.wait_for(process.wait(), timeout=self.timeout)

        except ClaudeWrapperError:
            raise
        except Exception:
            # Fallback: fetch the full response and emit word-by-word
            response = await self.chat(message)
            words = response.split()
            for i, word in enumerate(words):
                if i > 0:
                    yield " "
                yield word
                await asyncio.sleep(0.01)

    async def complete(
        self,
        prompt: str,
        _max_tokens: int | None = None,
        _temperature: float | None = None,
        _stop_sequences: list[str] | None = None,
    ) -> str:
        """Get a completion from Claude.

        Args:
            prompt: The prompt to complete
            max_tokens: Maximum tokens in response (not supported by CLI)
            temperature: Temperature for sampling (not supported by CLI)
            stop_sequences: Sequences to stop generation (not supported by CLI)

        Returns:
            The completion
        """
        # Claude CLI doesn't support these parameters directly,
        # so we just pass the prompt
        return await self.chat(prompt)

    async def count_tokens(self, text: str) -> dict[str, Any]:
        """Count tokens in text.

        Args:
            text: Text to count tokens for

        Returns:
            Token count information
        """
        # This is an approximation since Claude CLI doesn't provide token counting
        # Rough estimate: 1 token ≈ 4 characters
        char_count = len(text)
        estimated_tokens = char_count // 4

        return {
            "tokens": estimated_tokens,
            "characters": char_count,
            "note": "This is an approximation",
        }
