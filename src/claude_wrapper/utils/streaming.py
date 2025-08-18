"""Streaming utilities for Claude Wrapper."""

import asyncio
import json
from collections.abc import AsyncGenerator
from typing import Any


class StreamProcessor:
    """Process and format streaming responses."""

    _id_counter = 0

    @staticmethod
    def format_sse(data: dict[str, Any] | str, event: str | None = None) -> str:
        """Format data as Server-Sent Event.

        Args:
            data: Data to send
            event: Optional event type

        Returns:
            SSE formatted string
        """
        lines = []

        if event:
            lines.append(f"event: {event}")

        # Handle special [DONE] message
        if data == "[DONE]":
            lines.append("data: [DONE]")
        else:
            json_data = json.dumps(data, ensure_ascii=False)
            lines.append(f"data: {json_data}")

        lines.append("")  # Empty line to end the event
        return "\n".join(lines) + "\n"

    @staticmethod
    async def parse_claude_stream(
        stream: AsyncGenerator[str, None],
    ) -> AsyncGenerator[dict[str, Any], None]:
        """Parse Claude CLI streaming output into structured chunks.

        Args:
            stream: Raw stream from Claude CLI

        Yields:
            Parsed chunks as dictionaries
        """
        buffer = ""

        async for chunk in stream:
            buffer += chunk

            # Try to detect complete sentences or meaningful chunks
            # This is a simple heuristic - can be improved
            if any(delimiter in buffer for delimiter in ["\n", ". ", "! ", "? "]):
                # Find the last complete delimiter
                for delimiter in ["\n", ". ", "! ", "? "]:
                    if delimiter in buffer:
                        parts = buffer.rsplit(delimiter, 1)
                        if len(parts) > 1:
                            complete_part = parts[0] + delimiter
                            buffer = parts[1]

                            yield {"type": "content", "content": complete_part, "finished": False}
                            break

            # Yield partial content if buffer gets too large
            elif len(buffer) > 100:
                yield {"type": "content", "content": buffer, "finished": False}
                buffer = ""

        # Yield any remaining content
        if buffer:
            yield {"type": "content", "content": buffer, "finished": True}

    @classmethod
    def create_openai_stream_chunk(
        cls,
        content: str | None = None,
        role: str | None = None,
        finish_reason: str | None = None,
        model: str = "claude-3-opus-20240229",
        index: int = 0,
    ) -> dict[str, Any]:
        """Create an OpenAI-compatible streaming chunk.

        Args:
            content: Content to include in delta
            role: Role for the delta
            finish_reason: Finish reason if applicable
            model: Model name
            index: Choice index

        Returns:
            OpenAI-compatible chunk dictionary
        """
        import time

        # Increment counter for unique IDs
        cls._id_counter += 1

        delta = {}
        if role:
            delta["role"] = role
        if content:
            delta["content"] = content

        return {
            "id": f"chatcmpl-{int(time.time() * 1000)}-{cls._id_counter}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [
                {
                    "index": index,
                    "delta": delta,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
            "system_fingerprint": None,
        }

    @staticmethod
    async def rate_limit_stream(
        stream: AsyncGenerator[str, None],
        min_delay: float = 0.01,
        max_delay: float = 0.1,
    ) -> AsyncGenerator[str, None]:
        """Add rate limiting to a stream to prevent overwhelming clients.

        Args:
            stream: Input stream
            min_delay: Minimum delay between chunks in seconds
            max_delay: Maximum delay between chunks in seconds

        Yields:
            Rate-limited chunks
        """
        last_send = 0.0

        async for chunk in stream:
            now = asyncio.get_event_loop().time()
            elapsed = now - last_send

            if elapsed < min_delay:
                await asyncio.sleep(min_delay - elapsed)
            elif elapsed > max_delay:
                # Don't delay if we're already slow
                pass

            yield chunk
            last_send = asyncio.get_event_loop().time()
