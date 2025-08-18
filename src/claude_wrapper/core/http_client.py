"""HTTP client for claude-wrapper API calls."""

from collections.abc import AsyncIterator
from typing import Any

import httpx

from .exceptions import (
    ClaudeAuthError,
    ClaudeExecutionError,
    ClaudeTimeoutError,
    ClaudeWrapperError,
)


class ClaudeHTTPClient:
    """HTTP client for calling claude-wrapper API endpoints."""

    def __init__(
        self,
        base_url: str = "http://localhost:8000",
        timeout: float = 120.0,
        api_key: str | None = None,
    ):
        """Initialize HTTP client for claude-wrapper API.

        Args:
            base_url: Base URL of the claude-wrapper API server
            timeout: Request timeout in seconds
            api_key: Optional API key for authentication
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        """Async context manager entry."""
        await self._ensure_client()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def _ensure_client(self) -> httpx.AsyncClient:
        """Ensure HTTP client is initialized."""
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.api_key:
                headers["Authorization"] = f"Bearer {self.api_key}"

            self._client = httpx.AsyncClient(
                base_url=self.base_url,
                timeout=httpx.Timeout(self.timeout),
                headers=headers,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def chat(self, message: str, model: str = "sonnet", **kwargs: Any) -> str:
        """Send a chat message via API.

        Args:
            message: Message to send
            model: Model to use
            **kwargs: Additional parameters for the API

        Returns:
            Response from Claude

        Raises:
            ClaudeWrapperError: On API error
        """
        client = await self._ensure_client()

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            **kwargs,
        }

        try:
            response = await client.post("/v1/chat/completions", json=payload)
            response.raise_for_status()

            data = response.json()
            if "choices" in data and len(data["choices"]) > 0:
                return data["choices"][0]["message"]["content"]
            else:
                raise ClaudeExecutionError("Invalid API response format")

        except httpx.TimeoutException as e:
            raise ClaudeTimeoutError(f"API request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ClaudeAuthError("API authentication failed") from e
            else:
                raise ClaudeExecutionError(f"API request failed: {e}") from e
        except Exception as e:
            raise ClaudeWrapperError(f"Unexpected API error: {e}") from e

    async def stream_chat(
        self, message: str, model: str = "sonnet", **kwargs: Any
    ) -> AsyncIterator[str]:
        """Stream a chat response via API.

        Args:
            message: Message to send
            model: Model to use
            **kwargs: Additional parameters

        Yields:
            Response chunks

        Raises:
            ClaudeWrapperError: On API error
        """
        client = await self._ensure_client()

        payload = {
            "model": model,
            "messages": [{"role": "user", "content": message}],
            "stream": True,
            **kwargs,
        }

        try:
            async with client.stream("POST", "/v1/chat/completions", json=payload) as response:
                response.raise_for_status()

                async for line in response.aiter_lines():
                    line = line.strip()
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str == "[DONE]":
                            break

                        try:
                            import json

                            data = json.loads(data_str)
                            if (
                                "choices" in data
                                and len(data["choices"]) > 0
                                and "delta" in data["choices"][0]
                                and "content" in data["choices"][0]["delta"]
                            ):
                                content = data["choices"][0]["delta"]["content"]
                                if content:
                                    yield content
                        except json.JSONDecodeError:
                            continue  # Skip malformed JSON lines

        except httpx.TimeoutException as e:
            raise ClaudeTimeoutError(f"Streaming API request timed out: {e}") from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 401:
                raise ClaudeAuthError("API authentication failed") from e
            else:
                raise ClaudeExecutionError(f"Streaming API request failed: {e}") from e
        except Exception as e:
            raise ClaudeWrapperError(f"Unexpected streaming API error: {e}") from e

    async def health_check(self) -> bool:
        """Check if the API server is healthy.

        Returns:
            True if server is responding

        Raises:
            ClaudeWrapperError: If server is not accessible
        """
        client = await self._ensure_client()

        try:
            response = await client.get("/")
            response.raise_for_status()
            return True
        except Exception as e:
            raise ClaudeWrapperError(f"API server health check failed: {e}") from e
