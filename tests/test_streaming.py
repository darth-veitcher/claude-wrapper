"""Tests for streaming utilities."""

import asyncio
import json
import time
from unittest.mock import patch

import pytest

from claude_wrapper.utils.streaming import StreamProcessor


class TestStreamProcessor:
    """Test suite for StreamProcessor."""

    @pytest.mark.unit
    def test_format_sse_with_dict(self):
        """Test formatting dictionary as SSE."""
        processor = StreamProcessor()
        data = {"key": "value", "number": 42}

        result = processor.format_sse(data)

        assert "data: " in result
        assert json.dumps(data) in result
        assert result.endswith("\n\n")

    @pytest.mark.unit
    def test_format_sse_with_event(self):
        """Test formatting SSE with event type."""
        processor = StreamProcessor()
        data = {"message": "test"}

        result = processor.format_sse(data, event="custom")

        assert "event: custom\n" in result
        assert "data: " in result
        assert json.dumps(data) in result

    @pytest.mark.unit
    def test_format_sse_done_marker(self):
        """Test formatting [DONE] marker."""
        processor = StreamProcessor()

        result = processor.format_sse("[DONE]")

        assert result == "data: [DONE]\n\n"

    @pytest.mark.unit
    def test_create_openai_stream_chunk_with_content(self):
        """Test creating OpenAI-compatible stream chunk with content."""
        processor = StreamProcessor()

        chunk = processor.create_openai_stream_chunk(
            content="Hello", model="claude-3-opus-20240229", index=0
        )

        assert chunk["object"] == "chat.completion.chunk"
        assert chunk["model"] == "claude-3-opus-20240229"
        assert chunk["choices"][0]["delta"]["content"] == "Hello"
        assert chunk["choices"][0]["index"] == 0
        assert chunk["choices"][0]["finish_reason"] is None
        assert "id" in chunk
        assert "created" in chunk

    @pytest.mark.unit
    def test_create_openai_stream_chunk_with_role(self):
        """Test creating stream chunk with role."""
        processor = StreamProcessor()

        chunk = processor.create_openai_stream_chunk(
            role="assistant", model="claude-3-opus-20240229"
        )

        assert chunk["choices"][0]["delta"]["role"] == "assistant"
        assert "content" not in chunk["choices"][0]["delta"]

    @pytest.mark.unit
    def test_create_openai_stream_chunk_with_finish_reason(self):
        """Test creating stream chunk with finish reason."""
        processor = StreamProcessor()

        chunk = processor.create_openai_stream_chunk(
            finish_reason="stop", model="claude-3-opus-20240229"
        )

        assert chunk["choices"][0]["finish_reason"] == "stop"
        assert chunk["choices"][0]["delta"] == {}

    @pytest.mark.unit
    async def test_parse_claude_stream_simple(self):
        """Test parsing Claude stream into chunks."""
        processor = StreamProcessor()

        async def mock_stream():
            chunks = ["Hello ", "world", "! ", "How ", "are ", "you?"]
            for chunk in chunks:
                yield chunk

        parsed_chunks = []
        async for chunk in processor.parse_claude_stream(mock_stream()):
            parsed_chunks.append(chunk)

        # Should have parsed chunks with type and content
        assert len(parsed_chunks) > 0
        for chunk in parsed_chunks[:-1]:
            assert chunk["type"] == "content"
            assert "content" in chunk
            assert chunk["finished"] is False

        # Last chunk should be marked as finished
        assert parsed_chunks[-1]["finished"] is True

    @pytest.mark.unit
    async def test_parse_claude_stream_with_sentences(self):
        """Test parsing stream with sentence detection."""
        processor = StreamProcessor()

        async def mock_stream():
            # Stream that builds up to complete sentences
            chunks = ["This ", "is ", "a ", "sentence. ", "And ", "another ", "one!"]
            for chunk in chunks:
                yield chunk

        parsed_chunks = []
        async for chunk in processor.parse_claude_stream(mock_stream()):
            parsed_chunks.append(chunk)

        # Should detect sentence boundaries
        sentence_chunks = [
            c for c in parsed_chunks if ". " in c.get("content", "") or "!" in c.get("content", "")
        ]
        assert len(sentence_chunks) > 0

    @pytest.mark.unit
    async def test_parse_claude_stream_large_buffer(self):
        """Test handling of large buffer in stream parsing."""
        processor = StreamProcessor()

        async def mock_stream():
            # Create a large chunk without delimiters
            large_chunk = "a" * 150  # Over 100 char threshold
            yield large_chunk

        parsed_chunks = []
        async for chunk in processor.parse_claude_stream(mock_stream()):
            parsed_chunks.append(chunk)

        # Should yield the large chunk even without delimiters
        assert len(parsed_chunks) > 0
        assert len(parsed_chunks[0]["content"]) == 150

    @pytest.mark.unit
    async def test_rate_limit_stream(self):
        """Test rate limiting of stream."""
        processor = StreamProcessor()

        async def mock_stream():
            for i in range(5):
                yield f"chunk_{i}"

        start_time = time.time()
        chunks = []

        async for chunk in processor.rate_limit_stream(
            mock_stream(), min_delay=0.01, max_delay=0.1
        ):
            chunks.append(chunk)

        elapsed = time.time() - start_time

        assert len(chunks) == 5
        # Should have taken at least min_delay * (chunks - 1)
        assert elapsed >= 0.04  # 4 delays of 0.01s minimum

    @pytest.mark.unit
    async def test_rate_limit_stream_no_delay_when_slow(self):
        """Test that rate limiter doesn't add delay when stream is already slow."""
        processor = StreamProcessor()

        async def mock_stream():
            for i in range(3):
                await asyncio.sleep(0.2)  # Slow stream
                yield f"chunk_{i}"

        start_time = time.time()
        chunks = []

        async for chunk in processor.rate_limit_stream(
            mock_stream(), min_delay=0.01, max_delay=0.1
        ):
            chunks.append(chunk)

        elapsed = time.time() - start_time

        assert len(chunks) == 3
        # Should not add significant extra delay
        assert elapsed < 0.7  # 3 * 0.2s + small overhead

    @pytest.mark.unit
    def test_sse_format_unicode(self):
        """Test SSE formatting with Unicode characters."""
        processor = StreamProcessor()
        data = {"message": "Hello 世界 🌍"}

        result = processor.format_sse(data)

        # Should handle Unicode properly
        assert "Hello 世界 🌍" in result
        # ensure_ascii=False should preserve Unicode
        assert "\\u" not in result

    @pytest.mark.unit
    def test_openai_chunk_id_generation(self):
        """Test that chunk IDs are unique."""
        processor = StreamProcessor()

        chunk1 = processor.create_openai_stream_chunk(content="test1")
        chunk2 = processor.create_openai_stream_chunk(content="test2")

        # IDs should be different
        assert chunk1["id"] != chunk2["id"]

        # IDs should follow OpenAI format
        assert chunk1["id"].startswith("chatcmpl-")
        assert chunk2["id"].startswith("chatcmpl-")

    @pytest.mark.unit
    def test_openai_chunk_timestamp(self):
        """Test that chunks have proper timestamps."""
        processor = StreamProcessor()

        with patch("time.time", return_value=1234567890.123):
            chunk = processor.create_openai_stream_chunk(content="test")

            assert chunk["created"] == 1234567890
            assert "1234567890" in chunk["id"]  # ID includes timestamp

    @pytest.mark.unit
    @pytest.mark.parametrize("finish_reason", ["stop", "length", "function_call", "content_filter"])
    def test_openai_chunk_finish_reasons(self, finish_reason):
        """Test different finish reasons in OpenAI chunks."""
        processor = StreamProcessor()

        chunk = processor.create_openai_stream_chunk(
            finish_reason=finish_reason, model="claude-3-opus-20240229"
        )

        assert chunk["choices"][0]["finish_reason"] == finish_reason
        assert chunk["choices"][0]["delta"] == {}  # No content with finish reason
