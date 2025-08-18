"""Shared pytest fixtures and configuration for Claude Wrapper tests."""

import asyncio
import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
import pytest_asyncio
from faker import Faker
from freezegun import freeze_time

from claude_wrapper.core import ClaudeClient
from claude_wrapper.utils.config import Config

# Initialize Faker for test data generation
fake = Faker()


@pytest.fixture(scope="session")
def event_loop():
    """Create an event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def mock_config(_temp_dir: Path) -> Config:
    """Create a mock configuration."""
    return Config(
        claude_path="/usr/bin/claude",
        timeout=30,
        retry_attempts=3,
        api_key=None,
    )


@pytest.fixture
def mock_subprocess():
    """Mock subprocess.run for Claude CLI calls."""
    with patch("asyncio.create_subprocess_exec") as mock_proc:
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"Test response", b""))
        process.returncode = 0
        process.stdout = AsyncMock()
        process.stdout.readline = AsyncMock(return_value=b"Stream chunk\n")
        process.stderr = AsyncMock()
        process.stderr.readline = AsyncMock(return_value=b"")
        mock_proc.return_value = process
        yield mock_proc


@pytest.fixture
def mock_subprocess_error():
    """Mock subprocess.run to simulate Claude CLI errors."""
    with patch("asyncio.create_subprocess_exec") as mock_proc:
        process = AsyncMock()
        process.communicate = AsyncMock(return_value=(b"", b"Error: Authentication failed"))
        process.returncode = 1
        mock_proc.return_value = process
        yield mock_proc


@pytest_asyncio.fixture
async def claude_client(mock_config: Config) -> ClaudeClient:
    """Create a ClaudeClient instance."""
    with patch("shutil.which", return_value="/usr/bin/claude"), patch("subprocess.run") as mock_run:
        mock_run.return_value.returncode = 0
        return ClaudeClient(
            claude_path=mock_config.claude_path,
            timeout=mock_config.timeout,
            retry_attempts=mock_config.retry_attempts,
        )


# Removed sample_session fixture as sessions no longer exist


@pytest.fixture
def sample_messages() -> list[dict[str, str]]:
    """Create sample chat messages."""
    return [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language"},
        {"role": "user", "content": "Tell me more"},
    ]


@pytest.fixture
def mock_openai_request() -> dict[str, Any]:
    """Create a mock OpenAI-compatible request."""
    return {
        "model": "sonnet",
        "messages": [
            {"role": "system", "content": "You are helpful"},
            {"role": "user", "content": "Hello"},
        ],
        "max_tokens": 100,
        "temperature": 0.7,
        "stream": False,
    }


@pytest.fixture
def mock_time():
    """Freeze time for consistent testing."""
    with freeze_time("2024-01-01 12:00:00"):
        yield


# Removed mock_file_system fixture as sessions no longer exist


@pytest.fixture
def mock_claude_response():
    """Create a mock Claude response."""
    return "This is a test response from Claude. It contains multiple sentences for testing streaming functionality."


@pytest.fixture
def mock_stream_chunks():
    """Create mock streaming chunks."""
    return [
        "This is ",
        "a streaming ",
        "response ",
        "from Claude.",
    ]


@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset any singleton instances between tests."""
    # Reset config singleton
    import claude_wrapper.utils.config

    claude_wrapper.utils.config._config_instance = None
    yield


@pytest.fixture
def mock_httpx_client():
    """Mock httpx client for API testing."""
    with patch("httpx.AsyncClient") as mock_client:
        client = AsyncMock()
        mock_client.return_value.__aenter__.return_value = client
        mock_client.return_value.__aexit__.return_value = None
        yield client


@pytest.fixture
def cli_runner():
    """Create a Typer test runner."""
    from typer.testing import CliRunner

    return CliRunner()


# Markers for test organization
def pytest_configure(config):
    """Configure pytest with custom markers."""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "requires_claude: Tests requiring Claude CLI")
    config.addinivalue_line("markers", "mock: Tests using mocked dependencies")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "cli: CLI command tests")
    config.addinivalue_line("markers", "core: Core functionality tests")


# Test helpers
class MockProcess:
    """Mock process for subprocess testing."""

    def __init__(self, stdout: bytes = b"", stderr: bytes = b"", returncode: int = 0):
        self.stdout = AsyncMock()
        self.stderr = AsyncMock()
        self.returncode = returncode
        self._stdout = stdout
        self._stderr = stderr

    async def communicate(self):
        return self._stdout, self._stderr

    async def wait(self):
        return self.returncode


def create_mock_response(content: str, status_code: int = 200) -> Mock:
    """Create a mock HTTP response."""
    response = Mock()
    response.status_code = status_code
    response.text = content
    response.json = Mock(return_value=json.loads(content) if content else {})
    return response


# Removed create_test_session helper as sessions no longer exist
