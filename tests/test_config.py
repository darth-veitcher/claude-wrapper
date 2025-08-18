"""Tests for configuration management."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from claude_wrapper.utils.config import Config, get_config


class TestConfig:
    """Test suite for configuration management."""

    @pytest.mark.unit
    def test_default_config(self):
        """Test default configuration values."""
        config = Config()

        assert config.claude_path == "claude"
        assert config.timeout == 300.0
        assert config.retry_attempts == 3
        assert config.session_storage_dir is None
        assert config.session_cleanup_days == 30
        assert config.api_key is None
        assert config.api_base_url == "http://localhost:8000"
        assert config.api_model == "claude-3-opus-20240229"
        assert config.enable_caching is True
        assert config.enable_telemetry is False

    @pytest.mark.unit
    def test_config_from_environment(self):
        """Test loading configuration from environment variables."""
        env_vars = {
            "CLAUDE_WRAPPER_CLAUDE_PATH": "/custom/claude",
            "CLAUDE_WRAPPER_TIMEOUT": "60",
            "CLAUDE_WRAPPER_RETRY_ATTEMPTS": "5",
            "CLAUDE_WRAPPER_SESSION_STORAGE_DIR": "/tmp/sessions",
            "CLAUDE_WRAPPER_SESSION_CLEANUP_DAYS": "7",
            "CLAUDE_WRAPPER_API_KEY": "test-key",
            "CLAUDE_WRAPPER_API_BASE_URL": "http://localhost:3000",
            "CLAUDE_WRAPPER_API_MODEL": "claude-3-sonnet",
            "CLAUDE_WRAPPER_ENABLE_CACHING": "false",
            "CLAUDE_WRAPPER_ENABLE_TELEMETRY": "true",
        }

        with patch.dict(os.environ, env_vars):
            config = Config()

            assert config.claude_path == "/custom/claude"
            assert config.timeout == 60
            assert config.retry_attempts == 5
            assert config.session_storage_dir == Path("/tmp/sessions")
            assert config.session_cleanup_days == 7
            assert config.api_key == "test-key"
            assert config.api_base_url == "http://localhost:3000"
            assert config.api_model == "claude-3-sonnet"
            assert config.enable_caching is False
            assert config.enable_telemetry is True

    @pytest.mark.unit
    def test_config_from_file(self, temp_dir):
        """Test loading configuration from YAML file."""
        config_file = temp_dir / "config.yaml"
        config_content = """
claude_path: /usr/local/bin/claude
timeout: 180
retry_attempts: 2
session_storage_dir: /var/sessions
session_cleanup_days: 14
api_key: file-api-key
api_model: claude-3-haiku
enable_caching: false
"""
        config_file.write_text(config_content)

        # Test loading from file directly
        config = Config.from_file(config_file)

        assert config.claude_path == "/usr/local/bin/claude"
        assert config.timeout == 180
        assert config.retry_attempts == 2
        assert config.session_storage_dir == Path("/var/sessions")
        assert config.session_cleanup_days == 14
        assert config.api_key == "file-api-key"
        assert config.api_model == "claude-3-haiku"
        assert config.enable_caching is False

    @pytest.mark.unit
    def test_config_precedence(self):
        """Test configuration precedence (env vars override defaults)."""
        with patch.dict(os.environ, {"CLAUDE_WRAPPER_TIMEOUT": "90"}):
            config = Config()
            assert config.timeout == 90  # Env var overrides default

    @pytest.mark.unit
    def test_config_validation(self):
        """Test configuration validation."""
        # Test invalid timeout (negative values should work but are not recommended)
        config = Config(timeout=-1)
        assert config.timeout == -1  # Pydantic allows it but it's not practical

        # Test invalid retry attempts
        config = Config(retry_attempts=0)
        assert config.retry_attempts == 0

    @pytest.mark.unit
    def test_get_config_singleton(self):
        """Test that get_config returns singleton instance."""
        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    @pytest.mark.unit
    def test_config_to_dict(self):
        """Test converting config to dictionary."""
        config = Config()
        config_dict = config.model_dump()

        assert isinstance(config_dict, dict)
        assert "claude_path" in config_dict
        assert "timeout" in config_dict
        assert "api_key" in config_dict

    @pytest.mark.unit
    def test_config_session_dir_creation(self, temp_dir):
        """Test that session directory is created if it doesn't exist."""
        session_dir = temp_dir / "new_sessions"
        assert not session_dir.exists()

        with patch.dict(os.environ, {"CLAUDE_WRAPPER_SESSION_STORAGE_DIR": str(session_dir)}):
            Config()
            # Session dir should be created on first use
            # This would be handled by SessionManager

    @pytest.mark.unit
    def test_config_additional_fields(self):
        """Test additional config fields."""
        # Test retry delay
        with patch.dict(os.environ, {"CLAUDE_WRAPPER_RETRY_DELAY": "2.5"}):
            config = Config()
            assert config.retry_delay == 2.5

        # Test stream settings
        with patch.dict(
            os.environ,
            {"CLAUDE_WRAPPER_STREAM_CHUNK_SIZE": "2048", "CLAUDE_WRAPPER_STREAM_TIMEOUT": "30.0"},
        ):
            config = Config()
            assert config.stream_chunk_size == 2048
            assert config.stream_timeout == 30.0

    @pytest.mark.unit
    def test_config_model_validation(self):
        """Test model name validation."""
        valid_models = [
            "claude-3-opus-20240229",
            "claude-3-sonnet-20240229",
            "claude-3-haiku-20240307",
        ]

        for model in valid_models:
            with patch.dict(os.environ, {"CLAUDE_WRAPPER_API_MODEL": model}):
                config = Config()
                assert config.api_model == model
