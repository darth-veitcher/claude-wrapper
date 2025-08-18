"""Configuration management for Claude Wrapper."""

from pathlib import Path

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):
    """Configuration settings for Claude Wrapper."""

    model_config = SettingsConfigDict(
        env_prefix="CLAUDE_WRAPPER_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # CLI settings
    claude_path: str = Field(default="claude", description="Path to Claude CLI executable")
    timeout: float = Field(default=300.0, description="Command timeout in seconds")
    retry_attempts: int = Field(default=3, description="Number of retry attempts")
    retry_delay: float = Field(default=1.0, description="Delay between retries in seconds")

    # API server settings
    api_key: str | None = Field(default=None, description="API key for server authentication")
    api_base_url: str = Field(default="http://localhost:8000", description="API base URL")
    api_model: str = Field(default="claude-3-opus-20240229", description="Default model name")

    # Session settings
    session_storage_dir: Path | None = Field(default=None, description="Session storage directory")
    session_cleanup_days: int = Field(default=30, description="Days before session cleanup")

    # Streaming settings
    stream_chunk_size: int = Field(default=1024, description="Streaming chunk size in bytes")
    stream_timeout: float = Field(default=60.0, description="Streaming timeout in seconds")

    # Feature flags
    enable_caching: bool = Field(default=True, description="Enable response caching")
    enable_telemetry: bool = Field(default=False, description="Enable anonymous telemetry")

    @classmethod
    def from_file(cls, path: Path) -> "Config":
        """Load configuration from YAML file."""
        if path.exists():
            with open(path) as f:
                data = yaml.safe_load(f)
                return cls(**data)
        return cls()

    def to_yaml(self) -> str:
        """Export configuration to YAML string."""
        data = self.model_dump(exclude_none=True)
        return str(yaml.dump(data, default_flow_style=False))

    def save(self, path: Path | None = None) -> None:
        """Save configuration to file."""
        if path is None:
            path = Path.home() / ".claude-wrapper" / "config.yaml"

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_yaml())

    @property
    def config_dir(self) -> Path:
        """Get configuration directory."""
        return Path.home() / ".claude-wrapper"

    @property
    def cache_dir(self) -> Path:
        """Get cache directory."""
        cache_dir = self.config_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        return cache_dir


# Global config instance
_config_instance: Config | None = None


def get_config() -> Config:
    """Get global configuration instance (singleton)."""
    global _config_instance
    if _config_instance is None:
        config_file = Path.home() / ".claude-wrapper" / "config.yaml"
        _config_instance = Config.from_file(config_file) if config_file.exists() else Config()
    return _config_instance
