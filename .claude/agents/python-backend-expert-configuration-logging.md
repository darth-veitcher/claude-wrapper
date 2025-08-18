# Configuration and Logging Standards for Python Backend Expert

## 🎯 Overview

This document defines the **consistent, production-ready** configuration and logging standards that the python-backend-expert agent should implement in every project. These patterns ensure that solo developers have a robust foundation that works seamlessly from local development to production deployment.

## 📋 Configuration System

### Core Principles
- **12-Factor App Compliance**: Configuration via environment variables
- **Type Safety**: Full Pydantic validation and type checking
- **Domain Organization**: Settings grouped by functional area
- **Fail-Fast**: Invalid configuration detected at startup
- **Self-Documenting**: Clear defaults and descriptions

### Implementation

#### 1. Project Structure
```
src/
├── config/
│   ├── __init__.py
│   ├── base.py          # Base settings class
│   ├── app.py           # Application settings
│   ├── database.py      # Database settings
│   ├── redis.py         # Redis settings
│   ├── auth.py          # Authentication settings
│   ├── logging.py       # Logging configuration
│   └── validators.py    # Custom validators
├── core/
│   └── config.py        # Main settings aggregator
```

#### 2. Base Configuration Class
```python
# src/config/base.py
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class BaseConfig(BaseSettings):
    """Base configuration with common settings and validators."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        # Allow extra fields for forward compatibility
        extra="ignore",
        # Validate default values
        validate_default=True,
        # Use secrets directory for sensitive data
        secrets_dir="/run/secrets" if Path("/run/secrets").exists() else None,
    )

    # Environment
    ENVIRONMENT: Environment = Field(
        default=Environment.LOCAL,
        description="Current environment"
    )

    # Debugging
    DEBUG: bool = Field(
        default=False,
        description="Debug mode - NEVER true in production"
    )

    @field_validator("DEBUG")
    @classmethod
    def validate_debug(cls, v: bool, info) -> bool:
        """Ensure DEBUG is False in production."""
        env = info.data.get("ENVIRONMENT")
        if env == Environment.PRODUCTION and v:
            raise ValueError("DEBUG cannot be True in production")
        return v

    def is_production(self) -> bool:
        """Check if running in production."""
        return self.ENVIRONMENT == Environment.PRODUCTION

    def is_testing(self) -> bool:
        """Check if running tests."""
        return self.ENVIRONMENT == Environment.TESTING
```

#### 3. Application Settings
```python
# src/config/app.py
from pydantic import Field, HttpUrl, field_validator
from typing import List, Optional

from .base import BaseConfig


class AppConfig(BaseConfig):
    """Application-specific configuration."""

    # Basic Info
    APP_NAME: str = Field(
        default="My FastAPI App",
        description="Application name"
    )
    APP_VERSION: str = Field(
        default="0.1.0",
        description="Application version"
    )

    # API Settings
    API_V1_PREFIX: str = Field(
        default="/api/v1",
        description="API version 1 prefix"
    )
    API_DOCS_ENABLED: bool = Field(
        default=True,
        description="Enable API documentation endpoints"
    )

    @field_validator("API_DOCS_ENABLED")
    @classmethod
    def validate_api_docs(cls, v: bool, info) -> bool:
        """Disable docs in production unless explicitly enabled."""
        if info.data.get("ENVIRONMENT") == "production" and v:
            # This is a warning, not an error - docs CAN be enabled in prod if needed
            import warnings
            warnings.warn("API docs are enabled in production")
        return v

    # CORS Settings
    CORS_ORIGINS: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8001"],
        description="Allowed CORS origins"
    )
    CORS_ALLOW_CREDENTIALS: bool = Field(
        default=True,
        description="Allow credentials in CORS requests"
    )

    # Server Settings
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    WORKERS: int = Field(default=1, description="Number of workers")

    @field_validator("WORKERS")
    @classmethod
    def validate_workers(cls, v: int, info) -> int:
        """Auto-scale workers in production."""
        import multiprocessing

        if info.data.get("ENVIRONMENT") == "production" and v == 1:
            # Default to CPU count in production
            return multiprocessing.cpu_count()
        return v

    # External URLs
    FRONTEND_URL: Optional[HttpUrl] = Field(
        default=None,
        description="Frontend application URL"
    )

    # Feature Flags
    FEATURE_REGISTRATION_ENABLED: bool = Field(
        default=True,
        description="Enable user registration"
    )
    FEATURE_PAYMENT_ENABLED: bool = Field(
        default=False,
        description="Enable payment processing"
    )
```

#### 4. Database Configuration
```python
# src/config/database.py
from pydantic import Field, PostgresDsn, field_validator, SecretStr
from typing import Optional

from .base import BaseConfig


class DatabaseConfig(BaseConfig):
    """Database configuration with connection pooling."""

    # Connection
    DATABASE_URL: PostgresDsn = Field(
        ...,  # Required field
        description="PostgreSQL connection URL"
    )

    # Alternative connection parameters (if URL not provided)
    DB_HOST: Optional[str] = Field(default=None)
    DB_PORT: Optional[int] = Field(default=5432)
    DB_USER: Optional[str] = Field(default=None)
    DB_PASSWORD: Optional[SecretStr] = Field(default=None)
    DB_NAME: Optional[str] = Field(default=None)

    # Connection Pool
    DB_POOL_SIZE: int = Field(
        default=20,
        ge=1,
        le=100,
        description="Connection pool size"
    )
    DB_MAX_OVERFLOW: int = Field(
        default=0,
        ge=0,
        le=50,
        description="Max overflow connections"
    )
    DB_POOL_TIMEOUT: int = Field(
        default=30,
        ge=1,
        description="Pool timeout in seconds"
    )
    DB_POOL_RECYCLE: int = Field(
        default=3600,
        ge=60,
        description="Recycle connections after N seconds"
    )

    # Options
    DB_ECHO: bool = Field(
        default=False,
        description="Echo SQL statements (debug)"
    )
    DB_ECHO_POOL: bool = Field(
        default=False,
        description="Echo pool events (debug)"
    )

    @field_validator("DB_ECHO", "DB_ECHO_POOL")
    @classmethod
    def validate_echo(cls, v: bool, info) -> bool:
        """Prevent SQL echo in production."""
        if info.data.get("ENVIRONMENT") == "production" and v:
            raise ValueError("Database echo cannot be enabled in production")
        return v

    def get_async_database_url(self) -> str:
        """Convert to async PostgreSQL URL."""
        if self.DATABASE_URL:
            return str(self.DATABASE_URL).replace(
                "postgresql://", "postgresql+asyncpg://"
            )
        # Build from components
        password = self.DB_PASSWORD.get_secret_value() if self.DB_PASSWORD else ""
        return (
            f"postgresql+asyncpg://{self.DB_USER}:{password}@"
            f"{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"
        )
```

#### 5. Authentication Configuration
```python
# src/config/auth.py
from datetime import timedelta
from pydantic import Field, SecretStr, field_validator

from .base import BaseConfig


class AuthConfig(BaseConfig):
    """Authentication and security configuration."""

    # JWT Settings
    JWT_SECRET_KEY: SecretStr = Field(
        ...,
        min_length=32,
        description="JWT secret key (min 32 chars)"
    )
    JWT_ALGORITHM: str = Field(
        default="HS256",
        description="JWT algorithm"
    )
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(
        default=30,
        ge=5,
        le=1440,  # Max 24 hours
        description="Access token expiration (minutes)"
    )
    JWT_REFRESH_TOKEN_EXPIRE_DAYS: int = Field(
        default=7,
        ge=1,
        le=30,
        description="Refresh token expiration (days)"
    )

    # Password Policy
    PASSWORD_MIN_LENGTH: int = Field(default=8, ge=8)
    PASSWORD_REQUIRE_UPPERCASE: bool = Field(default=True)
    PASSWORD_REQUIRE_LOWERCASE: bool = Field(default=True)
    PASSWORD_REQUIRE_DIGIT: bool = Field(default=True)
    PASSWORD_REQUIRE_SPECIAL: bool = Field(default=True)

    # OAuth2 (Optional)
    OAUTH2_ENABLED: bool = Field(default=False)
    GOOGLE_CLIENT_ID: Optional[str] = Field(default=None)
    GOOGLE_CLIENT_SECRET: Optional[SecretStr] = Field(default=None)
    GITHUB_CLIENT_ID: Optional[str] = Field(default=None)
    GITHUB_CLIENT_SECRET: Optional[SecretStr] = Field(default=None)

    # Security Headers
    SECURE_HEADERS_ENABLED: bool = Field(default=True)
    ALLOWED_HOSTS: List[str] = Field(
        default=["*"],
        description="Allowed host headers"
    )

    @field_validator("ALLOWED_HOSTS")
    @classmethod
    def validate_allowed_hosts(cls, v: List[str], info) -> List[str]:
        """Require specific hosts in production."""
        if info.data.get("ENVIRONMENT") == "production" and "*" in v:
            raise ValueError("Wildcard hosts not allowed in production")
        return v

    def get_access_token_expire_timedelta(self) -> timedelta:
        """Get access token expiration as timedelta."""
        return timedelta(minutes=self.JWT_ACCESS_TOKEN_EXPIRE_MINUTES)

    def get_refresh_token_expire_timedelta(self) -> timedelta:
        """Get refresh token expiration as timedelta."""
        return timedelta(days=self.JWT_REFRESH_TOKEN_EXPIRE_DAYS)
```

#### 6. Main Settings Aggregator
```python
# src/core/config.py
from functools import lru_cache
from typing import Optional

from src.config.app import AppConfig
from src.config.auth import AuthConfig
from src.config.database import DatabaseConfig
from src.config.redis import RedisConfig
from src.config.logging import LoggingConfig


class Settings(
    AppConfig,
    DatabaseConfig,
    RedisConfig,
    AuthConfig,
    LoggingConfig,
):
    """Main settings class combining all configuration domains."""

    @property
    def is_local_development(self) -> bool:
        """Check if running in local development."""
        return self.ENVIRONMENT in ("local", "development") and self.DEBUG

    def __init__(self, **data):
        """Initialize settings with validation."""
        super().__init__(**data)
        self._validate_configuration()

    def _validate_configuration(self) -> None:
        """Perform cross-field validation."""
        # Example: Ensure Redis is configured if caching is enabled
        if hasattr(self, "CACHE_ENABLED") and self.CACHE_ENABLED:
            if not self.REDIS_URL:
                raise ValueError("Redis URL required when caching is enabled")

        # Add more cross-field validations as needed


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()


# Convenience exports
settings = get_settings()
```

#### 7. Environment Files
```bash
# .env.example
# Application
ENVIRONMENT=local
DEBUG=true
APP_NAME="My Awesome API"

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/myapp
DB_POOL_SIZE=20

# Redis
REDIS_URL=redis://localhost:6379/0

# Authentication
JWT_SECRET_KEY=your-super-secret-key-min-32-characters-long

# Feature Flags
FEATURE_REGISTRATION_ENABLED=true
FEATURE_PAYMENT_ENABLED=false
```

```bash
# .env.test
ENVIRONMENT=testing
DATABASE_URL=postgresql://test:test@localhost:5432/test_db
JWT_SECRET_KEY=test-secret-key-for-testing-only-not-for-prod
```

## 📊 Logging System

### Core Principles
- **Structured Logging**: JSON format with rich context
- **Correlation IDs**: Trace requests across services
- **Environment-Aware**: Pretty console for dev, JSON for prod
- **Performance**: Async logging with minimal overhead
- **Observability**: OpenTelemetry integration ready

### Implementation

#### 1. Logging Configuration
```python
# src/config/logging.py
from enum import Enum
from typing import List, Optional
from pydantic import Field

from .base import BaseConfig


class LogLevel(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LoggingConfig(BaseConfig):
    """Logging configuration."""

    # Log Levels
    LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Application log level"
    )
    UVICORN_LOG_LEVEL: LogLevel = Field(
        default=LogLevel.INFO,
        description="Uvicorn server log level"
    )

    # Output Format
    LOG_FORMAT: str = Field(
        default="json",
        description="Log format (json|pretty)"
    )

    @field_validator("LOG_FORMAT")
    @classmethod
    def validate_log_format(cls, v: str, info) -> str:
        """Use JSON in production."""
        if info.data.get("ENVIRONMENT") == "production":
            return "json"
        return v

    # Sampling
    LOG_SAMPLING_RATE: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Log sampling rate (0.0-1.0)"
    )

    # OpenTelemetry
    OTEL_ENABLED: bool = Field(
        default=False,
        description="Enable OpenTelemetry integration"
    )
    OTEL_SERVICE_NAME: Optional[str] = Field(
        default=None,
        description="OpenTelemetry service name"
    )
    OTEL_EXPORTER_ENDPOINT: Optional[str] = Field(
        default=None,
        description="OpenTelemetry collector endpoint"
    )

    # Performance
    LOG_ASYNC: bool = Field(
        default=True,
        description="Use async logging"
    )
    LOG_QUEUE_SIZE: int = Field(
        default=1000,
        ge=100,
        description="Async log queue size"
    )
```

#### 2. Logging Setup
```python
# src/core/logging.py
import logging
import sys
import structlog
from contextvars import ContextVar
from typing import Any, Dict, Optional
from uuid import uuid4

from src.core.config import settings


# Context variables for request tracking
request_id_var: ContextVar[Optional[str]] = ContextVar("request_id", default=None)
user_id_var: ContextVar[Optional[str]] = ContextVar("user_id", default=None)


def setup_logging() -> None:
    """Configure structlog with proper processors and formatters."""

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, settings.LOG_LEVEL),
    )

    # Base processors - always included
    base_processors = [
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.contextvars.merge_contextvars,  # Add context variables
        add_app_context,  # Custom processor for app context
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
    ]

    # Environment-specific processors
    if settings.is_local_development:
        # Pretty console output for development
        processors = base_processors + [
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.RichTracebackFormatter(
                    show_locals=True,
                    max_frames=5,
                ),
            )
        ]
    else:
        # JSON output for production
        processors = base_processors + [
            structlog.processors.dict_tracebacks,
            structlog.processors.JSONRenderer(),
        ]

    # Configure structlog
    structlog.configure(
        processors=processors,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.AsyncBoundLogger if settings.LOG_ASYNC else structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def add_app_context(logger: Any, method_name: str, event_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Add application context to all log entries."""
    # Add basic app info
    event_dict["service"] = settings.APP_NAME
    event_dict["environment"] = settings.ENVIRONMENT
    event_dict["version"] = settings.APP_VERSION

    # Add request context if available
    request_id = request_id_var.get()
    if request_id:
        event_dict["request_id"] = request_id

    user_id = user_id_var.get()
    if user_id:
        event_dict["user_id"] = user_id

    # Add sampling decision
    if settings.LOG_SAMPLING_RATE < 1.0:
        import random
        if random.random() > settings.LOG_SAMPLING_RATE:
            raise structlog.DropEvent

    return event_dict


def get_logger(name: Optional[str] = None) -> structlog.stdlib.BoundLogger:
    """Get a configured logger instance."""
    return structlog.get_logger(name)


# Convenience logger for immediate use
logger = get_logger(__name__)
```

#### 3. Middleware for Request Context
```python
# src/middleware/logging.py
import time
from uuid import uuid4
from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from src.core.logging import logger, request_id_var, user_id_var
import structlog


class LoggingMiddleware(BaseHTTPMiddleware):
    """Middleware to add request context and log requests."""

    async def dispatch(self, request: Request, call_next):
        # Generate or extract request ID
        request_id = request.headers.get("X-Request-ID", str(uuid4()))

        # Set context variables
        request_id_var.set(request_id)

        # Clear context variables for this request
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(
            request_id=request_id,
            method=request.method,
            path=request.url.path,
            client_host=request.client.host if request.client else None,
        )

        # Log request
        start_time = time.time()
        logger.info("request_started")

        try:
            response = await call_next(request)

            # Calculate duration
            duration = time.time() - start_time

            # Log response
            logger.info(
                "request_completed",
                status_code=response.status_code,
                duration_ms=round(duration * 1000, 2),
            )

            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id

            return response

        except Exception as e:
            duration = time.time() - start_time
            logger.exception(
                "request_failed",
                duration_ms=round(duration * 1000, 2),
                exception_type=type(e).__name__,
            )
            raise
        finally:
            # Clear context
            request_id_var.set(None)
            structlog.contextvars.clear_contextvars()
```

#### 4. Usage Examples
```python
# src/api/endpoints/users.py
from fastapi import APIRouter, Depends, HTTPException
from src.core.logging import logger
from src.core.config import settings
import structlog

router = APIRouter()


@router.post("/users")
async def create_user(user_data: UserCreate):
    """Create a new user with proper logging."""
    # Get a bound logger with endpoint context
    log = logger.bind(
        endpoint="create_user",
        feature_flag_registration=settings.FEATURE_REGISTRATION_ENABLED,
    )

    if not settings.FEATURE_REGISTRATION_ENABLED:
        log.warning("registration_attempted_while_disabled")
        raise HTTPException(
            status_code=403,
            detail="Registration is currently disabled"
        )

    try:
        # Log with structured data
        log.info(
            "creating_user",
            email=user_data.email,
            # Don't log sensitive data!
            # password=user_data.password  # NEVER DO THIS
        )

        user = await create_user_in_db(user_data)

        # Success logging
        log.info(
            "user_created",
            user_id=user.id,
            email=user.email,
        )

        return user

    except IntegrityError as e:
        # Log specific errors with context
        log.warning(
            "user_creation_failed",
            reason="duplicate_email",
            email=user_data.email,
            error=str(e),
        )
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )
    except Exception as e:
        # Log unexpected errors
        log.exception(
            "user_creation_error",
            error_type=type(e).__name__,
        )
        raise


@router.get("/users/{user_id}")
async def get_user(user_id: int):
    """Get user with automatic request context."""
    # Context from middleware is automatically included
    logger.info("fetching_user", user_id=user_id)

    user = await fetch_user(user_id)
    if not user:
        logger.warning("user_not_found", user_id=user_id)
        raise HTTPException(status_code=404)

    return user
```

#### 5. Testing Utilities
```python
# src/core/logging_test.py
import structlog
from structlog.testing import LogCapture
import pytest


@pytest.fixture
def log_output():
    """Fixture to capture log output in tests."""
    cap = LogCapture()
    structlog.configure(processors=[cap])
    yield cap.entries
    structlog.reset_defaults()


def test_user_creation_logging(log_output):
    """Test that user creation logs correctly."""
    # Your test code here
    create_user(...)

    # Verify logs
    assert len(log_output) == 2
    assert log_output[0]["event"] == "creating_user"
    assert log_output[1]["event"] == "user_created"
    assert "password" not in log_output[0]  # Ensure no sensitive data
```

#### 6. OpenTelemetry Integration
```python
# src/core/telemetry.py
from opentelemetry import trace
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.sqlalchemy import SQLAlchemyInstrumentor
from opentelemetry.instrumentation.redis import RedisInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

from src.core.config import settings


def setup_telemetry(app):
    """Setup OpenTelemetry instrumentation."""
    if not settings.OTEL_ENABLED:
        return

    # Set up tracer
    trace.set_tracer_provider(TracerProvider())
    tracer = trace.get_tracer(__name__)

    # Set up exporter
    otlp_exporter = OTLPSpanExporter(
        endpoint=settings.OTEL_EXPORTER_ENDPOINT,
        insecure=True,  # Use False in production with proper TLS
    )

    # Add span processor
    span_processor = BatchSpanProcessor(otlp_exporter)
    trace.get_tracer_provider().add_span_processor(span_processor)

    # Instrument libraries
    FastAPIInstrumentor.instrument_app(app, tracer_provider=trace.get_tracer_provider())
    SQLAlchemyInstrumentor().instrument(enable_commenter=True)
    RedisInstrumentor().instrument()

    # Add trace ID to logs
    def add_trace_context(logger, method_name, event_dict):
        span = trace.get_current_span()
        if span and span.is_recording():
            ctx = span.get_span_context()
            event_dict["trace_id"] = format(ctx.trace_id, "032x")
            event_dict["span_id"] = format(ctx.span_id, "016x")
        return event_dict

    # Add processor to structlog
    structlog.configure(
        processors=[add_trace_context] + structlog.get_config()["processors"]
    )
```

## 🚀 Quick Start

### 1. Project Setup
```bash
# Install dependencies
uv pip install pydantic-settings structlog python-json-logger opentelemetry-distro

# Create environment file
cp .env.example .env

# Run with proper configuration
python -m src.cli serve
```

### 2. Docker Integration
```yaml
# docker-compose.yml
services:
  app:
    environment:
      - ENVIRONMENT=development
      - DATABASE_URL=postgresql://user:pass@db:5432/app
      - REDIS_URL=redis://redis:6379
      - JWT_SECRET_KEY_FILE=/run/secrets/jwt_secret
    secrets:
      - jwt_secret

secrets:
  jwt_secret:
    file: ./secrets/jwt_secret.txt
```

### 3. Production Deployment
```bash
# Production environment variables
export ENVIRONMENT=production
export DEBUG=false
export LOG_FORMAT=json
export LOG_LEVEL=INFO
export DATABASE_URL="${DATABASE_URL}"
export JWT_SECRET_KEY="${JWT_SECRET_KEY}"
export OTEL_ENABLED=true
export OTEL_SERVICE_NAME=my-api
export OTEL_EXPORTER_ENDPOINT=http://otel-collector:4317
```

## 📚 Best Practices

1. **Never log sensitive data** - No passwords, tokens, or PII in logs
2. **Use structured logging** - Always log with key-value pairs
3. **Include context** - Add relevant context to every log entry
4. **Fail fast** - Invalid configuration should crash at startup
5. **Environment-specific** - Different settings for dev/staging/prod
6. **Type everything** - Full type hints for IDE support
7. **Validate early** - Catch configuration errors before runtime
8. **Document settings** - Use Field descriptions for clarity

## 🔍 Debugging Tips

1. **Print configuration** (non-sensitive only):
   ```python
   print(settings.model_dump(exclude={"JWT_SECRET_KEY", "DB_PASSWORD"}))
   ```

2. **Check environment**:
   ```python
   logger.info("app_config", config=settings.dict(exclude_secrets=True))
   ```

3. **Test configuration**:
   ```python
   def test_production_config():
       with pytest.raises(ValueError):
           Settings(ENVIRONMENT="production", DEBUG=True)
   ```

This configuration and logging system ensures **consistency**, **safety**, and **observability** across all Python backend applications.
