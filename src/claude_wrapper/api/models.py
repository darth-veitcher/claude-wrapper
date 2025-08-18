"""OpenAI-compatible API models."""

import time
from typing import Any, Literal

from pydantic import BaseModel, Field


class Message(BaseModel):
    """Chat message."""

    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: str | None = None
    function_call: dict[str, Any] | None = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""

    model: str = Field(default="claude-3-opus-20240229")
    messages: list[Message]
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1)
    stream: bool | None = Field(default=False)
    stop: str | list[str] | None = None
    max_tokens: int | None = Field(default=None, ge=1)
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: dict[str, float] | None = None
    user: str | None = None

    # Additional fields for session management
    session_id: str | None = None
    system_prompt: str | None = None


class Usage(BaseModel):
    """Token usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """Completion choice."""

    index: int
    message: Message
    finish_reason: Literal["stop", "length", "function_call", "content_filter"] | None = None
    logprobs: Any | None = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage | None = None
    system_fingerprint: str | None = None


class Delta(BaseModel):
    """Streaming delta."""

    role: str | None = None
    content: str | None = None
    function_call: dict[str, Any] | None = None


class StreamChoice(BaseModel):
    """Streaming choice."""

    index: int
    delta: Delta
    finish_reason: Literal["stop", "length", "function_call", "content_filter"] | None = None
    logprobs: Any | None = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming response."""

    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[StreamChoice]
    system_fingerprint: str | None = None


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""

    model: str = Field(default="claude-3-opus-20240229")
    prompt: str | list[str]
    suffix: str | None = None
    max_tokens: int | None = Field(default=16, ge=1)
    temperature: float | None = Field(default=1.0, ge=0.0, le=2.0)
    top_p: float | None = Field(default=1.0, ge=0.0, le=1.0)
    n: int | None = Field(default=1, ge=1)
    stream: bool | None = Field(default=False)
    logprobs: int | None = None
    echo: bool | None = Field(default=False)
    stop: str | list[str] | None = None
    presence_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: float | None = Field(default=0.0, ge=-2.0, le=2.0)
    best_of: int | None = Field(default=1, ge=1)
    logit_bias: dict[str, float] | None = None
    user: str | None = None


class CompletionChoice(BaseModel):
    """Completion choice."""

    text: str
    index: int
    logprobs: Any | None = None
    finish_reason: Literal["stop", "length"] | None = None


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""

    id: str = Field(default_factory=lambda: f"cmpl-{int(time.time() * 1000)}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage | None = None


class Model(BaseModel):
    """Model information."""

    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    permission: list[Any] = []
    root: str
    parent: str | None = None


class ModelList(BaseModel):
    """List of available models."""

    object: Literal["list"] = "list"
    data: list[Model]


class ErrorResponse(BaseModel):
    """Error response."""

    error: dict[str, Any]


class ErrorDetail(BaseModel):
    """Error detail."""

    message: str
    type: str
    param: str | None = None
    code: str | None = None
