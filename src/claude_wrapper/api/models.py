"""OpenAI-compatible API models."""

from typing import Optional, List, Dict, Any, Union, Literal
from datetime import datetime
from pydantic import BaseModel, Field
import time


class Message(BaseModel):
    """Chat message."""
    role: Literal["system", "user", "assistant", "function"]
    content: str
    name: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class ChatCompletionRequest(BaseModel):
    """OpenAI-compatible chat completion request."""
    model: str = Field(default="claude-3-opus-20240229")
    messages: List[Message]
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None
    max_tokens: Optional[int] = Field(default=None, ge=1)
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None
    
    # Additional fields for session management
    session_id: Optional[str] = None
    system_prompt: Optional[str] = None


class Usage(BaseModel):
    """Token usage information."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class Choice(BaseModel):
    """Completion choice."""
    index: int
    message: Message
    finish_reason: Optional[Literal["stop", "length", "function_call", "content_filter"]] = None
    logprobs: Optional[Any] = None


class ChatCompletionResponse(BaseModel):
    """OpenAI-compatible chat completion response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[Choice]
    usage: Optional[Usage] = None
    system_fingerprint: Optional[str] = None


class Delta(BaseModel):
    """Streaming delta."""
    role: Optional[str] = None
    content: Optional[str] = None
    function_call: Optional[Dict[str, Any]] = None


class StreamChoice(BaseModel):
    """Streaming choice."""
    index: int
    delta: Delta
    finish_reason: Optional[Literal["stop", "length", "function_call", "content_filter"]] = None
    logprobs: Optional[Any] = None


class ChatCompletionStreamResponse(BaseModel):
    """OpenAI-compatible streaming response."""
    id: str = Field(default_factory=lambda: f"chatcmpl-{int(time.time() * 1000)}")
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[StreamChoice]
    system_fingerprint: Optional[str] = None


class CompletionRequest(BaseModel):
    """OpenAI-compatible completion request."""
    model: str = Field(default="claude-3-opus-20240229")
    prompt: Union[str, List[str]]
    suffix: Optional[str] = None
    max_tokens: Optional[int] = Field(default=16, ge=1)
    temperature: Optional[float] = Field(default=1.0, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(default=1.0, ge=0.0, le=1.0)
    n: Optional[int] = Field(default=1, ge=1)
    stream: Optional[bool] = Field(default=False)
    logprobs: Optional[int] = None
    echo: Optional[bool] = Field(default=False)
    stop: Optional[Union[str, List[str]]] = None
    presence_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    frequency_penalty: Optional[float] = Field(default=0.0, ge=-2.0, le=2.0)
    best_of: Optional[int] = Field(default=1, ge=1)
    logit_bias: Optional[Dict[str, float]] = None
    user: Optional[str] = None


class CompletionChoice(BaseModel):
    """Completion choice."""
    text: str
    index: int
    logprobs: Optional[Any] = None
    finish_reason: Optional[Literal["stop", "length"]] = None


class CompletionResponse(BaseModel):
    """OpenAI-compatible completion response."""
    id: str = Field(default_factory=lambda: f"cmpl-{int(time.time() * 1000)}")
    object: Literal["text_completion"] = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[CompletionChoice]
    usage: Optional[Usage] = None


class Model(BaseModel):
    """Model information."""
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: str
    permission: List[Any] = []
    root: str
    parent: Optional[str] = None


class ModelList(BaseModel):
    """List of available models."""
    object: Literal["list"] = "list"
    data: List[Model]


class ErrorResponse(BaseModel):
    """Error response."""
    error: Dict[str, Any]


class ErrorDetail(BaseModel):
    """Error detail."""
    message: str
    type: str
    param: Optional[str] = None
    code: Optional[str] = None