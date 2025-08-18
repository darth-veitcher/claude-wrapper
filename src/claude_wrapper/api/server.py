"""FastAPI server with OpenAI-compatible endpoints."""

import time
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse

from claude_wrapper import get_version
from claude_wrapper.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Message,
    Model,
    ModelList,
    Usage,
)
from claude_wrapper.core import ClaudeClient
from claude_wrapper.core.exceptions import ClaudeWrapperError
from claude_wrapper.utils.config import get_config
from claude_wrapper.utils.streaming import StreamProcessor

# Global instances
config = get_config()
claude_client = ClaudeClient(
    claude_path=config.claude_path,
    timeout=config.timeout,
    retry_attempts=config.retry_attempts,
)
stream_processor = StreamProcessor()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncGenerator[None, None]:
    """Application lifespan manager."""
    # Startup
    print("Starting Claude Wrapper API server...")
    try:
        # Verify Claude CLI is available
        await claude_client.check_auth()
        print("✓ Claude CLI authenticated")
    except Exception as e:
        print(f"⚠ Claude CLI not authenticated: {e}")

    yield

    # Shutdown
    print("Shutting down Claude Wrapper API server...")


app = FastAPI(
    title="Claude Wrapper API",
    description="OpenAI-compatible API wrapper for Claude CLI",
    version=get_version(),
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


async def verify_api_key(authorization: str | None = Header(None)) -> bool:
    """Verify API key if configured."""
    if not config.api_key:
        return True  # No API key required

    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header required")

    # Extract token from "Bearer <token>" format
    token = authorization[7:] if authorization.startswith("Bearer ") else authorization

    if token != config.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return True


@app.get("/")
async def root() -> dict[str, Any]:
    """Root endpoint."""
    return {
        "message": "Claude Wrapper API",
        "version": get_version(),
        "endpoints": {
            "chat": "/v1/chat/completions",
            "completions": "/v1/completions",
            "models": "/v1/models",
        },
    }


@app.get("/health", response_model=None)
async def health() -> dict[str, Any] | JSONResponse:
    """Health check endpoint."""
    try:
        authenticated = await claude_client.check_auth()
        return {
            "status": "healthy",
            "claude_cli": "authenticated" if authenticated else "not authenticated",
        }
    except Exception as e:
        return JSONResponse(status_code=503, content={"status": "unhealthy", "error": str(e)})


@app.get("/v1/models", dependencies=[Depends(verify_api_key)])
async def list_models() -> ModelList:
    """List available models."""
    models = [
        Model(
            id="opus",
            created=int(time.time()),
            owned_by="anthropic",
            root="opus",
        ),
        Model(
            id="sonnet",
            created=int(time.time()),
            owned_by="anthropic",
            root="sonnet",
        ),
        Model(
            id="haiku",
            created=int(time.time()),
            owned_by="anthropic",
            root="haiku",
        ),
    ]

    return ModelList(data=models)


@app.post("/v1/chat/completions", dependencies=[Depends(verify_api_key)], response_model=None)
async def chat_completions(
    request: ChatCompletionRequest,
) -> ChatCompletionResponse | EventSourceResponse:
    """OpenAI-compatible chat completions endpoint."""

    try:
        # Extract messages
        system_prompt = None
        user_message = ""

        for msg in request.messages:
            if msg.role == "system":
                system_prompt = msg.content
            elif msg.role == "user":
                user_message = msg.content
            # For multi-turn, we'd need to handle this better

        if not user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # Handle streaming
        if request.stream:
            return EventSourceResponse(
                stream_chat_response(
                    user_message,
                    request.max_tokens,
                    request.temperature,
                    system_prompt,
                    request.model,
                )
            )

        # Non-streaming response
        response_text = await claude_client.chat(message=user_message)

        # Count tokens (simplified)
        prompt_tokens = sum(len(msg.content.split()) for msg in request.messages)
        completion_tokens = len(response_text.split())

        return ChatCompletionResponse(
            model=request.model,
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content=response_text),
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except ClaudeWrapperError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


async def stream_chat_response(
    message: str,
    _max_tokens: int | None,
    _temperature: float | None,
    _system_prompt: str | None,
    model: str,
) -> AsyncGenerator[str, None]:
    """Stream chat response as Server-Sent Events."""

    try:
        # Start streaming from Claude
        stream = claude_client.stream_chat(message=message)

        # Send initial chunk with role
        initial_chunk = stream_processor.create_openai_stream_chunk(
            role="assistant",
            model=model,
        )
        yield stream_processor.format_sse(initial_chunk)

        # Stream content chunks
        async for chunk in stream:
            content_chunk = stream_processor.create_openai_stream_chunk(
                content=chunk,
                model=model,
            )
            yield stream_processor.format_sse(content_chunk)

        # Send final chunk with finish reason
        final_chunk = stream_processor.create_openai_stream_chunk(
            finish_reason="stop",
            model=model,
        )
        yield stream_processor.format_sse(final_chunk)

        # Send [DONE] marker
        yield stream_processor.format_sse("[DONE]")

    except Exception as e:
        error_chunk = {
            "error": {
                "message": str(e),
                "type": "streaming_error",
            }
        }
        yield stream_processor.format_sse(error_chunk)


@app.post("/v1/completions", dependencies=[Depends(verify_api_key)])
async def completions(request: CompletionRequest) -> CompletionResponse:
    """OpenAI-compatible completions endpoint."""

    try:
        # Handle prompt as string or list
        prompt = "\n".join(request.prompt) if isinstance(request.prompt, list) else request.prompt

        # Get completion
        response_text = await claude_client.complete(
            prompt=prompt,
            _max_tokens=request.max_tokens,
            _temperature=request.temperature,
            _stop_sequences=request.stop
            if isinstance(request.stop, list)
            else [request.stop]
            if request.stop
            else None,
        )

        # Count tokens (simplified)
        prompt_tokens = len(prompt.split())
        completion_tokens = len(response_text.split())

        return CompletionResponse(
            model=request.model,
            choices=[
                CompletionChoice(
                    text=response_text,
                    index=0,
                    finish_reason="stop",
                )
            ],
            usage=Usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
            ),
        )

    except ClaudeWrapperError as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e


@app.exception_handler(ClaudeWrapperError)
async def claude_wrapper_exception_handler(
    _request: Request, exc: ClaudeWrapperError
) -> JSONResponse:
    """Handle Claude Wrapper exceptions."""
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "message": exc.message,
                "type": "claude_wrapper_error",
                "details": None,
            }
        },
    )


def main() -> None:
    """Main entry point for the API server."""
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
