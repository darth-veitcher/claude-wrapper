"""Tests for Pydantic models."""

import json
import time

import pytest
from pydantic import ValidationError

from claude_wrapper.api.models import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionStreamResponse,
    Choice,
    CompletionChoice,
    CompletionRequest,
    CompletionResponse,
    Delta,
    ErrorDetail,
    ErrorResponse,
    Message,
    Model,
    ModelList,
    StreamChoice,
    Usage,
)


class TestModels:
    """Test suite for API models."""

    @pytest.mark.unit
    def test_message_model(self):
        """Test Message model."""
        # Valid message
        msg = Message(role="user", content="Hello")
        assert msg.role == "user"
        assert msg.content == "Hello"
        assert msg.name is None
        assert msg.function_call is None

        # Message with all fields
        msg = Message(
            role="assistant",
            content="Response",
            name="bot",
            function_call={"name": "test", "arguments": "{}"},
        )
        assert msg.name == "bot"
        assert msg.function_call["name"] == "test"

        # Invalid role should fail
        with pytest.raises(ValidationError):
            Message(role="invalid", content="test")

    @pytest.mark.unit
    def test_chat_completion_request(self):
        """Test ChatCompletionRequest model."""
        request = ChatCompletionRequest(messages=[Message(role="user", content="Hello")])

        # Check defaults
        assert request.model == "claude-3-opus-20240229"
        assert request.temperature == 1.0
        assert request.stream is False
        assert request.max_tokens is None

        # With all parameters
        request = ChatCompletionRequest(
            model="claude-3-sonnet",
            messages=[
                Message(role="system", content="Be helpful"),
                Message(role="user", content="Hi"),
            ],
            temperature=0.5,
            max_tokens=100,
            stream=True,
            stop=["END"],
            session_id="test-123",
            system_prompt="Override system",
        )

        assert request.model == "claude-3-sonnet"
        assert len(request.messages) == 2
        assert request.temperature == 0.5
        assert request.max_tokens == 100
        assert request.stream is True
        assert request.stop == ["END"]
        assert request.session_id == "test-123"
        assert request.system_prompt == "Override system"

    @pytest.mark.unit
    def test_temperature_validation(self):
        """Test temperature parameter validation."""
        # Valid temperatures
        for temp in [0.0, 0.5, 1.0, 1.5, 2.0]:
            request = ChatCompletionRequest(
                messages=[Message(role="user", content="test")], temperature=temp
            )
            assert request.temperature == temp

        # Invalid temperatures
        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=[Message(role="user", content="test")], temperature=-0.1)

        with pytest.raises(ValidationError):
            ChatCompletionRequest(messages=[Message(role="user", content="test")], temperature=2.1)

    @pytest.mark.unit
    def test_usage_model(self):
        """Test Usage model."""
        usage = Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30)

        assert usage.prompt_tokens == 10
        assert usage.completion_tokens == 20
        assert usage.total_tokens == 30

        # Serialize to dict
        usage_dict = usage.model_dump()
        assert usage_dict["prompt_tokens"] == 10
        assert usage_dict["completion_tokens"] == 20
        assert usage_dict["total_tokens"] == 30

    @pytest.mark.unit
    def test_choice_model(self):
        """Test Choice model."""
        choice = Choice(
            index=0, message=Message(role="assistant", content="Response"), finish_reason="stop"
        )

        assert choice.index == 0
        assert choice.message.role == "assistant"
        assert choice.message.content == "Response"
        assert choice.finish_reason == "stop"
        assert choice.logprobs is None

    @pytest.mark.unit
    def test_chat_completion_response(self):
        """Test ChatCompletionResponse model."""
        response = ChatCompletionResponse(
            model="claude-3-opus-20240229",
            choices=[
                Choice(
                    index=0,
                    message=Message(role="assistant", content="Hello!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=10, total_tokens=15),
        )

        assert response.model == "claude-3-opus-20240229"
        assert response.object == "chat.completion"
        assert len(response.choices) == 1
        assert response.choices[0].message.content == "Hello!"
        assert response.usage.total_tokens == 15

        # Check auto-generated fields
        assert response.id.startswith("chatcmpl-")
        assert isinstance(response.created, int)
        assert response.system_fingerprint is None

    @pytest.mark.unit
    def test_delta_model(self):
        """Test Delta model for streaming."""
        # Delta with content
        delta = Delta(content="chunk")
        assert delta.content == "chunk"
        assert delta.role is None

        # Delta with role
        delta = Delta(role="assistant")
        assert delta.role == "assistant"
        assert delta.content is None

        # Empty delta
        delta = Delta()
        assert delta.role is None
        assert delta.content is None

    @pytest.mark.unit
    def test_stream_choice_model(self):
        """Test StreamChoice model."""
        choice = StreamChoice(index=0, delta=Delta(content="streaming chunk"), finish_reason=None)

        assert choice.index == 0
        assert choice.delta.content == "streaming chunk"
        assert choice.finish_reason is None

        # With finish reason
        choice = StreamChoice(index=0, delta=Delta(), finish_reason="stop")
        assert choice.finish_reason == "stop"

    @pytest.mark.unit
    def test_chat_completion_stream_response(self):
        """Test ChatCompletionStreamResponse model."""
        response = ChatCompletionStreamResponse(
            model="claude-3-opus-20240229",
            choices=[StreamChoice(index=0, delta=Delta(content="chunk"), finish_reason=None)],
        )

        assert response.model == "claude-3-opus-20240229"
        assert response.object == "chat.completion.chunk"
        assert response.choices[0].delta.content == "chunk"
        assert response.id.startswith("chatcmpl-")

    @pytest.mark.unit
    def test_completion_request(self):
        """Test CompletionRequest model."""
        # Simple request
        request = CompletionRequest(prompt="Complete this: ")

        assert request.prompt == "Complete this: "
        assert request.model == "claude-3-opus-20240229"
        assert request.max_tokens == 16
        assert request.temperature == 1.0

        # With list prompt
        request = CompletionRequest(prompt=["First", "Second"], max_tokens=100)

        assert request.prompt == ["First", "Second"]
        assert request.max_tokens == 100

    @pytest.mark.unit
    def test_completion_response(self):
        """Test CompletionResponse model."""
        response = CompletionResponse(
            model="claude-3-opus-20240229",
            choices=[CompletionChoice(text="Completed text", index=0, finish_reason="stop")],
            usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
        )

        assert response.model == "claude-3-opus-20240229"
        assert response.object == "text_completion"
        assert response.choices[0].text == "Completed text"
        assert response.id.startswith("cmpl-")

    @pytest.mark.unit
    def test_model_info(self):
        """Test Model info."""
        model = Model(
            id="claude-3-opus-20240229",
            created=int(time.time()),
            owned_by="anthropic",
            root="claude-3-opus-20240229",
        )

        assert model.id == "claude-3-opus-20240229"
        assert model.object == "model"
        assert model.owned_by == "anthropic"
        assert model.permission == []
        assert model.parent is None

    @pytest.mark.unit
    def test_model_list(self):
        """Test ModelList."""
        models = [
            Model(id="model-1", created=int(time.time()), owned_by="anthropic", root="model-1"),
            Model(id="model-2", created=int(time.time()), owned_by="anthropic", root="model-2"),
        ]

        model_list = ModelList(data=models)

        assert model_list.object == "list"
        assert len(model_list.data) == 2
        assert model_list.data[0].id == "model-1"
        assert model_list.data[1].id == "model-2"

    @pytest.mark.unit
    def test_error_models(self):
        """Test error response models."""
        error = ErrorDetail(
            message="Test error", type="test_error", param="test_param", code="TEST_001"
        )

        assert error.message == "Test error"
        assert error.type == "test_error"
        assert error.param == "test_param"
        assert error.code == "TEST_001"

        error_response = ErrorResponse(
            error={"message": "Error occurred", "type": "api_error", "code": "500"}
        )

        assert error_response.error["message"] == "Error occurred"

    @pytest.mark.unit
    def test_model_serialization(self):
        """Test model serialization to JSON."""
        request = ChatCompletionRequest(
            messages=[Message(role="user", content="Test")], temperature=0.7, max_tokens=100
        )

        # Serialize to JSON
        json_str = request.model_dump_json()
        data = json.loads(json_str)

        assert data["model"] == "claude-3-opus-20240229"
        assert data["messages"][0]["role"] == "user"
        assert data["messages"][0]["content"] == "Test"
        assert data["temperature"] == 0.7
        assert data["max_tokens"] == 100

        # Deserialize back
        request2 = ChatCompletionRequest.model_validate(data)
        assert request2.messages[0].content == "Test"
        assert request2.temperature == 0.7

    @pytest.mark.unit
    def test_finish_reason_validation(self):
        """Test finish_reason enum validation."""
        valid_reasons = ["stop", "length", "function_call", "content_filter"]

        for reason in valid_reasons:
            choice = Choice(
                index=0, message=Message(role="assistant", content="test"), finish_reason=reason
            )
            assert choice.finish_reason == reason

        # Invalid reason should fail
        with pytest.raises(ValidationError):
            Choice(
                index=0,
                message=Message(role="assistant", content="test"),
                finish_reason="invalid_reason",
            )
