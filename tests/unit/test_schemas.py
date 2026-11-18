import pytest
from pydantic import ValidationError

from app.api.v1.endpoints.health import HealthResponse
from app.modules.chat.schemas import MAX_MESSAGE_LENGTH, ChatRequest, ChatResponse


class TestChatRequestSchema:
    def test_valid_openai_request_accepted(self) -> None:
        req = ChatRequest(message="Hello OpenAI", provider="openai")
        assert req.message == "Hello OpenAI"
        assert req.provider == "openai"

    def test_valid_anthropic_request_accepted(self) -> None:
        req = ChatRequest(message="Hello Anthropic", provider="anthropic")
        assert req.message == "Hello Anthropic"
        assert req.provider == "anthropic"

    def test_valid_groq_request_accepted(self) -> None:
        req = ChatRequest(message="Hello Groq", provider="groq")
        assert req.message == "Hello Groq"
        assert req.provider == "groq"

    def test_provider_omitted_defaults_to_openai(self) -> None:
        req = ChatRequest(message="Default provider check")
        assert req.provider == "openai"

    def test_empty_message_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(message="", provider="openai")

    def test_unsupported_provider_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatRequest(message="Hello", provider="mistral")  # type: ignore[arg-type]

    def test_message_exceeds_configured_limit_rejected(self) -> None:
        oversized_message = "a" * (MAX_MESSAGE_LENGTH + 1)
        with pytest.raises(ValidationError):
            ChatRequest(message=oversized_message, provider="openai")

    def test_message_at_exact_limit_accepted(self) -> None:
        max_message = "a" * MAX_MESSAGE_LENGTH
        req = ChatRequest(message=max_message, provider="groq")
        assert len(req.message) == MAX_MESSAGE_LENGTH

    def test_prompt_version_optional_accepted(self) -> None:
        req = ChatRequest(message="Test", prompt_version="v2")
        assert req.prompt_version == "v2"
        req_none = ChatRequest(message="Test")
        assert req_none.prompt_version is None


class TestChatResponseSchema:
    def test_valid_response_object_accepted(self) -> None:
        res = ChatResponse(
            provider="openai",
            model="gpt-6-luna",
            reply="Hi there!",
            input_tokens=15,
            output_tokens=10,
            latency_ms=250.5,
            estimated_cost_usd=0.0000065,
            prompt_version="customer_support/v2",
        )
        assert res.provider == "openai"
        assert res.input_tokens == 15
        assert res.output_tokens == 10
        assert res.prompt_version == "customer_support/v2"

    def test_negative_input_token_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatResponse(
                provider="openai",
                model="gpt-6-luna",
                reply="Hi",
                input_tokens=-1,
                output_tokens=10,
                latency_ms=100.0,
                estimated_cost_usd=0.0,
            )

    def test_negative_output_token_count_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatResponse(
                provider="anthropic",
                model="claude-haiku-5-5",
                reply="Hi",
                input_tokens=10,
                output_tokens=-5,
                latency_ms=100.0,
                estimated_cost_usd=0.0,
            )

    def test_negative_latency_or_cost_rejected(self) -> None:
        with pytest.raises(ValidationError):
            ChatResponse(
                provider="groq",
                model="openai/gpt-oss-120b",
                reply="Hi",
                input_tokens=10,
                output_tokens=5,
                latency_ms=-10.0,
                estimated_cost_usd=0.0,
            )


class TestHealthResponseSchema:
    def test_valid_health_response(self) -> None:
        health = HealthResponse()
        assert health.status == "ok"
