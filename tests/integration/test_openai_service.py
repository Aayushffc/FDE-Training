from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx2
import openai
import pytest

from app.core.config import Settings
from app.core.exceptions import LLMProviderError
from app.infrastructure.llm_providers import openai as openai_service
from app.modules.chat.schemas import ChatRequest
from app.modules.prompts.service import SYSTEM_PROMPT
from tests.conftest import override_settings

_DUMMY_REQ = httpx2.Request("POST", "https://api.openai.com/v1/chat/completions")


class TestOpenAIService:
    async def test_passes_model_messages_system_prompt_and_extracts_usage(
        self,
        mock_settings: Settings,
    ) -> None:
        fake_response = SimpleNamespace(
            model="gpt-6-luna",
            choices=[SimpleNamespace(message=SimpleNamespace(content="OpenAI answer"))],
            usage=SimpleNamespace(prompt_tokens=42, completion_tokens=18),
        )
        with patch("app.infrastructure.llm_providers.openai.AsyncOpenAI") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.return_value = fake_response

            result = await openai_service.generate_chat_completion(
                ChatRequest(message="What is FastAPI?", provider="openai")
            )

        call_kwargs = inst.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == mock_settings.default_openai_model
        assert call_kwargs["messages"] == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "What is FastAPI?"},
        ]
        assert result.provider == "openai"
        assert result.model == "gpt-6-luna"
        assert result.reply == "OpenAI answer"
        assert result.input_tokens == 42
        assert result.output_tokens == 18
        assert result.latency_ms >= 0
        assert result.estimated_cost_usd == pytest.approx(0.0000132)

    async def test_missing_api_key_raises_503(
        self,
        mock_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        override_settings(monkeypatch, mock_settings, openai_api_key=None)
        with pytest.raises(LLMProviderError) as exc_info:
            await openai_service.generate_chat_completion(ChatRequest(message="Hi", provider="openai"))

        assert exc_info.value.status_code == 503
        assert exc_info.value.provider == "openai"

    async def test_empty_response_raises_502(self, mock_settings: Settings) -> None:
        fake_response = SimpleNamespace(model="gpt-6-luna", choices=[], usage=None)
        with patch("app.infrastructure.llm_providers.openai.AsyncOpenAI") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.return_value = fake_response

            with pytest.raises(LLMProviderError) as exc_info:
                await openai_service.generate_chat_completion(ChatRequest(message="Hi", provider="openai"))

        assert exc_info.value.status_code == 502

    async def test_rate_limit_error_mapped_to_429(self, mock_settings: Settings) -> None:
        exc = openai.RateLimitError(
            "Rate limit reached",
            response=httpx2.Response(429, request=_DUMMY_REQ),
            body=None,
        )
        with patch("app.infrastructure.llm_providers.openai.AsyncOpenAI") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await openai_service.generate_chat_completion(ChatRequest(message="Hi", provider="openai"))

        assert exc_info.value.status_code == 429
        assert exc_info.value.provider == "openai"

    async def test_authentication_error_mapped_to_401(self, mock_settings: Settings) -> None:
        exc = openai.AuthenticationError(
            "Incorrect API key provided: sk-secret12345",
            response=httpx2.Response(401, request=_DUMMY_REQ),
            body=None,
        )
        with patch("app.infrastructure.llm_providers.openai.AsyncOpenAI") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await openai_service.generate_chat_completion(ChatRequest(message="Hi", provider="openai"))

        assert exc_info.value.status_code == 401
        assert "sk-secret12345" not in exc_info.value.message

    async def test_timeout_error_mapped_to_504(self, mock_settings: Settings) -> None:
        exc = openai.APITimeoutError(request=_DUMMY_REQ)
        with patch("app.infrastructure.llm_providers.openai.AsyncOpenAI") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await openai_service.generate_chat_completion(ChatRequest(message="Hi", provider="openai"))

        assert exc_info.value.status_code == 504

    async def test_unexpected_sdk_error_does_not_leak_raw_secret(self, mock_settings: Settings) -> None:
        exc = openai.InternalServerError(
            "Upstream failure while processing sk-live-super-secret-999",
            response=httpx2.Response(500, request=_DUMMY_REQ),
            body=None,
        )
        with patch("app.infrastructure.llm_providers.openai.AsyncOpenAI") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await openai_service.generate_chat_completion(ChatRequest(message="Hi", provider="openai"))

        assert exc_info.value.status_code == 502
        assert "sk-live-super-secret-999" not in exc_info.value.message
        assert "[REDACTED]" in exc_info.value.message
