from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import groq
import httpx
import pytest

from app.core.config import Settings
from app.core.exceptions import LLMProviderError
from app.infrastructure.llm_providers import groq as groq_service
from app.modules.chat.schemas import ChatRequest
from app.modules.prompts.service import SYSTEM_PROMPT
from tests.conftest import override_settings

_DUMMY_REQ = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")


class TestGroqService:
    async def test_passes_model_messages_system_prompt_and_extracts_usage(
        self,
        mock_settings: Settings,
    ) -> None:
        fake_response = SimpleNamespace(
            model="openai/gpt-oss-120b",
            choices=[SimpleNamespace(message=SimpleNamespace(content="Groq answer"))],
            usage=SimpleNamespace(prompt_tokens=95, completion_tokens=40),
        )
        with patch("app.infrastructure.llm_providers.groq.AsyncGroq") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.return_value = fake_response

            result = await groq_service.generate_chat_completion(
                ChatRequest(message="Explain LPU inference.", provider="groq")
            )

        call_kwargs = inst.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == mock_settings.default_groq_model
        assert call_kwargs["messages"] == [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "Explain LPU inference."},
        ]
        assert result.provider == "groq"
        assert result.model == "openai/gpt-oss-120b"
        assert result.reply == "Groq answer"
        assert result.input_tokens == 95
        assert result.output_tokens == 40
        assert result.latency_ms >= 0
        assert result.estimated_cost_usd == pytest.approx(0.00003825)

    async def test_missing_api_key_raises_503(
        self,
        mock_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        override_settings(monkeypatch, mock_settings, groq_api_key=None)
        with pytest.raises(LLMProviderError) as exc_info:
            await groq_service.generate_chat_completion(ChatRequest(message="Hi", provider="groq"))

        assert exc_info.value.status_code == 503
        assert exc_info.value.provider == "groq"

    async def test_empty_response_raises_502(self, mock_settings: Settings) -> None:
        fake_response = SimpleNamespace(model="openai/gpt-oss-120b", choices=[], usage=None)
        with patch("app.infrastructure.llm_providers.groq.AsyncGroq") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.return_value = fake_response

            with pytest.raises(LLMProviderError) as exc_info:
                await groq_service.generate_chat_completion(ChatRequest(message="Hi", provider="groq"))

        assert exc_info.value.status_code == 502

    async def test_rate_limit_error_mapped_to_429(self, mock_settings: Settings) -> None:
        exc = groq.RateLimitError(
            "Rate limit reached",
            response=httpx.Response(429, request=_DUMMY_REQ),
            body=None,
        )
        with patch("app.infrastructure.llm_providers.groq.AsyncGroq") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await groq_service.generate_chat_completion(ChatRequest(message="Hi", provider="groq"))

        assert exc_info.value.status_code == 429
        assert exc_info.value.provider == "groq"

    async def test_authentication_error_mapped_to_401(self, mock_settings: Settings) -> None:
        exc = groq.AuthenticationError(
            "Invalid API key: gsk_secret12345",
            response=httpx.Response(401, request=_DUMMY_REQ),
            body=None,
        )
        with patch("app.infrastructure.llm_providers.groq.AsyncGroq") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await groq_service.generate_chat_completion(ChatRequest(message="Hi", provider="groq"))

        assert exc_info.value.status_code == 401
        assert "gsk_secret12345" not in exc_info.value.message

    async def test_timeout_error_mapped_to_504(self, mock_settings: Settings) -> None:
        exc = groq.APITimeoutError(request=_DUMMY_REQ)
        with patch("app.infrastructure.llm_providers.groq.AsyncGroq") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await groq_service.generate_chat_completion(ChatRequest(message="Hi", provider="groq"))

        assert exc_info.value.status_code == 504

    async def test_unexpected_sdk_error_does_not_leak_raw_secret(self, mock_settings: Settings) -> None:
        exc = groq.InternalServerError(
            "Fatal failure with key gsk_live_super_secret_999",
            response=httpx.Response(500, request=_DUMMY_REQ),
            body=None,
        )
        with patch("app.infrastructure.llm_providers.groq.AsyncGroq") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await groq_service.generate_chat_completion(ChatRequest(message="Hi", provider="groq"))

        assert exc_info.value.status_code == 502
        assert "gsk_live_super_secret_999" not in exc_info.value.message
        assert "[REDACTED]" in exc_info.value.message
