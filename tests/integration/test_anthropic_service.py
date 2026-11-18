from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import anthropic
import httpx2
import pytest

from app.core.config import Settings
from app.core.exceptions import LLMProviderError
from app.infrastructure.llm_providers import anthropic as anthropic_service
from app.modules.chat.schemas import ChatRequest
from app.modules.prompts.service import SYSTEM_PROMPT
from tests.conftest import override_settings

_DUMMY_REQ = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


class TestAnthropicService:
    async def test_passes_model_messages_system_prompt_and_extracts_usage(
        self,
        mock_settings: Settings,
    ) -> None:
        fake_response = SimpleNamespace(
            model="claude-haiku-5-5",
            content=[
                SimpleNamespace(type="text", text="Line one."),
                SimpleNamespace(type="tool_use", text="ignored"),
                SimpleNamespace(type="text", text="Line two."),
            ],
            usage=SimpleNamespace(input_tokens=50, output_tokens=20),
        )
        with patch("app.infrastructure.llm_providers.anthropic.AsyncAnthropic") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.messages.create.return_value = fake_response

            result = await anthropic_service.generate_chat_completion(
                ChatRequest(message="Explain ASGI.", provider="anthropic")
            )

        call_kwargs = inst.messages.create.call_args.kwargs
        assert call_kwargs["model"] == mock_settings.default_anthropic_model
        assert call_kwargs["system"] == SYSTEM_PROMPT
        assert call_kwargs["messages"] == [{"role": "user", "content": "Explain ASGI."}]
        assert result.provider == "anthropic"
        assert result.model == "claude-haiku-5-5"
        assert result.reply == "Line one.\nLine two."
        assert result.input_tokens == 50
        assert result.output_tokens == 20
        assert result.latency_ms >= 0
        assert result.estimated_cost_usd == pytest.approx(0.000015)

    async def test_missing_api_key_raises_503(
        self,
        mock_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        override_settings(monkeypatch, mock_settings, anthropic_api_key=None)
        with pytest.raises(LLMProviderError) as exc_info:
            await anthropic_service.generate_chat_completion(ChatRequest(message="Hi", provider="anthropic"))

        assert exc_info.value.status_code == 503
        assert exc_info.value.provider == "anthropic"

    async def test_empty_text_blocks_raises_502(self, mock_settings: Settings) -> None:
        fake_response = SimpleNamespace(
            model="claude-haiku-5-5",
            content=[SimpleNamespace(type="tool_use", text="only tool")],
            usage=None,
        )
        with patch("app.infrastructure.llm_providers.anthropic.AsyncAnthropic") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.messages.create.return_value = fake_response

            with pytest.raises(LLMProviderError) as exc_info:
                await anthropic_service.generate_chat_completion(ChatRequest(message="Hi", provider="anthropic"))

        assert exc_info.value.status_code == 502

    async def test_rate_limit_error_mapped_to_429(self, mock_settings: Settings) -> None:
        exc = anthropic.RateLimitError(
            "Rate limit reached",
            response=httpx2.Response(429, request=_DUMMY_REQ),
            body=None,
        )
        with patch("app.infrastructure.llm_providers.anthropic.AsyncAnthropic") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.messages.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await anthropic_service.generate_chat_completion(ChatRequest(message="Hi", provider="anthropic"))

        assert exc_info.value.status_code == 429
        assert exc_info.value.provider == "anthropic"

    async def test_authentication_error_mapped_to_401(self, mock_settings: Settings) -> None:
        exc = anthropic.AuthenticationError(
            "Invalid x-api-key: sk-ant-secret12345",
            response=httpx2.Response(401, request=_DUMMY_REQ),
            body=None,
        )
        with patch("app.infrastructure.llm_providers.anthropic.AsyncAnthropic") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.messages.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await anthropic_service.generate_chat_completion(ChatRequest(message="Hi", provider="anthropic"))

        assert exc_info.value.status_code == 401
        assert "sk-ant-secret12345" not in exc_info.value.message

    async def test_timeout_error_mapped_to_504(self, mock_settings: Settings) -> None:
        exc = anthropic.APITimeoutError(request=_DUMMY_REQ)
        with patch("app.infrastructure.llm_providers.anthropic.AsyncAnthropic") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.messages.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await anthropic_service.generate_chat_completion(ChatRequest(message="Hi", provider="anthropic"))

        assert exc_info.value.status_code == 504

    async def test_unexpected_sdk_error_does_not_leak_raw_secret(self, mock_settings: Settings) -> None:
        exc = anthropic.InternalServerError(
            "Crash while processing sk-ant-live-super-secret-999",
            response=httpx2.Response(500, request=_DUMMY_REQ),
            body=None,
        )
        with patch("app.infrastructure.llm_providers.anthropic.AsyncAnthropic") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.messages.create.side_effect = exc

            with pytest.raises(LLMProviderError) as exc_info:
                await anthropic_service.generate_chat_completion(ChatRequest(message="Hi", provider="anthropic"))

        assert exc_info.value.status_code == 502
        assert "sk-ant-live-super-secret-999" not in exc_info.value.message
        assert "[REDACTED]" in exc_info.value.message
