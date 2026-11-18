from unittest.mock import AsyncMock
import pytest

from app.infrastructure.llm_providers.base import BaseLLMProvider
from app.infrastructure.llm_providers.pricing import calculate_cost_usd
from app.infrastructure.llm_providers.registry import LLMProviderRegistry, default_registry
from app.modules.chat.exceptions import UnsupportedProviderError
from app.modules.chat.schemas import ChatRequest, ChatResponse
from app.modules.chat.service import ChatService
from app.modules.prompts.service import PromptService, default_prompt_service


class TestPromptModule:
    def test_default_prompt_service_loads_template(self) -> None:
        prompt = default_prompt_service.get_system_prompt()
        assert "concise, accurate, and reliable" in prompt
        assert "Guidelines:" in prompt

    def test_prompt_service_caching(self) -> None:
        service = PromptService()
        p1 = service.get_system_prompt("default")
        p2 = service.get_system_prompt("default")
        assert p1 is p2

    def test_prompt_service_missing_template_raises(self) -> None:
        service = PromptService()
        with pytest.raises(FileNotFoundError):
            service.get_system_prompt("nonexistent_template_xyz")


class TestProviderRegistry:
    def test_default_registry_has_expected_providers(self) -> None:
        providers = default_registry.list_providers()
        assert "openai" in providers
        assert "anthropic" in providers
        assert "groq" in providers

    def test_registry_case_insensitivity(self) -> None:
        registry = LLMProviderRegistry()
        mock_provider = AsyncMock(spec=BaseLLMProvider)
        registry.register("MyProvider", mock_provider)

        assert registry.get("myprovider") is mock_provider
        assert registry.get("MYPROVIDER") is mock_provider
        assert registry.get("unknown") is None


class TestChatServiceWithRegistry:
    async def test_chat_service_delegates_to_registered_provider(self) -> None:
        mock_provider = AsyncMock(spec=BaseLLMProvider)
        fake_response = ChatResponse(
            provider="openai",
            model="gpt-test",
            reply="Registry response",
            input_tokens=5,
            output_tokens=5,
            latency_ms=10.0,
            estimated_cost_usd=0.000001,
        )
        mock_provider.generate_chat_completion.return_value = fake_response

        registry = LLMProviderRegistry()
        registry.register("openai", mock_provider)

        service = ChatService(registry=registry)
        res = await service.generate_chat_completion(ChatRequest(message="Hello", provider="openai"))

        assert res.reply == "Registry response"
        mock_provider.generate_chat_completion.assert_awaited_once()

    async def test_chat_service_unsupported_provider_raises(self) -> None:
        registry = LLMProviderRegistry()
        service = ChatService(registry=registry)
        with pytest.raises(UnsupportedProviderError):
            await service.generate_chat_completion(ChatRequest(message="Hello", provider="openai"))


class TestInfrastructurePricing:
    def test_direct_infrastructure_pricing_calculation(self) -> None:
        cost = calculate_cost_usd("claude-haiku-5-5", input_tokens=10_000, output_tokens=2_000)
        # 10k * $0.10 + 2k * $0.50 per 1M = (1.0 + 1.0)/1000 = $0.002
        assert cost == pytest.approx(0.002)
