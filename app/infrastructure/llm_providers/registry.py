from app.infrastructure.llm_providers.anthropic import AnthropicProvider, anthropic_provider
from app.infrastructure.llm_providers.base import BaseLLMProvider
from app.infrastructure.llm_providers.groq import GroqProvider, groq_provider
from app.infrastructure.llm_providers.openai import OpenAIProvider, openai_provider


class LLMProviderRegistry:
    """Registry pattern for LLM provider adapters."""

    def __init__(self) -> None:
        self._providers: dict[str, BaseLLMProvider] = {}

    def register(self, name: str, provider: BaseLLMProvider) -> None:
        self._providers[name.lower()] = provider

    def get(self, name: str) -> BaseLLMProvider | None:
        return self._providers.get(name.lower())

    def list_providers(self) -> list[str]:
        return list(self._providers.keys())


def create_default_registry() -> LLMProviderRegistry:
    registry = LLMProviderRegistry()
    registry.register("openai", openai_provider)
    registry.register("anthropic", anthropic_provider)
    registry.register("groq", groq_provider)
    return registry


default_registry = create_default_registry()
