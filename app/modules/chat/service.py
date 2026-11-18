from typing import Any

from app.infrastructure.llm_providers.registry import LLMProviderRegistry, default_registry
from app.modules.chat.exceptions import UnsupportedProviderError
from app.modules.chat.schemas import ChatRequest, ChatResponse
from app.modules.prompts.service import PromptService, default_prompt_service

# Mock handler mapping for test overrides
_PROVIDER_HANDLERS: dict[str, Any] = {}


class ChatService:
    """Application domain service for orchestrating chat completions."""

    def __init__(
        self,
        registry: LLMProviderRegistry | None = None,
        prompt_service: PromptService | None = None,
    ) -> None:
        self.registry = registry or default_registry
        self.prompt_service = prompt_service or default_prompt_service

    async def generate_chat_completion(self, payload: ChatRequest) -> ChatResponse:
        # Check if an explicit handler mock has been registered for tests
        if payload.provider in _PROVIDER_HANDLERS:
            return await _PROVIDER_HANDLERS[payload.provider](payload)

        provider = self.registry.get(payload.provider)
        if provider is None:
            raise UnsupportedProviderError(payload.provider)

        system_prompt = self.prompt_service.get_system_prompt()
        return await provider.generate_chat_completion(payload.message, system_prompt=system_prompt)


default_chat_service = ChatService()


async def generate_chat_completion(payload: ChatRequest) -> ChatResponse:
    """Package-level convenience function for generating completions."""
    return await default_chat_service.generate_chat_completion(payload)


def get_chat_service() -> ChatService:
    """Dependency injection provider for FastAPI endpoints."""
    return default_chat_service
