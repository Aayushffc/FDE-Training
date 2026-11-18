from abc import ABC, abstractmethod
from typing import Any

from app.modules.chat.schemas import ChatRequest, ChatResponse


class BaseLLMProvider(ABC):
    """
    Abstract adapter interface for LLM providers.
    Follows Clean Architecture Dependency Inversion:
    application domain modules interact with BaseLLMProvider,
    while infrastructure adapters implement SDK-specific details.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Provider identifier string (e.g., 'openai', 'anthropic', 'groq')."""
        ...

    @property
    @abstractmethod
    def default_model(self) -> str:
        """Default model configured for this provider."""
        ...

    @abstractmethod
    async def generate_chat_completion(
        self,
        payload_or_message: ChatRequest | str,
        system_prompt: str | None = None,
    ) -> ChatResponse:
        """Generate a standardized ChatResponse from a prompt message or ChatRequest."""
        ...

    async def generate(
        self,
        message: str,
        system_prompt: str | None = None,
    ) -> ChatResponse:
        """Shortcut method to generate a chat completion."""
        return await self.generate_chat_completion(message, system_prompt=system_prompt)
