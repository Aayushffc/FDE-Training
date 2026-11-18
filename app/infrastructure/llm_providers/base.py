from abc import ABC, abstractmethod
from collections.abc import AsyncIterator

from app.core.exceptions import LLMProviderError
from app.modules.chat.schemas import ChatRequest, ChatResponse
from app.modules.prompts.service import SYSTEM_PROMPT


class BaseLLMProvider(ABC):
    """
    Abstract adapter interface for LLM providers.
    Follows Clean Architecture Dependency Inversion and DRY principles:
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

    @property
    @abstractmethod
    def api_key_env_var(self) -> str:
        """Environment variable name for the API key (e.g. 'OPENAI_API_KEY')."""
        ...

    @property
    @abstractmethod
    def raw_api_key(self) -> str | None:
        """Retrieve raw API key from configuration."""
        ...

    def require_api_key(self) -> str:
        """Ensure the API key is configured, or raise a standard 503 LLMProviderError."""
        key = self.raw_api_key
        if not key:
            raise LLMProviderError(
                self.provider_name,
                f"{self.api_key_env_var} environment variable is not set.",
                status_code=503,
            )
        return key

    def normalize_inputs(
        self,
        payload_or_message: ChatRequest | str,
        system_prompt: str | None = None,
    ) -> tuple[str, str]:
        """Extract user message and resolved system prompt."""
        message = (
            payload_or_message.message
            if isinstance(payload_or_message, ChatRequest)
            else payload_or_message
        )
        sys_prompt = system_prompt if system_prompt is not None else SYSTEM_PROMPT
        return message, sys_prompt

    @abstractmethod
    async def generate_chat_completion(
        self,
        payload_or_message: ChatRequest | str,
        system_prompt: str | None = None,
    ) -> ChatResponse:
        """Generate a standardized ChatResponse from a prompt message or ChatRequest."""
        ...

    @abstractmethod
    async def stream_chat_completion(
        self,
        payload_or_message: ChatRequest | str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        """Yield incremental token chunks as strings from the LLM provider."""
        ...

    async def generate(
        self,
        message: str,
        system_prompt: str | None = None,
    ) -> ChatResponse:
        """Shortcut method to generate a chat completion."""
        return await self.generate_chat_completion(message, system_prompt=system_prompt)


