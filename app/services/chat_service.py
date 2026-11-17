from collections.abc import Awaitable, Callable

from app.core.exceptions import AppBaseException
from app.schemas.chat import ChatRequest, ChatResponse, Provider
from app.services import anthropic_service, groq_service, openai_service

ProviderHandler = Callable[[ChatRequest], Awaitable[ChatResponse]]

_PROVIDER_HANDLERS: dict[Provider, ProviderHandler] = {
    "openai": openai_service.generate_chat_completion,
    "anthropic": anthropic_service.generate_chat_completion,
    "groq": groq_service.generate_chat_completion,
}


async def generate_chat_completion(payload: ChatRequest) -> ChatResponse:
    """Select the target LLM provider service and return a normalized response."""
    handler = _PROVIDER_HANDLERS.get(payload.provider)
    if handler is None:
        raise AppBaseException(
            message=f"Unsupported LLM provider: {payload.provider}",
            status_code=400,
            error_code="unsupported_provider",
            provider=payload.provider,
        )
    return await handler(payload)
