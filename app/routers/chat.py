from collections.abc import Awaitable, Callable

from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse, Provider
from app.services import anthropic_service, groq_service, openai_service

router = APIRouter(prefix="/chat", tags=["Chat"])

PROVIDER_SERVICES: dict[Provider, Callable[[ChatRequest], Awaitable[ChatResponse]]] = {
    "openai": openai_service.generate_chat_completion,
    "anthropic": anthropic_service.generate_chat_completion,
    "groq": groq_service.generate_chat_completion,
}


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    handler = PROVIDER_SERVICES[payload.provider]
    return await handler(payload)
