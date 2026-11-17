from fastapi import APIRouter
from app.schemas.chat import ChatRequest, ChatResponse
from app.services import anthropic_service, openai_service

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    if payload.provider == "openai":
        return await openai_service.generate_chat_completion(payload)
    return await anthropic_service.generate_chat_completion(payload)