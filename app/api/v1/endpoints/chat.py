from fastapi import APIRouter, Depends

from app.api.v1.dependencies import get_chat_service
from app.modules.chat.schemas import ChatRequest, ChatResponse
from app.modules.chat.service import ChatService

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(
    payload: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
) -> ChatResponse:
    """Handle chat completion requests by delegating to the chat domain service."""
    return await chat_service.generate_chat_completion(payload)
