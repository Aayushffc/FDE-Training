from fastapi import APIRouter

from app.schemas.chat import ChatRequest, ChatResponse
from app.services import chat_service

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("", response_model=ChatResponse)
async def chat(payload: ChatRequest) -> ChatResponse:
    return await chat_service.generate_chat_completion(payload)
