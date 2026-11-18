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


@router.get("/prompts")
async def list_available_prompts(
    chat_service: ChatService = Depends(get_chat_service),
) -> dict:
    """List available prompt categories and versions available for interactive testing."""
    return {
        "default_category": "customer_support",
        "available_customer_support_versions": chat_service.prompt_service.list_versions("customer_support"),
        "all_categories": chat_service.prompt_service.list_all(),
        "usage_example": {
            "message": "My order #12345 arrived damaged, please refund me.",
            "provider": "groq",
            "prompt_version": "v2",
        },
    }
