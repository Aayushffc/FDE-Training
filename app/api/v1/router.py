from fastapi import APIRouter

from app.api.v1.endpoints import chat, health, prompts

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(chat.router)
api_v1_router.include_router(prompts.router)

__all__ = ["api_v1_router"]
