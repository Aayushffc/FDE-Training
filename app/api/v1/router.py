from fastapi import APIRouter

from app.api.v1.endpoints import chat, documents, health, prompts, search

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(chat.router)
api_v1_router.include_router(prompts.router)
api_v1_router.include_router(documents.router)
api_v1_router.include_router(search.router)

__all__ = ["api_v1_router"]
