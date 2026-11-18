from fastapi import APIRouter

from app.api.v1.endpoints import chat, health

api_v1_router = APIRouter()
api_v1_router.include_router(health.router)
api_v1_router.include_router(chat.router)
api_v1_router.add_api_route("/prompts", chat.list_available_prompts, methods=["GET"], tags=["Prompts"])

__all__ = ["api_v1_router"]
