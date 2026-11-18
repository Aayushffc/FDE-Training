from typing import Literal

from fastapi import APIRouter
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = Field(default="ok", examples=["ok"])


router = APIRouter(tags=["Health"])


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Return health status of the service."""
    return HealthResponse(status="ok")
