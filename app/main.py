from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.core.exceptions import register_exception_handlers
from app.routers import chat

app = FastAPI(
    title="FDE Sprint LLM Gateway",
    version="0.1.0",
    description="Unified FastAPI wrapper for OpenAI and Anthropic with per-request observability.",
)

register_exception_handlers(app)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", tags=["Health"])
async def health_check() -> dict[str, str]:
    return {"status": "ok"}


app.include_router(chat.router)
