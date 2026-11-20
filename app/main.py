from app.core.logging import register_request_logging_middleware, setup_logging

setup_logging()

from fastapi import FastAPI
from fastapi.responses import RedirectResponse

from app.api.v1.router import api_v1_router
from app.core.exceptions import register_exception_handlers

app = FastAPI(
    title="FDE Sprint LLM Gateway",
    version="0.1.0",
    description="Unified Clean Architecture FastAPI gateway for OpenAI, Anthropic, and Groq with per-request observability.",
)

register_request_logging_middleware(app)
register_exception_handlers(app)


@app.get("/", include_in_schema=False)
async def root() -> RedirectResponse:
    """Redirect root path to interactive Swagger documentation."""
    return RedirectResponse(url="/docs")


# Mount versioned API routes under /v1
app.include_router(api_v1_router, prefix="/v1")
