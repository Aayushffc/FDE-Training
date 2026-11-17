import logging
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger("uvicorn.error")

class AppBaseException(Exception):
    """Base domain exception for the application (framework-agnostic)."""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "internal_error",
        provider: str | None = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.provider = provider
        super().__init__(message)


class LLMProviderError(AppBaseException):
    """Raised when an upstream LLM API call fails (timeout, rate limit, 5xx, auth)."""

    def __init__(self, provider: str, message: str, status_code: int = 502):
        super().__init__(
            message=message,
            status_code=status_code,
            error_code="llm_provider_error",
            provider=provider,
        )


class StructuredOutputValidationError(AppBaseException):
    """Raised when LLM output fails Pydantic schema validation after max retries."""

    def __init__(self, provider: str, message: str):
        super().__init__(
            message=message,
            status_code=422,
            error_code="structured_output_validation_error",
            provider=provider,
        )


async def app_exception_handler(request: Request, exc: AppBaseException) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.error_code,
            "provider": exc.provider,
            "detail": exc.message,
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception occurred: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "provider": None,
            "detail": str(exc) if str(exc) else "An unexpected internal error occurred.",
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppBaseException, app_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
