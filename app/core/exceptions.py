import logging
import re

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)

_SECRET_PATTERN = re.compile(
    r"(?:sk-ant-[A-Za-z0-9_\-]+|sk-[A-Za-z0-9_\-]+|gsk_[A-Za-z0-9_\-]+|Bearer\s+[A-Za-z0-9_\-\.]+)",
    re.IGNORECASE,
)


def _redact_secrets(text: str) -> str:
    return _SECRET_PATTERN.sub("[REDACTED]", text)


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

    def __init__(
        self,
        provider: str,
        message: str,
        status_code: int = 502,
        error_code: str = "llm_provider_error",
    ):
        super().__init__(
            message=message,
            status_code=status_code,
            error_code=error_code,
            provider=provider,
        )

    @classmethod
    def from_upstream(cls, provider: str, exc: Exception) -> "LLMProviderError":
        """
        Deliberately maps upstream SDK exceptions into domain LLMProviderError
        with the appropriate HTTP status code:
          - 504 Gateway Timeout: Upstream request timed out (APITimeoutError / TimeoutError)
          - 429 Too Many Requests: Upstream rate limit or quota exceeded (RateLimitError)
          - 401 Unauthorized: Invalid API key or authentication failure (AuthenticationError)
          - 403 Forbidden: Permission denied / model access restricted (PermissionDeniedError)
          - 400 Bad Request: Invalid request parameters / context window exceeded (BadRequestError)
          - 502 Bad Gateway: Upstream connection failure or 5xx server errors (APIConnectionError, InternalServerError)
        """
        exc_type = type(exc).__name__
        upstream_status = getattr(exc, "status_code", None)
        raw_detail = getattr(exc, "message", None) or str(exc)
        err_detail = _redact_secrets(raw_detail)
        provider_name = provider.capitalize()

        # 1. Gateway Timeout (504): client timed out waiting for upstream
        if "Timeout" in exc_type or isinstance(exc, TimeoutError):
            return cls(
                provider=provider,
                message=f"{provider_name} request timed out.",
                status_code=504,
            )

        # 2. Too Many Requests (429): upstream rate limit or quota exceeded
        if "RateLimit" in exc_type or upstream_status == 429:
            return cls(
                provider=provider,
                message=f"{provider_name} rate limit exceeded: {err_detail}",
                status_code=429,
            )

        # 3. Authentication (401): invalid or expired API key
        if "Authentication" in exc_type or upstream_status == 401:
            return cls(
                provider=provider,
                message=f"{provider_name} authentication failed. Check API key.",
                status_code=401,
            )

        # 4. Forbidden / Permission Denied (403): model access restricted or unauthorized
        if "PermissionDenied" in exc_type or upstream_status == 403:
            return cls(
                provider=provider,
                message=f"{provider_name} permission denied: {err_detail}",
                status_code=403,
            )

        # 5. Bad Request (400): invalid arguments or context length exceeded
        if "BadRequest" in exc_type or upstream_status == 400:
            return cls(
                provider=provider,
                message=f"{provider_name} bad request: {err_detail}",
                status_code=400,
            )

        # 6. Bad Gateway (502): connection dropped, DNS failed, or upstream 5xx
        return cls(
            provider=provider,
            message=f"{provider_name} upstream service error: {err_detail}",
            status_code=502,
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
            "detail": _redact_secrets(exc.message),
        },
    )


async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled exception occurred: %s", exc)
    return JSONResponse(
        status_code=500,
        content={
            "error": "internal_error",
            "provider": None,
            "detail": "An unexpected internal error occurred.",
        },
    )


def register_exception_handlers(app: FastAPI) -> None:
    app.add_exception_handler(AppBaseException, app_exception_handler)
    app.add_exception_handler(Exception, unhandled_exception_handler)
