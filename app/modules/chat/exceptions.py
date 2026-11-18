from app.core.exceptions import AppBaseException, LLMProviderError, StructuredOutputValidationError


class ChatModuleException(AppBaseException):
    """Base exception for the chat module."""
    pass


class UnsupportedProviderError(ChatModuleException):
    """Raised when an unsupported provider is requested."""

    def __init__(self, provider: str):
        super().__init__(
            message=f"Unsupported LLM provider: {provider}",
            status_code=400,
            error_code="unsupported_provider",
            provider=provider,
        )


class PromptNotFoundError(ChatModuleException):
    """Raised when a requested prompt version or template is not found."""

    def __init__(
        self,
        version: str,
        category: str | None = None,
        available: list[str] | None = None,
    ):
        avail_msg = f" Available versions for '{category}': {available}." if (available and category) else ""
        cat_msg = f" for category '{category}'" if category else ""
        super().__init__(
            message=f"Prompt version '{version}'{cat_msg} not found.{avail_msg}",
            status_code=400,
            error_code="prompt_not_found",
        )


__all__ = [
    "ChatModuleException",
    "UnsupportedProviderError",
    "PromptNotFoundError",
    "LLMProviderError",
    "StructuredOutputValidationError",
]

