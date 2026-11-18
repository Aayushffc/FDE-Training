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


__all__ = [
    "ChatModuleException",
    "UnsupportedProviderError",
    "LLMProviderError",
    "StructuredOutputValidationError",
]
