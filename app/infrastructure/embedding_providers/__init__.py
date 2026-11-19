from app.infrastructure.embedding_providers.base import BaseEmbeddingProvider
from app.infrastructure.embedding_providers.gemini import GeminiEmbeddingProvider


def get_embedding_provider() -> BaseEmbeddingProvider:
    """Factory creating the configured default embedding provider."""
    return GeminiEmbeddingProvider()


__all__ = [
    "BaseEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "get_embedding_provider",
]
