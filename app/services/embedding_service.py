"""
Embedding service module.
Re-exports the infrastructure GeminiEmbeddingProvider as GeminiEmbeddingService
following Clean Architecture principles.
"""

from app.infrastructure.embedding_providers import (
    BaseEmbeddingProvider,
    GeminiEmbeddingProvider,
    get_embedding_provider,
)

# Canonical aliases for the service layer
GeminiEmbeddingService = GeminiEmbeddingProvider
get_embedding_service = get_embedding_provider

__all__ = [
    "BaseEmbeddingProvider",
    "GeminiEmbeddingProvider",
    "GeminiEmbeddingService",
    "get_embedding_provider",
    "get_embedding_service",
]
