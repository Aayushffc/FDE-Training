"""
FastAPI V1 Dependency Injection Module
======================================
Centralized dependency providers for database sessions, infrastructure adapters,
and domain services following Clean Architecture principles.
"""

from typing import Optional

from fastapi import Depends
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.infrastructure.embedding_providers import BaseEmbeddingProvider, get_embedding_provider
from app.infrastructure.llm_providers.base import BaseLLMProvider
from app.infrastructure.llm_providers.registry import default_registry
from app.infrastructure.rerankers import BaseReranker, get_reranker
from app.modules.chat.service import ChatService, get_chat_service
from app.services.chunking_service import ChunkingService
from app.services.context_service import ContextEngineeringService
from app.services.ingestion_service import IngestionService
from app.services.prompt_service import PromptService
from app.services.rag_service import RAGService
from app.services.reranking_service import RerankingService
from app.services.search_service import SearchService


# ==============================================================================
# Infrastructure Providers
# ==============================================================================

def get_configured_embedding_provider() -> BaseEmbeddingProvider:
    """Dependency provider for the active embedding provider (Google Gemini)."""
    return get_embedding_provider()


def get_default_llm_provider() -> Optional[BaseLLMProvider]:
    """Dependency provider for the default registered LLM provider."""
    return default_registry.get("openai")


def get_configured_reranker() -> BaseReranker:
    """Dependency provider for the candidate chunk reranker."""
    return get_reranker()


# ==============================================================================
# Service Layer Providers
# ==============================================================================

def get_chunking_service() -> ChunkingService:
    """Dependency provider for ChunkingService."""
    return ChunkingService()


def get_context_service() -> ContextEngineeringService:
    """Dependency provider for ContextEngineeringService."""
    return ContextEngineeringService()


def get_reranking_service(
    reranker: BaseReranker = Depends(get_configured_reranker),
) -> RerankingService:
    """Dependency provider for RerankingService."""
    return RerankingService(reranker=reranker)


def get_prompt_service(
    db: Session = Depends(get_db),
) -> PromptService:
    """Dependency provider for PromptService."""
    return PromptService(db=db)


def get_search_service(
    db: Session = Depends(get_db),
    embedder: BaseEmbeddingProvider = Depends(get_configured_embedding_provider),
) -> SearchService:
    """Dependency provider for SearchService."""
    return SearchService(db=db, embedding_provider=embedder)


def get_ingestion_service(
    db: Session = Depends(get_db),
    embedder: BaseEmbeddingProvider = Depends(get_configured_embedding_provider),
    chunker: ChunkingService = Depends(get_chunking_service),
) -> IngestionService:
    """Dependency provider for IngestionService."""
    return IngestionService(db=db, embedding_provider=embedder, chunking_service=chunker)


def get_rag_service(
    db: Session = Depends(get_db),
    search_service: SearchService = Depends(get_search_service),
    reranking_service: RerankingService = Depends(get_reranking_service),
    context_service: ContextEngineeringService = Depends(get_context_service),
    prompt_service: PromptService = Depends(get_prompt_service),
) -> RAGService:
    """Dependency provider for RAGService."""
    return RAGService(
        db=db,
        search_service=search_service,
        reranking_service=reranking_service,
        context_service=context_service,
        prompt_service=prompt_service,
    )


__all__ = [
    # DB
    "get_db",
    # Infrastructure
    "get_configured_embedding_provider",
    "get_default_llm_provider",
    "get_configured_reranker",
    # Services
    "ChatService",
    "get_chat_service",
    "get_chunking_service",
    "get_context_service",
    "get_reranking_service",
    "get_prompt_service",
    "get_search_service",
    "get_ingestion_service",
    "get_rag_service",
]
