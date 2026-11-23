from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.search import (
    ChunkedSearchRequest,
    HybridSearchResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["Search"])


def get_search_service(db: Session = Depends(get_db)) -> SearchService:
    """Dependency provider for SearchService."""
    return SearchService(db=db)


@router.post(
    "/semantic",
    response_model=SemanticSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute semantic similarity search strictly over stored UNCHUNKED documents",
)
def semantic_search_endpoint(
    payload: SemanticSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SemanticSearchResponse:
    """
    Execute semantic cosine similarity search strictly over unchunked parent documents:
    1. Receives natural language search query and optional metadata filters (e.g., {'category': 'account'}).
    2. Embeds the query into a 768-dimensional vector via Google Gemini (gemini-embedding-001).
    3. Performs cosine distance similarity search in Neon PostgreSQL using pgvector `<=>` against the 'documents' table.
    4. Filters candidates by metadata and returns the top-K closest whole documents.
    """
    try:
        return search_service.semantic_search(
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Semantic search failed: {str(exc)}",
        )


@router.post(
    "/chunked",
    response_model=HybridSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute search strictly over stored DOCUMENT CHUNKS",
)
def chunked_search_endpoint(
    payload: ChunkedSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> HybridSearchResponse:
    """
    Execute search strictly over document chunks in the 'document_chunks' table:
    1. Supports strategy: 'hybrid' (vector + keyword RRF, default), 'vector' (semantic only), or 'keyword' (FTS only).
    2. When hybrid, merges candidates using Reciprocal Rank Fusion (RRF, k=60).
    3. Optionally applies cross-encoder reranking to reorder candidates by token density & phrase proximity.
    4. Returns ranked chunks with parent document references, citations, and scores.
    """
    try:
        return search_service.chunk_search(
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
            strategy=payload.strategy,
            alpha=payload.alpha,
            rrf_k=payload.rrf_k,
            rerank=payload.rerank,
        )
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Chunked search failed: {str(exc)}",
        )


