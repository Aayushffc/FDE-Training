from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.search import (
    ChunkSearchResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    KeywordSearchRequest,
    KeywordSearchResponse,
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
    summary="Execute semantic similarity search over stored vector documents",
)
def semantic_search_endpoint(
    payload: SemanticSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SemanticSearchResponse:
    """
    Execute semantic cosine similarity search over ingested vector documents.

    Workflow:
    1. Receives natural language search query and optional metadata filters (e.g., {'category': 'account'}).
    2. Embeds the query into a 768-dimensional vector via Google Gemini (gemini-embedding-001).
    3. Performs cosine distance similarity search in Neon PostgreSQL using pgvector `<=>`.
    4. Filters candidates by metadata and returns the top-K closest documents.
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
    "/keyword",
    response_model=KeywordSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute PostgreSQL full-text keyword search over stored documents",
)
def keyword_search_endpoint(
    payload: KeywordSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> KeywordSearchResponse:
    """
    Execute keyword search using PostgreSQL native full-text search.

    Workflow:
    1. Receives search query keywords and optional metadata filters (e.g., {'category': 'billing'}).
    2. Constructs tsvector and websearch_to_tsquery against document content in PostgreSQL.
    3. Matches using `@@` and ranks results using `ts_rank`.
    4. Returns the top-K highest-ranking documents.
    """
    try:
        return search_service.keyword_search(
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
            detail=f"Keyword search failed: {str(exc)}",
        )


@router.post(
    "/hybrid",
    response_model=HybridSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute hybrid search (Keyword + Vector with Reciprocal Rank Fusion) over document chunks",
)
def hybrid_search_endpoint(
    payload: HybridSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> HybridSearchResponse:
    """
    Execute hybrid search over document chunks:
    1. Executes vector semantic search via Gemini embeddings & pgvector.
    2. Executes PostgreSQL full-text keyword search via tsvector and ts_rank.
    3. Merges and re-ranks results using Reciprocal Rank Fusion (RRF) or normalized weighted scoring.
    4. Optionally applies cross-encoder reranking to refine relevance.
    5. Returns ranked chunks along with vector/keyword rank citations and parent document IDs.
    """
    try:
        resp = search_service.chunk_hybrid_search(
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
            alpha=payload.alpha,
            rrf_k=payload.rrf_k,
            fusion_method=payload.fusion_method,
        )
        if payload.rerank and resp.results:
            from app.services.reranking_service import RerankingService
            reranker = RerankingService()
            reranked_results = reranker.rerank(query=payload.query, items=resp.results, top_k=payload.top_k)
            resp.results = reranked_results
        return resp
    except ValueError as val_err:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(val_err),
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Hybrid search failed: {str(exc)}",
        )


