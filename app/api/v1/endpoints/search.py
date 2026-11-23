from fastapi import APIRouter, Depends, HTTPException, status

from app.api.v1.dependencies import get_reranking_service, get_search_service
from app.schemas.search import (
    ChunkSearchResponse,
    DocumentHybridSearchResponse,
    HybridSearchRequest,
    HybridSearchResponse,
    KeywordSearchRequest,
    KeywordSearchResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
)
from app.services.reranking_service import RerankingService
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["Search"])


# ==============================================================================
# UNCHUNKED SEARCH ENDPOINTS (Operate strictly on 'documents' table)
# ==============================================================================

@router.post(
    "/semantic",
    response_model=SemanticSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute semantic vector search strictly over stored UNCHUNKED documents",
)
def semantic_search_endpoint(
    payload: SemanticSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> SemanticSearchResponse:
    """Execute cosine similarity search over unchunked parent documents via Gemini embeddings & pgvector."""
    try:
        return search_service.semantic_search(
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Semantic search failed: {str(exc)}")


@router.post(
    "/keyword",
    response_model=KeywordSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute full-text keyword search strictly over stored UNCHUNKED documents",
)
def keyword_search_endpoint(
    payload: KeywordSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> KeywordSearchResponse:
    """Execute PostgreSQL full-text search over unchunked parent documents using to_tsvector and ts_rank."""
    try:
        return search_service.keyword_search(
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Keyword search failed: {str(exc)}")


@router.post(
    "/hybrid",
    response_model=DocumentHybridSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute hybrid search (RRF) strictly over stored UNCHUNKED documents",
)
def document_hybrid_search_endpoint(
    payload: HybridSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> DocumentHybridSearchResponse:
    """Execute hybrid search combining vector and keyword search using Reciprocal Rank Fusion over unchunked documents."""
    try:
        return search_service.document_hybrid_search(
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
            alpha=payload.alpha,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Document hybrid search failed: {str(exc)}")


# ==============================================================================
# CHUNKED SEARCH ENDPOINTS (Operate strictly on 'document_chunks' table)
# ==============================================================================

@router.post(
    "/chunks/semantic",
    response_model=ChunkSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute semantic vector search strictly over stored DOCUMENT CHUNKS",
)
def chunk_semantic_search_endpoint(
    payload: SemanticSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> ChunkSearchResponse:
    """Execute cosine similarity vector search over document_chunks with Gemini embeddings."""
    try:
        return search_service.chunk_vector_search(
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Chunk semantic search failed: {str(exc)}")


@router.post(
    "/chunks/keyword",
    response_model=ChunkSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute full-text keyword search strictly over stored DOCUMENT CHUNKS",
)
def chunk_keyword_search_endpoint(
    payload: KeywordSearchRequest,
    search_service: SearchService = Depends(get_search_service),
) -> ChunkSearchResponse:
    """Execute PostgreSQL full-text keyword search over document_chunks with ts_rank scoring."""
    try:
        return search_service.chunk_keyword_search(
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
        )
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Chunk keyword search failed: {str(exc)}")


@router.post(
    "/chunks/hybrid",
    response_model=HybridSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute hybrid search (RRF) over stored DOCUMENT CHUNKS with optional reranking",
)
def chunk_hybrid_search_endpoint(
    payload: HybridSearchRequest,
    search_service: SearchService = Depends(get_search_service),
    reranker: RerankingService = Depends(get_reranking_service),
) -> HybridSearchResponse:
    """
    Execute hybrid search over document_chunks fusing vector and keyword candidates using
    Reciprocal Rank Fusion (RRF). Optionally applies cross-encoder reranking to promote
    the most relevant chunk to rank 1.
    """
    try:
        resp = search_service.chunk_hybrid_search(
            query=payload.query,
            top_k=payload.top_k,
            filters=payload.filters,
            alpha=payload.alpha,
        )
        if payload.rerank and resp.results:
            resp.results = reranker.rerank(query=payload.query, items=resp.results, top_k=payload.top_k)
        return resp
    except ValueError as val_err:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(val_err))
    except Exception as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Chunk hybrid search failed: {str(exc)}")
