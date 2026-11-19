from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.search import SemanticSearchRequest, SemanticSearchResponse
from app.services.search_service import SearchService

router = APIRouter(prefix="/search", tags=["Search"])


def get_search_service(db: Session = Depends(get_db)) -> SearchService:
    """Dependency provider for SearchService."""
    return SearchService(db=db)


@router.post(
    "/semantic",
    response_model=SemanticSearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Execute semantic similarity search over stored documents",
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
