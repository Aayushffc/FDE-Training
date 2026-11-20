from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.rag import RAGRequest, RAGResponse
from app.services.rag_service import RAGService

router = APIRouter(prefix="/rag", tags=["RAG & Answers"])


def get_rag_service(db: Session = Depends(get_db)) -> RAGService:
    """Dependency provider for RAGService."""
    return RAGService(db=db)


@router.post(
    "/ask",
    response_model=RAGResponse,
    status_code=status.HTTP_200_OK,
    summary="Generate answer grounded in retrieved document chunks with source citations",
)
async def ask_rag_endpoint(
    payload: RAGRequest,
    rag_service: RAGService = Depends(get_rag_service),
) -> RAGResponse:
    """
    Retrieval Augmented Generation (RAG) Answer Endpoint:
    1. Retrieves relevant document chunks using the specified strategy ('hybrid', 'vector', or 'keyword').
    2. Optionally reranks candidates using the cross-encoder relevance reranker.
    3. Assembles deduplicated, budget-trimmed, and source-labeled context.
    4. Synthesizes an answer grounded strictly in the retrieved context using the LLM.
    5. Returns citations and falls back to 'I don't know based on the provided context.' if unsupported.
    """
    try:
        return await rag_service.generate_rag_answer(payload)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"RAG answer generation failed: {str(exc)}",
        )
