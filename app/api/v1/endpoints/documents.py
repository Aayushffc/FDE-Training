import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.document import (
    DocumentBatchIngestRequest,
    DocumentIngestRequest,
    DocumentListResponse,
    DocumentResponse,
)
from app.services.ingestion_service import IngestionService

router = APIRouter(prefix="/documents", tags=["Documents & Ingestion"])


@router.post(
    "",
    response_model=DocumentResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a single document with Gemini vector embedding",
)
def ingest_document(
    payload: DocumentIngestRequest,
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """
    Take raw text content, generate a 768-dimensional vector embedding via Google Gemini API
    (`gemini-embedding-001`), and persist the document and embedding into PostgreSQL.
    """
    service = IngestionService(db)
    doc = service.ingest_document(
        content=payload.content,
        metadata=payload.metadata,
    )
    db.commit()
    db.refresh(doc)
    return DocumentResponse.model_validate(doc)


@router.post(
    "/batch",
    response_model=List[DocumentResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Ingest multiple documents in a single batch",
)
def ingest_batch_documents(
    payload: DocumentBatchIngestRequest,
    db: Session = Depends(get_db),
) -> List[DocumentResponse]:
    """
    Batch-embed and persist multiple documents in an optimized Gemini API call.
    """
    service = IngestionService(db)
    docs = service.ingest_batch(payload.documents)
    db.commit()
    for doc in docs:
        db.refresh(doc)
    return [DocumentResponse.model_validate(d) for d in docs]


@router.get(
    "",
    response_model=DocumentListResponse,
    summary="List ingested documents with pagination",
)
def list_documents(
    limit: int = Query(default=20, ge=1, le=100, description="Max records to return"),
    offset: int = Query(default=0, ge=0, description="Record offset for pagination"),
    db: Session = Depends(get_db),
) -> DocumentListResponse:
    """
    Retrieve stored documents and their metadata.
    """
    service = IngestionService(db)
    docs, total = service.list_documents(limit=limit, offset=offset)
    return DocumentListResponse(
        total=total,
        documents=[DocumentResponse.model_validate(d) for d in docs],
    )


@router.get(
    "/{document_id}",
    response_model=DocumentResponse,
    summary="Retrieve an ingested document by UUID",
)
def get_document(
    document_id: uuid.UUID = Path(..., description="Unique document UUID"),
    db: Session = Depends(get_db),
) -> DocumentResponse:
    """
    Fetch a document record by its UUID.
    """
    service = IngestionService(db)
    doc = service.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' was not found.",
        )
    return DocumentResponse.model_validate(doc)
