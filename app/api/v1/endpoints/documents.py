import uuid
from typing import List

from fastapi import APIRouter, Depends, HTTPException, Path, Query, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.chunk import (
    ChunkDocumentRequest,
    ChunkListResponse,
    ChunkResponse,
    DocumentChunkIngestRequest,
    DocumentWithChunksResponse,
)
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


@router.post(
    "/{document_id}/chunks",
    response_model=List[ChunkResponse],
    status_code=status.HTTP_201_CREATED,
    summary="Split an existing document into chunks and generate chunk embeddings",
)
def chunk_existing_document_endpoint(
    document_id: uuid.UUID = Path(..., description="Unique parent document UUID"),
    payload: ChunkDocumentRequest = ChunkDocumentRequest(),
    db: Session = Depends(get_db),
) -> List[ChunkResponse]:
    """
    Split an existing stored document into smaller pieces (chunks), calculate Gemini 768-dim
    embeddings for each chunk, and persist records to the 'document_chunks' table.
    """
    service = IngestionService(db)
    try:
        chunks = service.chunk_existing_document(
            document_id=document_id,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            strategy=payload.strategy,
            preserve_boundaries=payload.preserve_boundaries,
            replace_existing=True,
        )
        return [ChunkResponse.model_validate(c) for c in chunks]
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to chunk document: {str(exc)}",
        )


@router.get(
    "/{document_id}/chunks",
    response_model=ChunkListResponse,
    summary="List all chunks belonging to a document",
)
def list_document_chunks(
    document_id: uuid.UUID = Path(..., description="Parent document UUID"),
    limit: int = Query(default=100, ge=1, le=500),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> ChunkListResponse:
    """Retrieve all stored chunks for a given parent document."""
    service = IngestionService(db)
    chunks, total = service.list_chunks(document_id=document_id, limit=limit, offset=offset)
    return ChunkListResponse(
        total=total,
        document_id=document_id,
        chunks=[ChunkResponse.model_validate(c) for c in chunks],
    )


@router.get(
    "/{document_id}/with-chunks",
    response_model=DocumentWithChunksResponse,
    summary="Retrieve parent document together with all its chunks",
)
def get_document_with_chunks(
    document_id: uuid.UUID = Path(..., description="Parent document UUID"),
    db: Session = Depends(get_db),
) -> DocumentWithChunksResponse:
    """Retrieve a parent document and its ordered list of chunks."""
    service = IngestionService(db)
    doc = service.get_document(document_id)
    if not doc:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Document with ID '{document_id}' was not found.",
        )
    chunks, _ = service.list_chunks(document_id=document_id, limit=1000)
    return DocumentWithChunksResponse(
        id=doc.id,
        content=doc.content,
        metadata=doc.doc_metadata,
        embedding_model=doc.embedding_model,
        total_chunks=len(chunks),
        chunks=[ChunkResponse.model_validate(c) for c in chunks],
        created_at=doc.created_at,
        updated_at=doc.updated_at,
    )


@router.get(
    "/chunks/{chunk_id}",
    response_model=ChunkResponse,
    summary="Retrieve a specific chunk by its UUID",
)
def get_single_chunk(
    chunk_id: uuid.UUID = Path(..., description="Unique chunk UUID"),
    db: Session = Depends(get_db),
) -> ChunkResponse:
    """Fetch an individual chunk record and its metadata."""
    service = IngestionService(db)
    chunk = service.get_chunk(chunk_id)
    if not chunk:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Chunk with ID '{chunk_id}' was not found.",
        )
    return ChunkResponse.model_validate(chunk)


@router.post(
    "/chunked",
    response_model=DocumentWithChunksResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Ingest a document and immediately create embedded chunks",
)
def ingest_document_with_chunks_endpoint(
    payload: DocumentChunkIngestRequest,
    db: Session = Depends(get_db),
) -> DocumentWithChunksResponse:
    """
    Ingest text into the 'documents' table as the parent record, chunk it into smaller pieces,
    embed each chunk via Gemini, and store them in the 'document_chunks' table.
    """
    service = IngestionService(db)
    try:
        doc, chunks = service.ingest_document_with_chunks(
            content=payload.content,
            metadata=payload.metadata,
            chunk_size=payload.chunk_size,
            chunk_overlap=payload.chunk_overlap,
            strategy=payload.strategy,
            preserve_boundaries=payload.preserve_boundaries,
            skip_duplicates=False,
        )
        return DocumentWithChunksResponse(
            id=doc.id,
            content=doc.content,
            metadata=doc.doc_metadata,
            embedding_model=doc.embedding_model,
            total_chunks=len(chunks),
            chunks=[ChunkResponse.model_validate(c) for c in chunks],
            created_at=doc.created_at,
            updated_at=doc.updated_at,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to ingest chunked document: {str(exc)}",
        )

