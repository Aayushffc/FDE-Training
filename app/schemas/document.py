import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class DocumentIngestRequest(BaseModel):
    """Payload for injecting a single document into the vector store."""

    content: str = Field(
        ...,
        min_length=1,
        description="Original document text content to be embedded and stored",
        examples=["To reset your password, open the login page and select Forgot Password."],
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary JSON metadata (category, source, language, tags, etc.)",
        examples=[{"category": "account", "source": "faq", "language": "en", "tags": ["password", "login"]}],
    )


class DocumentBatchIngestRequest(BaseModel):
    """Payload for injecting multiple documents in a single batch."""

    documents: List[DocumentIngestRequest] = Field(
        ...,
        min_length=1,
        max_length=100,
        description="List of documents to ingest",
    )


class DocumentResponse(BaseModel):
    """Response representing an ingested document stored in PostgreSQL."""

    id: uuid.UUID = Field(..., description="Unique document UUID")
    content: str = Field(..., description="Original document content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("doc_metadata", "metadata"),
        serialization_alias="metadata",
        description="Document metadata dictionary",
    )
    embedding_model: str = Field(..., description="Embedding model identifier used")
    embedding_dimensions: Optional[int] = Field(default=768, description="Dimensionality of the vector embedding")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last modification timestamp")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class DocumentListResponse(BaseModel):
    """Paginated or listed collection of documents."""

    total: int
    documents: List[DocumentResponse]
