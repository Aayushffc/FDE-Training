import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, model_validator


class ChunkResponse(BaseModel):
    """Response schema representing an individual document chunk."""

    id: uuid.UUID = Field(..., description="Unique chunk UUID")
    document_id: uuid.UUID = Field(..., description="Foreign key to parent document UUID")
    chunk_index: int = Field(..., description="Positional index of chunk within the parent document (0-indexed)")
    chunk_size: Optional[int] = Field(default=400, description="Target chunk size configured for this chunk")
    overlap: Optional[int] = Field(default=80, description="Chunk overlap configured for this chunk")
    content: str = Field(..., description="Text content of the chunk")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("chunk_metadata", "metadata"),
        serialization_alias="metadata",
        description="Chunk and parent metadata dictionary",
    )
    embedding_model: str = Field(..., description="Embedding model identifier used")
    embedding_dimensions: Optional[int] = Field(default=768, description="Dimensionality of the vector embedding")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: datetime = Field(..., description="Last modification timestamp")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ChunkListResponse(BaseModel):
    """Collection of chunks for a document or query."""

    total: int = Field(..., description="Total number of chunks")
    document_id: Optional[uuid.UUID] = Field(default=None, description="Parent document UUID if scoped")
    chunks: List[ChunkResponse] = Field(default_factory=list, description="List of chunk responses")


class DocumentWithChunksResponse(BaseModel):
    """Parent document response along with its associated chunk records."""

    id: uuid.UUID = Field(..., description="Parent document UUID")
    content: str = Field(..., description="Full original document content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("doc_metadata", "metadata"),
        serialization_alias="metadata",
    )
    embedding_model: str
    total_chunks: int = Field(..., description="Total number of generated chunks")
    chunks: List[ChunkResponse] = Field(default_factory=list, description="Ordered chunks belonging to this document")
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ChunkDocumentRequest(BaseModel):
    """Request payload to chunk an existing document."""

    chunk_size: int = Field(default=400, ge=50, le=4000, description="Target size per chunk (chars or words)")
    chunk_overlap: int = Field(default=80, ge=0, le=2000, description="Overlap between consecutive chunks")
    strategy: str = Field(default="character", description="Chunking strategy: 'character' or 'word'")
    preserve_boundaries: bool = Field(default=True, description="Whether to split on sentence/word boundaries")

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkDocumentRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        return self


class DocumentChunkIngestRequest(BaseModel):
    """Request to ingest a new document and immediately create embedded chunks."""

    content: str = Field(..., min_length=1, description="Raw text content of the parent document")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata dictionary for the document")
    chunk_size: int = Field(default=400, ge=50, le=4000, description="Target chunk size")
    chunk_overlap: int = Field(default=80, ge=0, le=2000, description="Chunk overlap")
    strategy: str = Field(default="character", description="'character' or 'word'")
    preserve_boundaries: bool = Field(default=True, description="Whether to preserve sentence/word boundaries")

    @model_validator(mode="after")
    def validate_overlap(self) -> "DocumentChunkIngestRequest":
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be strictly less than chunk_size")
        return self
