import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Any, Dict, List

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base

if TYPE_CHECKING:
    from app.models.document import Document


class DocumentChunk(Base):
    """
    Represents a chunk of a parent document.
    Stores chunk-level content, Gemini vector embedding (768 dimensions),
    foreign key to parent document, position/index within the document,
    and metadata for retrieval and source citation.
    """

    __tablename__ = "document_chunks"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
    )
    chunk_size: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=400,
    )
    overlap: Mapped[int] = mapped_column(
        Integer,
        nullable=False,
        default=80,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    embedding: Mapped[List[float]] = mapped_column(
        Vector(768),
        nullable=False,
    )
    chunk_metadata: Mapped[Dict[str, Any]] = mapped_column(
        "metadata",
        JSONB().with_variant(JSON(), "sqlite"),
        nullable=False,
        default=dict,
    )
    embedding_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default="gemini-embedding-001",
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    document: Mapped["Document"] = relationship(
        "Document",
        back_populates="chunks",
    )

    __table_args__ = (
        Index("ix_document_chunks_doc_chunk_idx", "document_id", "chunk_index"),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk(id={self.id}, doc_id={self.document_id}, index={self.chunk_index}, len={len(self.content)})>"
