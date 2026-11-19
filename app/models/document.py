import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, JSON, String, Text, Uuid, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base


class Document(Base):
    """
    Stores documents with raw text content, Gemini vector embeddings (768 dimensions),
    and JSON metadata for semantic search and RAG retrieval.
    """

    __tablename__ = "documents"

    id: Mapped[uuid.UUID] = mapped_column(
        Uuid,
        primary_key=True,
        default=uuid.uuid4,
    )
    content: Mapped[str] = mapped_column(
        Text,
        nullable=False,
    )
    embedding: Mapped[List[float]] = mapped_column(
        Vector(768),
        nullable=False,
    )
    doc_metadata: Mapped[Dict[str, Any]] = mapped_column(
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

    def __repr__(self) -> str:
        return f"<Document(id={self.id}, model='{self.embedding_model}', content_len={len(self.content)})>"
