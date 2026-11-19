import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.infrastructure.embedding_providers import BaseEmbeddingProvider, get_embedding_provider
from app.models.document import Document
from app.schemas.document import DocumentIngestRequest

get_embedding_service = get_embedding_provider


class IngestionService:
    """
    Coordinates document ingestion: takes text, calculates embeddings (768-dim)
    via an infrastructure embedding provider, and persists records into the PostgreSQL vector database.
    """

    def __init__(
        self,
        db: Session,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        embedding_service: Optional[BaseEmbeddingProvider] = None,
    ) -> None:
        self.db = db
        self.embedding_provider = embedding_provider or embedding_service or get_embedding_provider()

    @property
    def model_name(self) -> str:
        name = getattr(self.embedding_provider, "model_name", None)
        if isinstance(name, str) and name:
            return name
        model = getattr(self.embedding_provider, "model", None)
        if isinstance(model, str) and model:
            return model
        return "gemini-embedding-001"

    def ingest_document(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Document:
        """
        Embed and persist a single document.
        """
        embedding = self.embedding_provider.embed_text(content)

        doc = Document(
            id=uuid.uuid4(),
            content=content.strip(),
            embedding=embedding,
            doc_metadata=metadata or {},
            embedding_model=self.model_name,
        )
        self.db.add(doc)
        self.db.flush()
        return doc

    def ingest_batch(
        self,
        items: List[DocumentIngestRequest],
    ) -> List[Document]:
        """
        Embed and persist a batch of documents in an optimized single embedding call.
        """
        if not items:
            return []

        contents = [item.content.strip() for item in items]
        embeddings = self.embedding_provider.embed_batch(contents)

        docs = []
        for item, emb in zip(items, embeddings):
            doc = Document(
                id=uuid.uuid4(),
                content=item.content.strip(),
                embedding=emb,
                doc_metadata=item.metadata or {},
                embedding_model=self.model_name,
            )
            docs.append(doc)

        self.db.add_all(docs)
        self.db.flush()
        return docs

    def get_document(self, doc_id: uuid.UUID) -> Optional[Document]:
        """Fetch an ingested document by its unique UUID."""
        return self.db.scalar(select(Document).where(Document.id == doc_id))

    def list_documents(
        self,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[Document], int]:
        """Retrieve paginated documents along with the total count."""
        total = self.db.scalar(select(func.count(Document.id))) or 0
        stmt = (
            select(Document)
            .order_by(Document.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        docs = list(self.db.scalars(stmt).all())
        return docs, total
