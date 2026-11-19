import json
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
    Includes duplicate detection to ensure rerun safety and idempotency.
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
        skip_duplicates: bool = False,
    ) -> Document:
        """
        Embed and persist a single document.
        If skip_duplicates is True and identical content already exists, returns the existing record.
        """
        cleaned = content.strip()
        if skip_duplicates:
            existing = self.db.scalar(select(Document).where(Document.content == cleaned))
            if existing:
                return existing

        embedding = self.embedding_provider.embed_text(cleaned)

        doc = Document(
            id=uuid.uuid4(),
            content=cleaned,
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
        skip_duplicates: bool = False,
    ) -> List[Document]:
        """
        Embed and persist a batch of documents in an optimized embedding call.
        Optionally skips documents whose content already exists in the database.
        """
        if not items:
            return []

        to_embed: List[DocumentIngestRequest] = []
        existing_docs: List[Document] = []

        for item in items:
            cleaned = item.content.strip()
            if skip_duplicates:
                existing = self.db.scalar(select(Document).where(Document.content == cleaned))
                if existing:
                    existing_docs.append(existing)
                    continue
            to_embed.append(item)

        new_docs: List[Document] = []
        if to_embed:
            contents = [item.content.strip() for item in to_embed]
            embeddings = self.embedding_provider.embed_batch(contents)

            for item, emb in zip(to_embed, embeddings):
                doc = Document(
                    id=uuid.uuid4(),
                    content=item.content.strip(),
                    embedding=emb,
                    doc_metadata=item.metadata or {},
                    embedding_model=self.model_name,
                )
                new_docs.append(doc)

            self.db.add_all(new_docs)
            self.db.flush()

        return existing_docs + new_docs

    def ingest_from_json_file(
        self,
        file_path: str,
        skip_duplicates: bool = True,
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """
        Read a JSON dataset file (e.g., data/sample_documents.json), extract contents and metadata,
        calculate 768-dim embeddings, and persist them into the database.
        Detects and skips duplicate content to remain idempotent across reruns.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            records = json.load(f)

        if not isinstance(records, list):
            raise ValueError(f"Expected a list of documents in {file_path}, got {type(records)}")

        inserted_count = 0
        skipped_count = 0
        inserted_docs: List[Document] = []

        to_process: List[Dict[str, Any]] = []
        for item in records:
            content = item.get("content", "").strip()
            if not content:
                continue
            metadata = item.get("metadata", {})
            if skip_duplicates:
                existing = self.db.scalar(select(Document).where(Document.content == content))
                if existing:
                    skipped_count += 1
                    continue
            to_process.append({"content": content, "metadata": metadata})

        # Process in batches for performance
        for i in range(0, len(to_process), batch_size):
            chunk = to_process[i : i + batch_size]
            contents = [c["content"] for c in chunk]
            embeddings = self.embedding_provider.embed_batch(contents)

            for c_item, emb in zip(chunk, embeddings):
                doc = Document(
                    id=uuid.uuid4(),
                    content=c_item["content"],
                    embedding=emb,
                    doc_metadata=c_item["metadata"] or {},
                    embedding_model=self.model_name,
                )
                self.db.add(doc)
                inserted_docs.append(doc)
                inserted_count += 1
            self.db.flush()

        self.db.commit()
        return {
            "total_records": len(records),
            "inserted": inserted_count,
            "skipped": skipped_count,
            "document_ids": [str(d.id) for d in inserted_docs],
        }

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
