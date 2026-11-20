import json
import uuid
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import delete, func, select
from sqlalchemy.orm import Session

from app.infrastructure.embedding_providers import BaseEmbeddingProvider, get_embedding_provider
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.schemas.document import DocumentIngestRequest
from app.services.chunking_service import ChunkingService

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
        chunking_service: Optional[ChunkingService] = None,
    ) -> None:
        self.db = db
        self.embedding_provider = embedding_provider or embedding_service or get_embedding_provider()
        self.chunking_service = chunking_service or ChunkingService()

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

    def chunk_document_record(
        self,
        document: Document,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        strategy: str = "character",
        preserve_boundaries: bool = True,
    ) -> List[DocumentChunk]:
        """
        Takes an existing Document record, cleans text, splits it into chunks,
        embeds each chunk in batch via the embedding provider, and creates DocumentChunk records.
        Preserves reference to parent document (document_id) and merges parent metadata.
        """
        raw_chunks = self.chunking_service.chunk_document(
            content=document.content,
            metadata=document.doc_metadata,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
            preserve_boundaries=preserve_boundaries,
        )
        if not raw_chunks:
            return []

        chunk_texts = [c["content"] for c in raw_chunks]
        embeddings = self.embedding_provider.embed_batch(chunk_texts)

        chunk_records: List[DocumentChunk] = []
        for c_data, emb in zip(raw_chunks, embeddings):
            chunk = DocumentChunk(
                id=uuid.uuid4(),
                document_id=document.id,
                chunk_index=c_data["chunk_index"],
                chunk_size=chunk_size,
                overlap=chunk_overlap,
                content=c_data["content"],
                embedding=emb,
                chunk_metadata=c_data["metadata"],
                embedding_model=self.model_name,
            )
            chunk_records.append(chunk)

        self.db.add_all(chunk_records)
        self.db.flush()
        return chunk_records

    def chunk_existing_document(
        self,
        document_id: uuid.UUID,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        strategy: str = "character",
        preserve_boundaries: bool = True,
        replace_existing: bool = True,
    ) -> List[DocumentChunk]:
        """
        Chunk an existing stored document by its UUID.
        Optionally replaces previous chunks for this document.
        """
        doc = self.get_document(document_id)
        if not doc:
            raise ValueError(f"Document with ID {document_id} not found.")

        if replace_existing:
            self.db.execute(delete(DocumentChunk).where(DocumentChunk.document_id == document_id))
            self.db.flush()

        chunks = self.chunk_document_record(
            document=doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
            preserve_boundaries=preserve_boundaries,
        )
        self.db.commit()
        return chunks

    def ingest_document_with_chunks(
        self,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        strategy: str = "character",
        preserve_boundaries: bool = True,
        skip_duplicates: bool = False,
    ) -> Tuple[Document, List[DocumentChunk]]:
        """
        End-to-end chunked ingestion:
        1. Ingest/store parent document in 'documents' table.
        2. Clean text and split document into chunks.
        3. Generate embeddings for each chunk using Gemini.
        4. Store chunks, embeddings, and metadata in 'document_chunks' table.
        Preserves foreign key to the parent document.
        """
        doc = self.ingest_document(
            content=content,
            metadata=metadata,
            skip_duplicates=skip_duplicates,
        )

        # Check if chunks already exist (e.g. if skip_duplicates found existing doc)
        existing_chunks = list(
            self.db.scalars(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == doc.id)
                .order_by(DocumentChunk.chunk_index.asc())
            ).all()
        )
        if existing_chunks and skip_duplicates:
            return doc, existing_chunks

        chunks = self.chunk_document_record(
            document=doc,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy=strategy,
            preserve_boundaries=preserve_boundaries,
        )
        self.db.commit()
        return doc, chunks

    def get_chunk(self, chunk_id: uuid.UUID) -> Optional[DocumentChunk]:
        """Fetch an individual chunk by its UUID."""
        return self.db.scalar(select(DocumentChunk).where(DocumentChunk.id == chunk_id))

    def list_chunks(
        self,
        document_id: Optional[uuid.UUID] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> Tuple[List[DocumentChunk], int]:
        """Retrieve paginated chunks, optionally filtered by parent document_id."""
        stmt = select(DocumentChunk)
        count_stmt = select(func.count(DocumentChunk.id))

        if document_id is not None:
            stmt = stmt.where(DocumentChunk.document_id == document_id)
            count_stmt = count_stmt.where(DocumentChunk.document_id == document_id)

        total = self.db.scalar(count_stmt) or 0
        stmt = (
            stmt.order_by(DocumentChunk.chunk_index.asc(), DocumentChunk.created_at.desc())
            .offset(offset)
            .limit(limit)
        )
        chunks = list(self.db.scalars(stmt).all())
        return chunks, total

    def chunk_all_existing_documents(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        strategy: str = "character",
        preserve_boundaries: bool = True,
        replace_existing: bool = False,
    ) -> Dict[str, Any]:
        """
        Backfills chunks for all documents currently in the 'documents' table.
        Rerun-safe: can skip documents that already have chunks.
        """
        docs = list(self.db.scalars(select(Document)).all())
        total_created = 0
        processed_docs = 0

        for doc in docs:
            if not replace_existing:
                existing_count = self.db.scalar(
                    select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == doc.id)
                )
                if existing_count and existing_count > 0:
                    continue

            chunks = self.chunk_existing_document(
                document_id=doc.id,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                strategy=strategy,
                preserve_boundaries=preserve_boundaries,
                replace_existing=replace_existing,
            )
            total_created += len(chunks)
            processed_docs += 1

        self.db.commit()
        return {
            "total_documents": len(docs),
            "processed_documents": processed_docs,
            "total_chunks_created": total_created,
        }

