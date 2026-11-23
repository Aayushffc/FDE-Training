"""
Seed Document Chunks Runner
===========================
Ingests the comprehensive 1,000-word guide into the 'documents' table (if absent),
then chunks all stored documents into the new 'document_chunks' table using
ChunkingService and Gemini 768-dim embeddings.

Fully idempotent: detects existing chunks and preserves prior whole-document records.
"""

import json
import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import func, select

from app.db.session import SessionLocal
from app.infrastructure.embedding_providers import get_embedding_provider
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.services.chunking_service import ChunkingService
from app.services.ingestion_service import IngestionService


def seed_chunks(chunk_size: int = 400, chunk_overlap: int = 80):
    db = SessionLocal()
    embedder = get_embedding_provider()
    chunker = ChunkingService()
    ingestion = IngestionService(db=db, embedding_provider=embedder, chunking_service=chunker)

    print("=" * 80)
    print("PHASE 2: SEEDING DOCUMENT CHUNKS INTO NEON POSTGRESQL")
    print(f"Target Configuration: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")
    print("=" * 80)

    # 1. Ensure the 1,000-word comprehensive guide exists in 'documents'
    guide_path = Path(__file__).resolve().parent.parent / "data" / "account_payments_subscriptions_guide.json"
    if guide_path.exists():
        with open(guide_path, "r", encoding="utf-8") as f:
            guide_data = json.load(f)

        if isinstance(guide_data, list):
            doc_items = guide_data
        else:
            doc_items = [guide_data]

        for item in doc_items:
            content = item["content"].strip()
            meta = item.get("metadata", {})
            title = item.get("title", "Guide Document")
            meta["title"] = title
            existing = db.scalar(select(Document).where(Document.content == content))
            if not existing:
                print(f"\nIngesting '{title}' as a parent Document...")
                guide_doc = ingestion.ingest_document(
                    content=content,
                    metadata=meta,
                    skip_duplicates=True,
                )
                db.commit()
                print(f"  -> Created Document ID: {guide_doc.id}")
            else:
                print(f"\n'{title}' already stored as Document ID: {existing.id}")

    # 2. Iterate through all documents and create chunks
    all_docs = list(db.scalars(select(Document).order_by(Document.created_at.asc())).all())
    total_docs = len(all_docs)
    print(f"\nTotal Parent Documents in Database: {total_docs}")

    total_chunks_created = 0
    skipped_docs = 0
    chunked_docs = 0

    for idx, doc in enumerate(all_docs, start=1):
        # Check if chunks already exist for this document
        existing_chunk_count = db.scalar(
            select(func.count(DocumentChunk.id)).where(DocumentChunk.document_id == doc.id)
        ) or 0

        if existing_chunk_count > 0:
            skipped_docs += 1
            continue

        print(f"[{idx}/{total_docs}] Chunking Document {doc.id} (length: {len(doc.content)} chars)...")
        chunks = ingestion.chunk_existing_document(
            document_id=doc.id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="character",
            preserve_boundaries=True,
            replace_existing=False,
        )
        total_chunks_created += len(chunks)
        chunked_docs += 1
        print(f"  -> Generated & Embedded {len(chunks)} chunks.")

    total_db_chunks = db.scalar(select(func.count(DocumentChunk.id))) or 0

    print("\n" + "=" * 80)
    print("SEEDING SUMMARY:")
    print(f"  Parent Documents:       {total_docs}")
    print(f"  Documents Chunked:      {chunked_docs}")
    print(f"  Documents Skipped:      {skipped_docs} (already had chunks)")
    print(f"  New Chunks Created:     {total_chunks_created}")
    print(f"  Total Chunks in DB:     {total_db_chunks}")
    print("=" * 80)

    db.close()
    return total_db_chunks


if __name__ == "__main__":
    seed_chunks()
