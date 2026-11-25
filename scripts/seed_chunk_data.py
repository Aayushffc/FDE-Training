"""
Seed Chunk Data Runner
======================
Seeds `data/chunk_data.json` into both `documents` (parent) and `document_chunks` (chunks).
Cleans previous data from both tables before seeding.
Chunk size and overlap are configurable at the top of the script.
"""

import json
import sys
import time
from pathlib import Path

# ==========================================
# CONFIGURABLE PARAMETERS
# ==========================================
CHUNK_SIZE = 400      # Target character length per chunk
CHUNK_OVERLAP = 80    # Overlap characters between consecutive chunks
# ==========================================

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sqlalchemy import delete, func, select

from app.db.session import SessionLocal
from app.infrastructure.embedding_providers import get_embedding_provider
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.models.prompt import PromptCategory, PromptVersion
from app.services.chunking_service import ChunkingService
from app.services.ingestion_service import IngestionService

RAG_SYSTEM_PROMPT = (
    "You are a helpful, strictly grounded AI support assistant.\n"
    "Your instructions:\n"
    "1. Answer the user's question using ONLY the provided context.\n"
    "2. Even if the context does not contain a complete formal definition or exhaustive guide, synthesize and explain everything the context specifies regarding the entities, terms, processes, or capabilities mentioned in the user's prompt (e.g. how it is integrated, configured, or used).\n"
    "3. If the context contains NO relevant or related information whatsoever regarding the user's question, say exactly: 'I don't know based on the provided context.'\n"
    "4. Do NOT make assumptions, extrapolate, or introduce external knowledge not grounded in the context.\n"
    "5. Cite the supporting source(s) for each factual statement using bracketed references (e.g. [Source 1], [Source 2]).\n"
    "6. Be concise, professional, and factual."
)


def ensure_rag_prompt(db) -> None:
    """Ensure the RAG grounding prompt exists in PostgreSQL prompt tables."""
    cat = db.scalar(select(PromptCategory).where(PromptCategory.name == "rag"))
    if not cat:
        cat = PromptCategory(name="rag", description="Retrieval-Augmented Generation Grounding Prompts")
        db.add(cat)
        db.flush()

    for ver_name in ("v1", "default"):
        p_ver = db.scalar(
            select(PromptVersion).where(PromptVersion.category_id == cat.id, PromptVersion.version == ver_name)
        )
        if not p_ver:
            p_ver = PromptVersion(
                category_id=cat.id,
                version=ver_name,
                name=f"RAG Grounding Prompt ({ver_name})",
                prompt_text=RAG_SYSTEM_PROMPT,
                description="Strict grounding support prompt",
                is_active=True,
                variables=["context", "question"],
            )
            db.add(p_ver)
        else:
            p_ver.prompt_text = RAG_SYSTEM_PROMPT
            p_ver.is_active = True
    db.commit()


def seed_chunk_data(chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> None:
    db = SessionLocal()
    embedder = get_embedding_provider()
    chunker = ChunkingService()
    ingestion = IngestionService(db=db, embedding_provider=embedder, chunking_service=chunker)

    print("=" * 80)
    print("SEEDING DATA FROM data/chunk_data.json")
    print(f"Configuration: CHUNK_SIZE = {chunk_size} chars | CHUNK_OVERLAP = {chunk_overlap} chars")
    print("=" * 80)

    # 1. Ensure RAG grounding prompt exists
    print("\n[Step 1] Ensuring RAG Grounding Prompt exists in DB...")
    ensure_rag_prompt(db)
    print("  -> RAG prompt verified.")

    # 2. Clean previous data
    print("\n[Step 2] Cleaning previous data from 'document_chunks' and 'documents'...")
    del_chunks = db.execute(delete(DocumentChunk)).rowcount
    del_docs = db.execute(delete(Document)).rowcount
    db.commit()
    print(f"  -> Deleted {del_chunks} rows from document_chunks")
    print(f"  -> Deleted {del_docs} rows from documents")

    # 3. Load chunk_data.json
    data_path = Path(__file__).resolve().parent.parent / "data" / "chunk_data.json"
    if not data_path.exists():
        raise FileNotFoundError(f"File not found: {data_path}")

    with open(data_path, "r", encoding="utf-8") as f:
        documents_data = json.load(f)

    print(f"\n[Step 3] Ingesting {len(documents_data)} parent documents and generating chunk embeddings...")

    total_chunks_created = 0

    for idx, item in enumerate(documents_data, start=1):
        content = item["content"].strip()
        metadata = item.get("metadata", {})
        title = item.get("title", f"Document {idx}")
        metadata["title"] = title

        print(f"\n  [{idx}/{len(documents_data)}] Ingesting parent document: '{title[:60]}...'")
        doc = ingestion.ingest_document(
            content=content,
            metadata=metadata,
            skip_duplicates=False,
        )
        db.commit()
        print(f"      Parent Document ID: {doc.id} (length: {len(content)} chars)")

        # Pacing between documents to prevent rate limiting
        time.sleep(2)

        print(f"      Splitting into chunks (size={chunk_size}, overlap={chunk_overlap})...")
        chunks = ingestion.chunk_existing_document(
            document_id=doc.id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="character",
            preserve_boundaries=True,
            replace_existing=True,
        )
        total_chunks_created += len(chunks)
        print(f"      Created and embedded {len(chunks)} chunks.")

        # Pacing between chunk batches
        time.sleep(3)

    final_docs = db.scalar(select(func.count(Document.id))) or 0
    final_chunks = db.scalar(select(func.count(DocumentChunk.id))) or 0

    print("\n" + "=" * 80)
    print("SEEDING COMPLETED SUCCESSFULLY")
    print(f"Total Parent Documents in DB: {final_docs}")
    print(f"Total Document Chunks in DB:  {final_chunks}")
    print(f"Avg Chunks per Document:     {final_chunks / max(final_docs, 1):.1f}")
    print("=" * 80)


if __name__ == "__main__":
    seed_chunk_data(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
