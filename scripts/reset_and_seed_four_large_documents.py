"""
Reset & Seed 4 Large Documents and RAG Prompt
============================================
1. Seeds the RAG system prompt into prompt_categories and prompt_versions.
2. Clears legacy sample_documents from 'documents' and clears 'document_chunks'.
3. Ingests the 4 comprehensive enterprise guides from data/account_payments_subscriptions_guide.json.
4. Chunks each guide using ChunkingService (chunk_size=400, overlap=80, boundary preservation)
   and generates Gemini 768-dim embeddings for all chunks.
"""

import json
import sys
from pathlib import Path

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
    "2. If the answer is not available in the context, say exactly: 'I don't know based on the provided context.'\n"
    "3. Do NOT make assumptions, extrapolate, or use outside knowledge.\n"
    "4. Cite the source(s) supporting each statement using bracketed references (e.g. [Source 1], [Source 2]).\n"
    "5. Be concise, professional, and factual."
)


def seed_rag_prompt(db):
    """Seed RAG grounding prompt into PostgreSQL prompts table."""
    print("\n--- Seeding RAG Grounding Prompt into prompt_categories & prompt_versions ---")
    cat = db.scalar(select(PromptCategory).where(PromptCategory.name == "rag"))
    if not cat:
        cat = PromptCategory(name="rag", description="Retrieval-Augmented Generation Grounding Prompts")
        db.add(cat)
        db.flush()
        print(f"Created PromptCategory: 'rag' (ID: {cat.id})")
    else:
        print(f"Found existing PromptCategory: 'rag' (ID: {cat.id})")

    ver_v1 = db.scalar(
        select(PromptVersion).where(PromptVersion.category_id == cat.id, PromptVersion.version == "v1")
    )
    if not ver_v1:
        ver_v1 = PromptVersion(
            category_id=cat.id,
            version="v1",
            name="Strict Grounding RAG Support Assistant",
            prompt_text=RAG_SYSTEM_PROMPT,
            description="Strict grounding support assistant with negative citation defense",
            is_active=True,
            variables=["context", "question"],
        )
        db.add(ver_v1)
        print("Created PromptVersion: 'rag/v1'")
    else:
        ver_v1.prompt_text = RAG_SYSTEM_PROMPT
        ver_v1.is_active = True
        print("Updated existing PromptVersion: 'rag/v1'")

    ver_def = db.scalar(
        select(PromptVersion).where(PromptVersion.category_id == cat.id, PromptVersion.version == "default")
    )
    if not ver_def:
        ver_def = PromptVersion(
            category_id=cat.id,
            version="default",
            name="Default RAG Grounding Prompt",
            prompt_text=RAG_SYSTEM_PROMPT,
            description="Default RAG grounding prompt",
            is_active=True,
            variables=["context", "question"],
        )
        db.add(ver_def)
        print("Created PromptVersion: 'rag/default'")
    else:
        ver_def.prompt_text = RAG_SYSTEM_PROMPT
        ver_def.is_active = True
        print("Updated existing PromptVersion: 'rag/default'")

    db.commit()


def reset_and_seed_four_large_documents(chunk_size: int = 400, chunk_overlap: int = 80):
    db = SessionLocal()
    embedder = get_embedding_provider()
    chunker = ChunkingService()
    ingestion = IngestionService(db=db, embedding_provider=embedder, chunking_service=chunker)

    print("=" * 80)
    print("STEP 1: SEEDING RAG PROMPT")
    seed_rag_prompt(db)

    print("\n" + "=" * 80)
    print("STEP 2: CLEANING TABLES (DELETING SAMPLE_DOCUMENTS & OLD CHUNKS)")
    # Delete all chunks
    del_chunks = db.execute(delete(DocumentChunk)).rowcount
    print(f"Deleted {del_chunks} rows from 'document_chunks'")
    # Delete all documents
    del_docs = db.execute(delete(Document)).rowcount
    print(f"Deleted {del_docs} rows from 'documents'")
    db.commit()

    print("\n" + "=" * 80)
    print("STEP 3: LOADING 4 LARGE COMPREHENSIVE DOCUMENTS")
    guide_path = Path(__file__).resolve().parent.parent / "data" / "account_payments_subscriptions_guide.json"
    with open(guide_path, "r", encoding="utf-8") as f:
        doc_list = json.load(f)

    print(f"Found {len(doc_list)} comprehensive documents to ingest.")

    total_chunks_created = 0
    created_parent_docs = []

    for idx, item in enumerate(doc_list, start=1):
        title = item.get("title", f"Document {idx}")
        content = item["content"].strip()
        metadata = item.get("metadata", {})
        metadata["title"] = title

        char_len = len(content)
        word_len = len(content.split())
        print(f"\n[{idx}/4] Ingesting & Embedding Parent Document: '{title}'")
        print(f"     Size: {char_len} chars, {word_len} words")

        # Ingest into documents table with Gemini 768-dim embedding
        parent_doc = ingestion.ingest_document(
            content=content,
            metadata=metadata,
            skip_duplicates=False,
        )
        db.commit()
        db.refresh(parent_doc)
        created_parent_docs.append(parent_doc)
        print(f"     -> Stored Parent Document ID: {parent_doc.id}")

        # Chunk and embed into document_chunks
        chunks = ingestion.chunk_existing_document(
            document_id=parent_doc.id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            strategy="character",
            preserve_boundaries=True,
            replace_existing=True,
        )
        total_chunks_created += len(chunks)
        print(f"     -> Generated & Persisted {len(chunks)} Chunks into 'document_chunks'")

    # Final Verification
    final_docs_count = db.scalar(select(func.count(Document.id))) or 0
    final_chunks_count = db.scalar(select(func.count(DocumentChunk.id))) or 0

    print("\n" + "=" * 80)
    print("FINAL DATABASE VERIFICATION:")
    print(f"  Parent Documents in 'documents':        {final_docs_count}")
    print(f"  Total Chunks in 'document_chunks':     {final_chunks_count}")
    print(f"  Average Chunks per Parent Document:    {final_chunks_count / max(1, final_docs_count):.1f}")
    print("=" * 80)

    db.close()
    return final_docs_count, final_chunks_count


if __name__ == "__main__":
    reset_and_seed_four_large_documents()
