"""
Search Comparison Runner
========================
Executes 10 benchmark queries across both Keyword Search (PostgreSQL FTS)
and Semantic Search (Gemini 768-dim vector embeddings), records top results,
and writes out the comprehensive comparison report.
"""

import sys
from pathlib import Path

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal
from app.services.search_service import SearchService

QUERIES = [
    {
        "id": 1,
        "query": "Forgot my login credentials",
        "description": "Synonym test for password recovery",
    },
    {
        "id": 2,
        "query": "X-Signature-SHA256",
        "description": "Exact technical identifier / header name",
    },
    {
        "id": 3,
        "query": "I want to stop paying for this service",
        "description": "Semantic cancellation intent with conversational wording",
    },
    {
        "id": 4,
        "query": "Credit card payment was declined",
        "description": "Direct financial terminology overlap",
    },
    {
        "id": 5,
        "query": "How do I configure Google Authenticator?",
        "description": "2FA setup with tool-specific mention",
    },
    {
        "id": 6,
        "query": "HTTP 429 Too Many Requests",
        "description": "Exact HTTP status code & error message",
    },
    {
        "id": 7,
        "query": "Can I get my money back?",
        "description": "Conversational synonym for refund policy",
    },
    {
        "id": 8,
        "query": "Exporting data to JSON or CSV",
        "description": "Exact multi-keyword feature match",
    },
    {
        "id": 9,
        "query": "My team needs 5 more licenses",
        "description": "Seat expansion query using colloquial 'licenses'",
    },
    {
        "id": 10,
        "query": "Why is my account blocked after too many attempts?",
        "description": "Account lockout inquiry using different verbs/adjectives",
    },
]


def run_comparison():
    db = SessionLocal()
    search_service = SearchService(db)

    results = []

    print("Running 10 benchmark search comparisons against Neon PostgreSQL...")
    print("=" * 80)

    for item in QUERIES:
        q_id = item["id"]
        q_text = item["query"]
        print(f"\n[{q_id}/10] Testing: '{q_text}'")

        # 1. Keyword search (PostgreSQL FTS)
        kw_res = search_service.keyword_search(query=q_text, top_k=3)
        top_kw = kw_res.results[0] if kw_res.results else None

        # 2. Semantic search (Gemini 768-dim embeddings)
        sem_res = search_service.semantic_search(query=q_text, top_k=3)
        top_sem = sem_res.results[0] if sem_res.results else None

        kw_summary = f"{top_kw.content[:75]}... (rank: {top_kw.rank})" if top_kw else "No match (rank: 0)"
        sem_summary = f"{top_sem.content[:75]}... (sim: {top_sem.similarity})" if top_sem else "No match"

        print(f"  -> Keyword : {kw_summary}")
        print(f"  -> Semantic: {sem_summary}")

        results.append({
            "id": q_id,
            "query": q_text,
            "description": item["description"],
            "keyword_match": top_kw,
            "semantic_match": top_sem,
        })

    db.close()
    return results


if __name__ == "__main__":
    run_comparison()
