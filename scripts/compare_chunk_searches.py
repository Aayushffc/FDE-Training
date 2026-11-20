"""
Chunk Search Comparison Runner
==============================
Executes 10 benchmark queries across Keyword Search, Vector Search, and Hybrid Search
(Reciprocal Rank Fusion) over DocumentChunk records in PostgreSQL.
Compares retrieval behavior, score distributions, and failure modes.
"""

import json
import sys
from pathlib import Path
from typing import Any, Dict, List

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal
from app.services.search_service import SearchService

QUERIES = [
    {
        "id": 1,
        "query": "I want to stop paying for this service",
        "category": "Semantic / Conversational Intent",
        "intent": "Find cancellation & refund policies using colloquial phrases with zero keyword overlap.",
        "expected_topic": "Cancellation, downgrade, or refund policy",
    },
    {
        "id": 2,
        "query": "HTTP 423 Locked",
        "category": "Exact Technical Identifier",
        "intent": "Target exact HTTP status code emitted during brute-force account lockout.",
        "expected_topic": "Account lockout cooling-off period",
    },
    {
        "id": 3,
        "query": "VIES VAT registration number",
        "category": "Exact Identifier & Acronym",
        "intent": "Target EU business tax exemption rules and reverse charge mechanism.",
        "expected_topic": "Tax exemption & invoice reverse charge",
    },
    {
        "id": 4,
        "query": "twenty percent discount for paying annually",
        "category": "Compound / Hybrid Query",
        "intent": "Find annual billing commitment incentives and tier pricing.",
        "expected_topic": "Annual commitment discounts",
    },
    {
        "id": 5,
        "query": "FIDO2 biometric passkeys and WebAuthn",
        "category": "Technical Security Terminology",
        "intent": "Locate modern passwordless multi-factor authentication methods.",
        "expected_topic": "2FA & Passkey implementation",
    },
    {
        "id": 6,
        "query": "smart retry schedule over fourteen-day recovery",
        "category": "Domain Specific Process",
        "intent": "Identify dunning management cycle for declined cards.",
        "expected_topic": "Card declines & dunning management",
    },
    {
        "id": 7,
        "query": "Forgot my login credentials and cannot enter",
        "category": "Semantic / Synonym Query",
        "intent": "Retrieve password reset guidance when user says 'credentials' instead of 'password'.",
        "expected_topic": "Password reset & identity verification",
    },
    {
        "id": 8,
        "query": "X-Signature-SHA256 webhook header",
        "category": "Exact Technical Header",
        "intent": "Exact match for HMAC signature verification on inbound webhooks.",
        "expected_topic": "Technical support webhook authentication",
    },
    {
        "id": 9,
        "query": "prorated billing adjustments for tier upgrade",
        "category": "Billing Domain Terminology",
        "intent": "Explain immediate plan tier upgrade billing mechanics.",
        "expected_topic": "Upgrades & prorated charges",
    },
    {
        "id": 10,
        "query": "cooling-off lockout period after failed attempts",
        "category": "Security Architecture",
        "intent": "Account protection against credential stuffing attacks.",
        "expected_topic": "Account lockout and cooling-off duration",
    },
]


def run_comparison() -> List[Dict[str, Any]]:
    db = SessionLocal()
    search_service = SearchService(db)

    print("=" * 90)
    print("PHASE 3: BENCHMARK RETRIEVAL COMPARISON OVER DOCUMENT CHUNKS")
    print("Evaluating Keyword (FTS) vs Vector (Gemini 768-dim) vs Hybrid (RRF k=60, alpha=0.5)")
    print("=" * 90)

    comparison_results: List[Dict[str, Any]] = []

    for item in QUERIES:
        q_id = item["id"]
        q_text = item["query"]
        category = item["category"]
        print(f"\n[{q_id}/10] Query: \"{q_text}\" ({category})")

        # 1. Keyword search (PostgreSQL FTS on chunks)
        kw_resp = search_service.chunk_keyword_search(query=q_text, top_k=3)
        top_kw = kw_resp.results[0] if kw_resp.results else None

        # 2. Vector search (Gemini 768-dim embeddings + pgvector <=> on chunks)
        vec_resp = search_service.chunk_vector_search(query=q_text, top_k=3)
        top_vec = vec_resp.results[0] if vec_resp.results else None

        # 3. Hybrid search (RRF on chunks)
        hyb_resp = search_service.chunk_hybrid_search(query=q_text, top_k=3, alpha=0.5, rrf_k=60)
        top_hyb = hyb_resp.results[0] if hyb_resp.results else None

        kw_summary = (
            f"Rank {top_kw.rank:.4f} | \"{top_kw.content[:85]}...\""
            if top_kw else "NO MATCH (ts_rank = 0)"
        )
        vec_summary = (
            f"Sim {top_vec.similarity:.4f} | \"{top_vec.content[:85]}...\""
            if top_vec else "NO MATCH"
        )
        hyb_summary = (
            f"Score {top_hyb.combined_score:.5f} ({top_hyb.matched_by}) | \"{top_hyb.content[:85]}...\""
            if top_hyb else "NO MATCH"
        )

        print(f"  [KEYWORD] : {kw_summary}")
        print(f"  [VECTOR]  : {vec_summary}")
        print(f"  [HYBRID]  : {hyb_summary}")

        comparison_results.append({
            "id": q_id,
            "query": q_text,
            "category": category,
            "intent": item["intent"],
            "expected_topic": item["expected_topic"],
            "keyword": {
                "matched": top_kw is not None,
                "score": top_kw.rank if top_kw else 0.0,
                "content_preview": top_kw.content[:120] if top_kw else None,
                "chunk_id": str(top_kw.id) if top_kw else None,
                "doc_id": str(top_kw.document_id) if top_kw else None,
            },
            "vector": {
                "matched": top_vec is not None,
                "score": top_vec.similarity if top_vec else 0.0,
                "content_preview": top_vec.content[:120] if top_vec else None,
                "chunk_id": str(top_vec.id) if top_vec else None,
                "doc_id": str(top_vec.document_id) if top_vec else None,
            },
            "hybrid": {
                "matched": top_hyb is not None,
                "score": top_hyb.combined_score if top_hyb else 0.0,
                "matched_by": top_hyb.matched_by if top_hyb else None,
                "vector_rank": top_hyb.vector_rank if top_hyb else None,
                "keyword_rank": top_hyb.keyword_rank if top_hyb else None,
                "content_preview": top_hyb.content[:120] if top_hyb else None,
                "chunk_id": str(top_hyb.id) if top_hyb else None,
                "doc_id": str(top_hyb.document_id) if top_hyb else None,
            },
        })

    db.close()
    return comparison_results


if __name__ == "__main__":
    run_comparison()
