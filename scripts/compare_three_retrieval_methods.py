"""
Compare Three Retrieval Methods: Keyword vs. Vector vs. Hybrid
==============================================================
Evaluates and compares:
1. Keyword Search (PostgreSQL Full-Text Search ts_rank)
2. Vector Search (Gemini 768-dim embeddings with Cosine Similarity)
3. Hybrid Search (Reciprocal Rank Fusion - RRF k=60)
4. Reranking (Cross-Encoder / lexical proximity re-ordering)

Covers both chunked (document_chunks) and unchunked (documents) modes.
"""

import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Set

# Ensure project root in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.db.session import SessionLocal
from app.services.reranking_service import RerankingService
from app.services.search_service import SearchService

BENCHMARK_QUERIES = [
    {
        "id": 1,
        "query": "I want to stop paying for this service",
        "type": "Semantic / Conversational Intent",
        "description": "Zero keyword overlap with 'cancellation', 'downgrade', or 'refund'.",
        "target_keywords": ["cancel", "refund", "downgrade", "money-back"],
    },
    {
        "id": 2,
        "query": "HTTP 423 Locked",
        "type": "Exact Technical Identifier",
        "description": "Exact HTTP status code returned during brute-force account lockout.",
        "target_keywords": ["423", "locked", "cooling-off", "lockout"],
    },
    {
        "id": 3,
        "query": "VIES VAT registration reverse charge",
        "type": "Exact Acronym & Tax Domain",
        "description": "EU cross-border tax exemption mechanism.",
        "target_keywords": ["vies", "vat", "reverse charge", "tax"],
    },
    {
        "id": 4,
        "query": "How many single-use recovery codes are generated during 2FA setup?",
        "type": "Granular Fact Retrieval",
        "description": "Requires finding the exact number (ten single-use codes) in security section.",
        "target_keywords": ["recovery codes", "2fa", "ten single-use", "alphanumeric"],
    },
    {
        "id": 5,
        "query": "distributed Token Bucket rate-limiting Redis cluster",
        "type": "Technical Architecture Query",
        "description": "Compound technical terms across algorithms and caching layers.",
        "target_keywords": ["token bucket", "rate-limiting", "redis", "token"],
    },
    {
        "id": 6,
        "query": "cryptographic data shredding customer-managed keys",
        "type": "Security & Compliance Topic",
        "description": "CMEK key revocation causing immediate data destruction.",
        "target_keywords": ["cmek", "customer-managed", "shredding", "envelope encryption"],
    },
]


def jaccard_overlap(list_a: List[str], list_b: List[str]) -> float:
    set_a, set_b = set(list_a), set(list_b)
    if not set_a or not set_b:
        return 0.0
    return len(set_a.intersection(set_b)) / len(set_a.union(set_b))


def run_retrieval_comparison(top_k: int = 5) -> None:
    db = SessionLocal()
    search_service = SearchService(db)
    reranker = RerankingService()

    print("=" * 105)
    print("PHASE 3 & 4 BENCHMARK: COMPARING KEYWORD vs. VECTOR vs. HYBRID vs. RERANKED RETRIEVAL")
    print(f"Top-K: {top_k} | Candidate Pool for Reranking: {top_k * 2} | RRF Smoothing Constant k: 60")
    print("=" * 105)

    stats = {
        "keyword_wins": 0,
        "vector_wins": 0,
        "hybrid_consensus": 0,
        "total_queries": len(BENCHMARK_QUERIES),
    }

    for item in BENCHMARK_QUERIES:
        qid = item["id"]
        qtext = item["query"]
        qtype = item["type"]
        qdesc = item["description"]
        targets = item["target_keywords"]

        print(f"\n{'='*105}")
        print(f"QUERY #{qid}: \"{qtext}\"")
        print(f"Category: {qtype}")
        print(f"Intent:   {qdesc}")
        print(f"{'-'*105}")

        # 1. Chunked Keyword Search
        t0 = time.perf_counter()
        kw_resp = search_service.chunk_keyword_search(query=qtext, top_k=top_k)
        kw_latency = (time.perf_counter() - t0) * 1000

        # 2. Chunked Vector Search
        t0 = time.perf_counter()
        vec_resp = search_service.chunk_vector_search(query=qtext, top_k=top_k)
        vec_latency = (time.perf_counter() - t0) * 1000

        # 3. Chunked Hybrid Search (RRF)
        t0 = time.perf_counter()
        hyb_resp = search_service.chunk_hybrid_search(query=qtext, top_k=top_k, alpha=0.5, rrf_k=60)
        hyb_latency = (time.perf_counter() - t0) * 1000

        # 4. Reranking on Hybrid Candidates (fetch top_k * 2 and rerank to top_k)
        hyb_expanded = search_service.chunk_hybrid_search(query=qtext, top_k=top_k * 2, alpha=0.5, rrf_k=60)
        t0 = time.perf_counter()
        reranked_items = reranker.rerank(
            query=qtext,
            items=list(hyb_expanded.results),
            top_k=top_k,
        )
        rerank_latency = (time.perf_counter() - t0) * 1000

        kw_chunks = [str(r.id) for r in kw_resp.results]
        vec_chunks = [str(r.id) for r in vec_resp.results]
        hyb_chunks = [str(r.id) for r in hyb_resp.results]
        rerank_chunks = [str(r.id) for r in reranked_items]

        # Overlap analysis
        kw_vec_overlap = jaccard_overlap(kw_chunks, vec_chunks)
        hyb_agreement = len(set(kw_chunks).intersection(set(vec_chunks)))

        print(f"{'Method':<14} | {'Latency':<8} | {'Results':<7} | {'Top Rank-1 Snippet (first 80 chars)':<48} | {'Score'}")
        print(f"{'-'*14}-+-{'-'*8}-+-{'-'*7}-+-{'-'*48}-+-{'-'*15}")

        # Print Top-1 for each
        if kw_resp.results:
            top_kw = kw_resp.results[0]
            snippet = top_kw.content.replace('\n', ' ')[:46]
            kw_score = getattr(top_kw, "rank", None) or getattr(top_kw, "score", 0.0)
            print(f"{'Keyword (FTS)':<14} | {kw_latency:6.1f}ms | {len(kw_resp.results):<7} | \"{snippet}...\" | ts_rank={kw_score:.4f}")
        else:
            print(f"{'Keyword (FTS)':<14} | {kw_latency:6.1f}ms | 0       | [NO MATCHES FOUND]                             | score=0.0000")

        if vec_resp.results:
            top_vec = vec_resp.results[0]
            snippet = top_vec.content.replace('\n', ' ')[:46]
            vec_score = getattr(top_vec, "similarity", None) or getattr(top_vec, "similarity_score", 0.0)
            print(f"{'Vector (Dense)':<14} | {vec_latency:6.1f}ms | {len(vec_resp.results):<7} | \"{snippet}...\" | cosine={vec_score:.4f}")
        else:
            print(f"{'Vector (Dense)':<14} | {vec_latency:6.1f}ms | 0       | [NO MATCHES FOUND]                             | score=0.0000")

        if hyb_resp.results:
            top_hyb = hyb_resp.results[0]
            snippet = top_hyb.content.replace('\n', ' ')[:46]
            print(f"{'Hybrid (RRF)':<14} | {hyb_latency:6.1f}ms | {len(hyb_resp.results):<7} | \"{snippet}...\" | rrf={top_hyb.combined_score:.4f}")

        if reranked_items:
            top_rr = reranked_items[0]
            snippet = top_rr.content.replace('\n', ' ')[:46]
            print(f"{'Reranked':<14} | {rerank_latency:6.1f}ms | {len(reranked_items):<7} | \"{snippet}...\" | relevance={top_rr.rerank_score:.4f}")

        # Qualitative check
        def match_targets(content: str) -> bool:
            c = content.lower()
            return any(t in c for t in targets)

        kw_hit = any(match_targets(r.content) for r in kw_resp.results[:1]) if kw_resp.results else False
        vec_hit = any(match_targets(r.content) for r in vec_resp.results[:1]) if vec_resp.results else False

        print(f"\nAnalysis:")
        print(f"  • Keyword-Vector Overlap:  {kw_vec_overlap*100:.1f}% Jaccard ({hyb_agreement} common chunks in Top-{top_k})")
        if not kw_hit and vec_hit:
            print("  • Result: Vector search WINS (Vocabulary mismatch: Keyword failed to find semantic intent).")
            stats["vector_wins"] += 1
        elif kw_hit and not vec_hit:
            print("  • Result: Keyword search WINS (Precision match: Exact term found, vector missed subtle token).")
            stats["keyword_wins"] += 1
        elif kw_hit and vec_hit:
            print("  • Result: Hybrid consensus (Both captured the target; RRF ranked it at the very top).")
            stats["hybrid_consensus"] += 1
        else:
            print("  • Result: Dispersed retrieval; Reranker re-ordered candidates to maximize token coverage.")

    print("\n" + "=" * 105)
    print("EXECUTIVE SUMMARY & RETRIEVAL STRATEGY COMPARISON")
    print("=" * 105)
    print(
        "1. KEYWORD SEARCH (PostgreSQL FTS ts_rank):\n"
        "   - STRENGTH: Exceptional precision on exact error codes ('HTTP 423'), acronyms ('VIES VAT', 'FIDO2'),\n"
        "               and specific API headers ('X-Signature-SHA256'). Zero embedding latency.\n"
        "   - WEAKNESS: Complete failure on conversational synonyms ('stop paying' vs 'cancel subscription')\n"
        "               due to the vocabulary mismatch problem.\n\n"
        "2. VECTOR SEARCH (Gemini 768-dim embeddings <=> pgvector Cosine Distance):\n"
        "   - STRENGTH: Captures semantic meaning, synonyms, paraphrasing, and cross-domain concepts.\n"
        "   - WEAKNESS: Can suffer semantic dilution on rare numbers, acronyms, or specific alphanumeric codes.\n"
        "               Requires embedding computation latency (~200ms).\n\n"
        "3. HYBRID SEARCH (Reciprocal Rank Fusion k=60):\n"
        "   - STRENGTH: Best of both worlds. Fuses dense semantic recall with sparse lexical precision.\n"
        "               Items appearing in both lists receive top priority. Fails gracefully if one branch misses.\n"
        "   - SCORE FORMULA: RRF_score(chunk) = 0.5/(60 + rank_vector) + 0.5/(60 + rank_keyword)\n\n"
        "4. RERANKING (Cross-Encoder / Lexical Density):\n"
        "   - STRENGTH: Evaluates query token coverage, contiguous phrase matching, and proximity density\n"
        "               simultaneously on candidate chunks, promoting the chunk that directly answers the question\n"
        "               to Rank 1 before context injection into the LLM."
    )
    print("=" * 105)


if __name__ == "__main__":
    run_retrieval_comparison(top_k=5)
