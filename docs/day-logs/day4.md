# Day 4 — Daily Progress Log
**Date:** 21 November 2026  
**Track:** AI Engineering / Forward Deployed Engineer  
**Member:** Member B — AI Agents  
**Focus:** Document Chunking, Ingestion Pipeline Modernization, `document_chunks` Database Table, Hybrid Retrieval (RRF), Cross-Encoder Reranking, Context Engineering, Grounded RAG with Citations, Evaluation Benchmarking, and LangChain Teach-Back

---

## 1. Work Completed

### Phase 1 — Document Chunking Service & Experimentation
- Implemented `ChunkingService` in `app/services/chunking_service.py`:
  - Built regex text cleaner normalizing irregular whitespace, non-breaking spaces (`\xa0`), and linebreaks.
  - Implemented boundary-preserving hierarchical chunker using progressive separators (`\n\n` $\to$ `\n` $\to$ `[.!?]` $\to$ `" "`) ensuring chunks terminate on complete grammatical sentences without mid-word cuts.
  - Built token/word sliding window chunker (`split_by_words`).
  - Added strict parameter validation (`chunk_size > 0`, `0 <= chunk_overlap < chunk_size`).
  - Integrated metadata enrichment (`chunk_index`, `total_chunks`, `char_start`, `char_end`, `word_count`, `chunking_strategy`).
- Created an 8,320-character (1,101-word) multi-topic guide in `data/account_payments_subscriptions_guide.json` covering Account Security, Payments Architecture, and Subscription Tiers.
- Authored experimentation script `scripts/experiment_chunking.py`:
  - Compared 3 configurations:
    - **Config A (200 chars / 40 overlap):** 55 chunks, average 163.2 chars (21.8 words). High single-fact specificity.
    - **Config B (400 chars / 80 overlap):** 29 chunks, average 307.8 chars (40.8 words). Balanced context and optimal sentence preservation.
    - **Config C (800 chars / 150 overlap):** 14 chunks, average 603.5 chars (80.0 words). Macro contextual blocks.

### Phase 2 — Database Schema & Ingestion Pipeline Modernization
- Preserved existing `documents` table and its 31 whole-document records with zero destructive modifications.
- Created `DocumentChunk` model in `app/models/document_chunk.py` with foreign key relationship:
  - `document_id`: Foreign key referencing `documents.id` with `ondelete="CASCADE"`.
  - `chunk_index`: Integer positional index within the parent document.
  - `chunk_size`: Base character chunk size configuration.
  - `overlap`: Overlap size between adjacent chunks.
  - `content`: Chunk text content.
  - `embedding`: Vector(768) high-dimensional column.
  - `chunk_metadata`: JSONB attributes preserving parent metadata, char start/end offsets, and strategy.
  - `embedding_model`: `gemini-embedding-001`.
- Authored and applied Alembic migrations:
  - `b8b93b33f9c7_create_document_chunks_table.py`
  - `92b7963ec1b5_add_chunk_size_and_overlap_to_document_.py`
- Modernized `IngestionService` in `app/services/ingestion_service.py`:
  - Added `chunk_document_record`: splits and batch-embeds chunks via Gemini API.
  - Added `chunk_existing_document`: chunks an existing document by UUID with duplicate cleanup.
  - Added `ingest_document_with_chunks`: end-to-end ingestion storing parent in `documents` and chunks in `document_chunks`.
  - Added `list_chunks` and `get_chunk`.
- Created idempotent seeding runner `scripts/seed_chunks.py`:
  - Ingested the 1,101-word guide into `documents`.
  - Chunked and embedded all 32 parent documents into 60 chunks in `document_chunks` on Neon PostgreSQL.

### Phase 3 — Hybrid Search Retrieval & Route Deduplication
- Implemented chunk-level retrieval methods in `SearchService` (`app/services/search_service.py`):
  - `chunk_vector_search`: Embeds query via Gemini, executes cosine distance search via pgvector `<=>`.
  - `chunk_keyword_search`: Executes PostgreSQL full-text search using `to_tsvector` and `websearch_to_tsquery`.
  - `chunk_hybrid_search`: Merges vector and keyword candidate sets using **Reciprocal Rank Fusion (RRF)**:
    $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{vector}, \text{keyword}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
    with $k = 60$ and $\alpha = 0.5$, plus support for weighted linear score normalization.
- Canonical API Route Consolidation:
  - Enforced single endpoint per capability under `/v1/search/`:
    - `POST /v1/search/semantic` (adapted to search chunks automatically when chunks exist)
    - `POST /v1/search/keyword` (adapted to search chunks automatically when chunks exist)
    - `POST /v1/search/hybrid` (hybrid RRF search over document chunks with optional reranking)
  - Removed duplicate route aliases (`/chunks/vector`, `/chunks/semantic`, `/chunks/keyword`, `/chunks/hybrid`).
- Ingestion & Chunk Management Endpoints:
  - `POST /v1/documents/{document_id}/chunks`
  - `GET /v1/documents/{document_id}/chunks`
  - `GET /v1/documents/{document_id}/with-chunks`
  - `GET /v1/documents/chunks/{chunk_id}`
  - `POST /v1/documents/chunked`

### Phase 4 — Cross-Encoder Reranking
- Built `RerankingService` in `app/services/reranking_service.py`:
  - Cross-encoder heuristic scoring combining:
    1. Query token coverage (unigrams and bi-grams).
    2. Contiguous exact phrase matches.
    3. Token density and lexical proximity within chunk text.
    4. Prior vector/keyword fusion rank.
  - Added `compare_retrieval_with_and_without_reranking`: measures rank shifts, promotions, and demotions.

### Phase 5 — Context Engineering & Prompt Defense
- Implemented `ContextEngineeringService` in `app/services/context_service.py`:
  - **Deduplication:** Jaccard set-intersection deduplication dropping candidates with $>80\%$ token overlap.
  - **Budget Trimming:** Enforces a hard character ceiling (default 4,000 characters) while preserving source header overhead.
  - **Source Labeling:** Formats chunks with explicit citation anchors (`[Source 1] (Document ID: ..., Chunk: ...)`).
  - **Prompt Defense:** Instructs LLM to answer strictly from provided context and respond deterministically with `"I don't know based on the provided context."` when evidence is missing.

### Phase 6 — RAG Answer Generation Endpoint
- Implemented `RAGService` in `app/services/rag_service.py`:
  - Coordinates Retrieval $\to$ Reranking (optional) $\to$ Context Assembly $\to$ LLM Completion $\to$ Citation Verification.
  - Regex citation extraction supporting both `[Source N]` and `【Source N】` bracket notations.
  - Deterministic ignorance fallback when context similarity or lexical overlap is absent.
- Exposed `POST /v1/rag/ask`:
  - Request schema: `question`, `retrieval_strategy` (`hybrid`/`vector`/`keyword`), `rerank` (boolean), `top_k`, `provider`, `model`.
  - Response schema: `question`, `answer`, `sources` (with `document_id`, `chunk_id`, `chunk_index`, `content`, `relevance_score`), `citations`, `is_grounded`.

### Phase 7 — Evaluation Benchmark & Test Sheet
- Authored automated evaluation runner `scripts/evaluate_rag.py`.
- Evaluated **10 systematically designed test questions** (8 in-domain across security, billing, subscriptions, quotas + 2 out-of-domain ignorance test cases).
- Generated full evaluation report in `docs/rag-evaluation-report.md` and structured dataset in `docs/rag_evaluation_data.json`:
  - **Keyword Retrieval Relevance:** 12.5% (fails on conversational queries).
  - **Vector Retrieval Relevance:** 91.9% (robust semantic recall).
  - **Hybrid (RRF) Relevance:** 91.9% (combines semantic breadth with keyword anchoring).
  - **Reranked Relevance:** 86.2% (promotes high-density exact phrase sentences).
  - **Answer Grounding Score:** 78.4% (factual, zero ungrounded hallucinations).
  - **"I Don't Know" Precision:** 100.0% (both out-of-domain queries successfully triggered ignorance admission).

### Phase 8 — LangChain Teach-Back Document
- Authored comprehensive 45-minute teach-back guide in `docs/study/langchain-teach-back-guide.md`:
  - Deep-dive into LangChain Retrievers (`BaseRetriever`, `VectorStoreRetriever`, `EnsembleRetriever`, `ContextualCompressionRetriever`).
  - Deep-dive into LangChain Runnables & LCEL (`RunnablePassthrough`, `RunnableParallel`, LCEL `|` operator).
  - Direct side-by-side comparison with our plain Python implementation (`SearchService`, `RerankingService`, `ContextEngineeringService`, `RAGService`).
  - Production trade-offs: observability, debuggability, dependency weight, execution latency, and vendor lock-in.

---

## 2. Technical Decisions

- **Separate `document_chunks` Table:** Maintained strict relational separation between full parent documents and chunks. Preserves legacy records, enables source citations, and permits re-chunking without altering master document provenance.
- **Reciprocal Rank Fusion (RRF) over Weighted Linear Scoring:** Avoids fragile score normalization between bounded cosine similarity ($[0, 1]$) and unbounded PostgreSQL `ts_rank` ($[0, \infty)$). RRF rewards consensus candidates while gracefully handling single-branch retrieval when keyword search yields zero hits.
- **Hierarchical Boundary-Preserving Chunking:** Avoids naive character slicing that cuts words and sentences in half. Separator hierarchy (`\n\n` $\to$ `\n` $\to$ `[.!?]` $\to$ `" "`) ensures semantic coherence within each chunk.
- **Batch Embedding Optimization:** Chunk embeddings are generated in vectorized batch requests to Google Gemini, minimizing network overhead and avoiding per-chunk HTTP latency.
- **Plain Python Clean Architecture:** Decoupled business logic into single-responsibility domain services without framework bloat, ensuring full observability, clean stack traces, and direct SQL execution.

---

## 3. Testing and Validation

- **Automated Test Suite:**
  - Ran pytest suite with comprehensive coverage:
    - Unit tests for chunking service, regex cleaning, boundary preservation (`tests/unit/test_chunking.py`).
    - Unit tests for chunk ingestion, parent foreign key linkage, vector search, keyword search, and hybrid RRF retrieval (`tests/unit/test_chunk_search.py`).
    - Unit tests for reranking, context deduplication, budget trimming, prompt construction, and RAG answer endpoint (`tests/unit/test_rag.py`).
    - 100% backward compatibility maintained with Day 1–3 chat, prompt, document, and search tests.
- **Live Database Verifications:**
  - 32 parent documents and 60 chunk records verified in Neon PostgreSQL.
  - Live benchmark queries executed against Neon PostgreSQL confirming that Hybrid Search eliminates vocabulary mismatches while retaining technical precision.
  - Live RAG queries verified via Groq LLM returning verified source citations and 100% precision on out-of-domain queries.

---

## 4. Day 4 Outcome

The complete Day 4 RAG architecture is fully implemented, verified, and benchmarked:
1. Two-tier relational document-chunk persistence in PostgreSQL with foreign keys and pgvector.
2. Multi-strategy retrieval (Keyword, Vector, Hybrid RRF) and cross-encoder reranking.
3. Clean context engineering pipeline (deduplication, budget trimming, source labeling).
4. Grounded RAG answer endpoint with verified citations and robust hallucination suppression ("I don't know").
5. Exhaustive evaluation benchmark (10 test questions) and 45-minute LangChain teach-back technical guide.
