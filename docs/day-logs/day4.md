# Day 4 — Daily Progress Log
**Date:** 21 November 2026  
**Track:** AI Engineering / Forward Deployed Engineer  
**Member:** Member B — AI Agents  
**Focus:** Document Chunking, Ingestion Pipeline Modernization, `document_chunks` Database Table, 4 Large Document Corpus, Dedicated Search APIs (Unchunked vs. Chunked), Dynamic Database Prompt Templates, Information Retrieval Benchmarking (P@K, R@K, MRR, NDCG), and LangChain Teach-Back

---

## 1. Work Completed

### Phase 1 — Document Chunking Service & Metric Clarification
- **Chunk Size Clarification:**
  - Default chunk size is **400 characters** with an **80 character overlap** (~60–70 words per chunk in technical English prose).
  - Built boundary-preserving hierarchical chunker using progressive separators (`\n\n` $\to$ `\n` $\to$ `[.!?]` $\to$ `" "`) ensuring chunks terminate on complete sentences without slicing mid-word.
  - Also provides sliding window word chunking (`split_by_words`).
  - Added strict parameter validation (`chunk_size > 0`, `0 <= chunk_overlap < chunk_size`).
  - Integrated rich metadata enrichment (`chunk_index`, `total_chunks`, `char_start`, `char_end`, `word_count`, `chunking_strategy`).

### Phase 2 — 4 Large Comprehensive Documents & Database Purge
- **Corpus Cleanup:** Purged all 31 legacy snippet records from `sample_documents.json` and cleared old test chunks.
- **4 Large Comprehensive Enterprise Guides** in `data/account_payments_subscriptions_guide.json`:
  1. `doc-001-account-billing`: Account Management, Access Security, Payments, Invoicing, and Subscriptions (8,320 chars, 1,101 words).
  2. `doc-002-api-webhooks`: API Infrastructure, Rate Limiting, Authentication, and Webhooks Architecture (6,785 chars, 893 words).
  3. `doc-003-security-compliance`: Enterprise Security, Compliance, Audit Logging, and Data Governance (6,474 chars, 845 words).
  4. `doc-004-ai-vector-rag`: AI Model Deployment, Vector Search, Embedding Pipelines, and RAG (7,186 chars, 955 words).
- **Relational Persistence in Neon PostgreSQL:**
  - `documents`: Exactly 4 parent records embedded with Gemini 768-dim vectors.
  - `document_chunks`: Exactly 96 high-quality chunks with `document_id` foreign keys, `chunk_size=400`, `overlap=80`, and 768-dim `pgvector` embeddings.

### Phase 3 — Two Consolidated Search APIs
- Enforced strict two-endpoint search architecture in `app/api/v1/endpoints/search.py`:
  1. `POST /v1/search/semantic`: Executes semantic cosine similarity search strictly over stored **UNCHUNKED** documents (`documents` table).
  2. `POST /v1/search/chunked`: Executes search strictly over stored **DOCUMENT CHUNKS** (`document_chunks` table), supporting `strategy="hybrid"|"vector"|"keyword"` and optional cross-encoder reranking.
- Removed legacy duplicate `/keyword` and `/hybrid` routes from the router.

### Phase 4 & 5 — Cross-Encoder Reranking & Dynamic DB Prompts
- **Dynamic Prompt Resolution:**
  - Seeded category `rag` (version `v1` and `default`) into PostgreSQL `prompt_categories` and `prompt_versions`.
  - Updated `ContextEngineeringService` and `RAGService` to load the system prompt directly from the `prompts` table via `PromptService`.
- **Cross-Encoder Reranker:**
  - Heuristic scoring factoring query token coverage, exact contiguous n-grams, and lexical proximity density.

### Phase 6 — RAG Answer Generation Endpoint
- Exposed `POST /v1/rag/ask`:
  - Dynamically retrieves chunks from `document_chunks`.
  - Assembles deduplicated, budget-trimmed context with `[Source N]` tags.
  - Generates grounded completions via Groq LLM using the database prompt.
  - Provides 100% negative grounding ("I don't know") when evidence is absent.

### Phase 7 — Formal Information Retrieval (IR) Benchmark
- Executed benchmark runner `scripts/evaluate_chunking_and_retrieval.py` over 12 queries.
- Measured standard Information Retrieval metrics across 5 retrieval architectures:
  - **Chunked Vector & Hybrid RRF:** P@1 = **1.000**, P@3 = **0.849**, R@3 = **1.000**, MRR = **1.000**, MAP@5 = **0.946**, NDCG@5 = **0.993**.
  - **Unchunked Semantic:** P@1 = 0.909, P@3 = 0.546, MRR = 0.955, NDCG@5 = 0.831 (delivers ~7,200 chars/hit, causing severe context dilution).
  - **Chunked Keyword:** P@1 = 0.182, MRR = 0.182, NDCG@5 = 0.111 (fails on conversational vocabulary).
  - **Out-of-Domain Precision:** **100%** on ignorance queries ("I don't know based on the provided context").
- Published technical report in `docs/chunking-and-hybrid-search-report.md`.

### Phase 8 — LangChain Teach-Back Document
- Authored 45-minute teach-back guide in `docs/study/langchain-teach-back-guide.md` analyzing Retrievers, Runnables, LCEL, and our Plain Python Clean Architecture.

---

## 2. Technical Decisions & Findings

- **Context Noise Reduction (95.1%):** Unchunked retrieval returns an entire document (~7,200 chars) per hit, which floods the LLM with multi-topic noise. Chunking isolates the target paragraph (~350 chars), reducing prompt tokens while improving answer grounding.
- **Characters vs. Words:** 400 characters (~65 words) with 80 character overlap provides the ideal granularity for factual question-answering without splitting multi-sentence explanations.
- **RRF Consensus:** Reciprocal Rank Fusion ($k=60$) balances dense vector recall with exact sparse keyword precision without score normalization fragility.
- **Database-Driven Prompt Governance:** Storing prompt templates in PostgreSQL `prompt_categories` and `prompt_versions` allows runtime prompt updates without code deployments.

---

## 3. Testing and Live Database Verification

- **Automated Test Suite:**
  - **150 passed, 3 deselected in 14.48s** across all unit and API endpoint tests.
- **Neon PostgreSQL Production State:**
  - `documents`: 4 parent enterprise guide records.
  - `document_chunks`: 96 chunks with 768-dim embeddings.
  - `prompt_categories` & `prompt_versions`: active `rag/v1` grounding prompt template.
