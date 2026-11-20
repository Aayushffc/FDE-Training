# Document Chunking, Ingestion Pipeline & Hybrid Search Evaluation Report

**Track:** AI Engineering / Forward Deployed Engineer (FDE)  
**Date:** 21 November 2026  
**Focus:** Document Chunking, Ingestion Pipeline Modernization, Neon PostgreSQL `document_chunks` Table, Gemini 768-Dim Vector Embeddings, PostgreSQL Full-Text Search, and Reciprocal Rank Fusion (RRF) Hybrid Retrieval.

---

## Executive Summary

As document lengths scale from short FAQ blurbs to multi-page guides, embedding entire documents into single vector embeddings leads to **semantic dilution**, **poor retrieval precision**, and **loss of exact section context**. 

In this phase, we modernized the ingestion and retrieval architecture:
1. **Document Chunking Service:** Built a zero-dependency, boundary-aware hierarchical text chunker with configurable chunk size, chunk overlap, and rich metadata enrichment.
2. **Chunking Experiments:** Tested three configurations (Fine-Grained 200/40, Balanced 400/80, Contextual 800/150) across an 8,320-character (1,101-word) guide covering Account Management, Payments, and Subscriptions.
3. **Clean Database Architecture:** Created a dedicated `document_chunks` table in Neon PostgreSQL linked to the original `documents` table via a `document_id` foreign key (`ON DELETE CASCADE`), ensuring backward compatibility with earlier whole-document records.
4. **Upgraded Ingestion Pipeline:** Automated end-to-end cleaning, splitting, Gemini 768-dimensional batch embedding, and idempotent storage.
5. **Hybrid Retrieval with Reciprocal Rank Fusion (RRF):** Implemented chunk-level Vector Search (pgvector `<=>`), Keyword Search (PostgreSQL `tsvector`/`ts_rank`), and Hybrid Search (RRF with $k=60, \alpha=0.5$).
6. **Live Benchmark Evaluation:** Evaluated 10 benchmark queries against Neon PostgreSQL, demonstrating that Hybrid Search eliminates vocabulary mismatches while retaining technical precision.

---

## 1. Architectural Evolution

```
========================================================================================
PREVIOUS ARCHITECTURE (Whole-Document Ingestion)
========================================================================================
Document (Raw Text)
       │
       ▼
Google Gemini API (gemini-embedding-001)
       │
       ▼
One 768-Dim Embedding
       │
       ▼
Neon PostgreSQL: 'documents' table
----------------------------------------------------------------------------------------
Limitation: A 1,000-word document covering accounts, billing, and subscriptions
averages out all topics into one vector. Search queries for specific sub-topics
(e.g., "VAT reverse charge exemption") suffer from low cosine similarity.
========================================================================================

========================================================================================
CURRENT ARCHITECTURE (Chunk-Level Ingestion & Retrieval)
========================================================================================
Parent Document (Full Text) ──► Stored in 'documents' table (Preserved & Intact)
       │
       ▼
Text Cleaning & Normalization (app.services.chunking_service)
       │
       ▼
Boundary-Preserving Hierarchical Chunking
       │
   ┌───┴───────────────────────┬───────────────────────┐
   ▼                           ▼                       ▼
Chunk 1                     Chunk 2                 Chunk 3 ... (N Chunks)
(Index 0, Chars 0-396)      (Index 1, Chars 310-705)(Index 2, Chars 625-1010)
   │                           │                       │
   ▼                           ▼                       ▼
Gemini Embeddings Batch     Gemini Embeddings Batch Gemini Embeddings Batch
(768-dim Vector)            (768-dim Vector)        (768-dim Vector)
   │                           │                       │
   └───────────────────────────┼───────────────────────┘
                               ▼
        Neon PostgreSQL: 'document_chunks' table
        ├─ id: UUID (PK)
        ├─ document_id: UUID (FK -> documents.id ON DELETE CASCADE)
        ├─ chunk_index: Integer
        ├─ content: Text
        ├─ embedding: VECTOR(768)
        ├─ metadata: JSONB (chunk & parent attributes)
        └─ embedding_model: 'gemini-embedding-001'
========================================================================================
```

---

## 2. Core Concepts

| Concept | Definition & Implementation in System |
| :--- | :--- |
| **Chunk Size** | The maximum target volume of text contained within a single chunk (e.g., 200, 400, or 800 characters). Controls the semantic granularity of the vector representation. |
| **Chunk Overlap** | The shared character window repeated between adjacent chunks (e.g., 40, 80, or 150 characters). Prevents semantic severance and context loss when key facts span chunk boundaries. |
| **Chunk Metadata** | Positional and relational context preserved in JSONB: `document_id`, `chunk_index`, `total_chunks`, `char_start`, `char_end`, `word_count`, `category`, and parent document source tags. Enables precise source citations in RAG generation. |
| **Foreign Key Reference** | `document_chunks.document_id` references `documents.id`. Ensures parent documents remain the authoritative source while allowing multiple chunking strategies or re-chunking without data duplication. |

---

## 3. Phase 1 — Chunking Experiments & Configuration Analysis

We evaluated three distinct chunking configurations against our curated 8,320-character (1,101-word) multi-topic document covering:
- **Section 1:** Account Management & Access Security (Registration, entropy, 2FA, lockout, RBAC).
- **Section 2:** Payments, Invoicing & Billing Architecture (Payment methods, dunning, VAT reverse charge, refunds).
- **Section 3:** Subscriptions, Tiers & Enterprise Licensing (Starter/Pro/Enterprise, proration, seat pools, SLAs).

### Experimental Results

| Metric | Config A (Fine-Grained) | Config B (Balanced) | Config C (Contextual) |
| :--- | :---: | :---: | :---: |
| **Chunk Size Setting** | 200 characters | 400 characters | 800 characters |
| **Chunk Overlap Setting** | 40 characters | 80 characters | 150 characters |
| **Total Chunks Produced** | **55 chunks** | **29 chunks** | **14 chunks** |
| **Average Character Length** | 163.2 chars | 307.8 chars | 603.5 chars |
| **Min / Max Characters** | 62 / 200 chars | 153 / 396 chars | 304 / 792 chars |
| **Average Word Count** | 21.8 words | 40.8 words | 80.0 words |
| **Min / Max Word Count** | 7 / 30 words | 16 / 51 words | 38 / 105 words |
| **Boundary Preservation** | Sentence fragments | Full clean sentences | Complete paragraphs |
| **Retrieval Specificity** | Highest (single fact) | Optimal balance | Lower (contains extra context) |

### Key Findings & Trade-offs

1. **Config A (200 / 40 — Fine-Grained):**
   - *Strengths:* Exceptional precision for single-fact extraction (e.g. retrieving only the exact 30-minute lockout rule).
   - *Weaknesses:* Loses surrounding conversational context. Generates 55 embeddings for a single document, increasing Gemini API quota consumption and database vector index footprint.
2. **Config B (400 / 80 — Balanced) [Selected Production Default]:**
   - *Strengths:* Perfectly accommodates 2–4 full sentences without fragmenting phrases. Preserves complete thought units while remaining compact enough for high semantic density in vector space.
   - *Embedding Efficiency:* Produced 29 chunks (a 47% reduction in API calls compared to Config A).
3. **Config C (800 / 150 — Contextual):**
   - *Strengths:* Captures multi-paragraph explanations and complete tables/tiers.
   - *Weaknesses:* For focused queries (e.g., "What is the penalty for failed logins?"), the chunk includes extraneous details about session termination and password complexity, introducing noise to downstream LLM prompts.

---

## 4. Phase 2 — Ingestion Pipeline Modernization

The ingestion pipeline (`app/services/ingestion_service.py`) was restructured into five deterministic steps:

1. **Load Original Document:** The parent document is received or loaded and stored in the `documents` table.
2. **Clean Text:** Whitespace, non-breaking characters (`\xa0`), carriage returns (`\r\n`), and duplicate empty lines are normalized.
3. **Hierarchical Boundary Chunking:** The document is split using priority separators (`\n\n` $\to$ `\n` $\to$ sentence ends `[.!?]` $\to$ spaces `" "`).
4. **Batch Vector Embedding:** Chunks are grouped and embedded via Google Gemini (`gemini-embedding-001`, 768 dimensions) in unified batch calls, optimizing network round-trips.
5. **Relational Storage:** Chunks are persisted to `document_chunks` with `document_id` referencing the parent record, alongside positional and categorical metadata.

### Database Schema Migration

Alembic migration `b8b93b33f9c7_create_document_chunks_table.py` was applied to Neon PostgreSQL:

```sql
CREATE TABLE document_chunks (
    id UUID PRIMARY KEY,
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    embedding VECTOR(768) NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'gemini-embedding-001',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW() NOT NULL
);

CREATE INDEX ix_document_chunks_doc_chunk_idx ON document_chunks(document_id, chunk_index);
CREATE INDEX ix_document_chunks_document_id ON document_chunks(document_id);
```

---

## 5. Phase 3 — Retrieval Paradigms: Keyword vs Vector vs Hybrid

We implemented and compared three distinct retrieval modes over `document_chunks`:

### 1. Keyword Search (PostgreSQL Full-Text Search)
- **Engine:** PostgreSQL `to_tsvector('english', DocumentChunk.content)` and `websearch_to_tsquery('english', query)`.
- **Scoring:** `ts_rank` evaluating term frequency and token proximity.
- **When it excels:** Exact technical identifiers (`HTTP 423`, `X-Signature-SHA256`), acronyms (`VIES`, `VAT`, `TOTP`), error codes, and exact SKU names.
- **When it fails:** Synonyms, conversational phrasing, morphological variance, or queries with zero keyword overlap (`"stop paying"` vs `"cancel subscription"`).

### 2. Vector Search (Gemini 768-Dim Semantic Retrieval)
- **Engine:** Neon PostgreSQL `pgvector` with cosine distance operator `<=>`.
- **Scoring:** Normalized cosine similarity ($1.0 - \text{distance}$).
- **When it excels:** Resolving vocabulary mismatch, conceptual associations, conversational intent, multilingual queries, and intent abstraction.
- **When it fails:** Exact version numbers, technical headers, or rare alphanumeric strings where the embedding vector blurs character-level distinctions.

### 3. Hybrid Search with Reciprocal Rank Fusion (RRF)
- **Formula:**
  $$\text{RRF\_Score}(d) = \sum_{m \in \{\text{vector}, \text{keyword}\}} \frac{w_m}{k + \text{rank}_m(d)}$$
  Where $k = 60$ (standard smoothing constant), $w_{\text{vector}} = \alpha = 0.5$, and $w_{\text{keyword}} = 1 - \alpha = 0.5$.
- **Why RRF over Linear Score Normalization:**
  - Cosine similarity is bounded between $[0, 1]$, while PostgreSQL `ts_rank` is an unbounded float dependent on document length and term frequency. Direct linear weighting ($\alpha \cdot S_{\text{vec}} + (1-\alpha) \cdot S_{\text{kw}}$) requires fragile normalization heuristics.
  - RRF operates purely on ordinal ranks, making it immune to differing score distributions across retrieval backends.
- **Behavior:**
  - **Consensus Items:** When both vector and keyword find a chunk in rank #1, its score is $\frac{0.5}{61} + \frac{0.5}{61} = 0.01639$ (maximum score).
  - **Single-Branch Fallback:** If keyword search finds 0 matches (e.g. synonym query), the chunk still receives $\frac{0.5}{61} = 0.00820$ from vector search, ensuring zero empty results.

---

## 6. Live Benchmark Comparison (10 Real Queries on Neon PostgreSQL)

The benchmark runner `scripts/compare_chunk_searches.py` was executed directly against the 60 chunks stored in Neon PostgreSQL:

| # | Query | Category | Keyword Rank | Vector Sim | Hybrid Score & Match Mode | Top Retrieved Chunk Snippet |
| :-: | :--- | :--- | :---: | :---: | :---: | :--- |
| **1** | *"I want to stop paying for this service"* | Semantic / Conversational | **0.0000 (No Match)** | **0.6834** | **0.00820 (vector_only)** | *"You can cancel your paid subscription at any time by going to Billing Settings > Subs..."* |
| **2** | *"HTTP 423 Locked"* | Exact Technical Identifier | **0.2683** | **0.6912** | **0.01639 (both)** | *"The default lockout duration is exactly thirty minutes. During this period, all inbou..."* |
| **3** | *"VIES VAT registration number"* | Exact Identifier & Acronym | **0.4636** | **0.6338** | **0.01639 (both)** | *"European Union businesses possessing a valid VIES VAT registration number can input t..."* |
| **4** | *"twenty percent discount for paying annually"* | Compound / Synonym Intent | **0.0000 (No Match)** | **0.7914** | **0.00820 (vector_only)** | *"Annual billing plans provide an upfront 20% discount compared to monthly subscription..."* |
| **5** | *"FIDO2 biometric passkeys and WebAuthn"* | Technical Security Terminology | **0.4613** | **0.7278** | **0.01639 (both)** | *"During enrollment, a secret seed key is generated alongside ten single-use alphanumer..."* |
| **6** | *"smart retry schedule over fourteen-day recovery"* | Domain Process | **0.9810** | **0.7775** | **0.01639 (both)** | *"In the event of a declined charge—whether caused by insufficient funds, invalid expir..."* |
| **7** | *"Forgot my login credentials and cannot enter"* | Semantic / Paraphrasing | **0.0000 (No Match)** | **0.6751** | **0.00820 (vector_only)** | *"To reset your password, open the login page and select Forgot Password. Follow the in..."* |
| **8** | *"X-Signature-SHA256 webhook header"* | Exact Technical Header | **0.8655** | **0.7719** | **0.01639 (both)** | *"Webhook deliveries are signed using HMAC-SHA256 signatures passed in the 'X-Signature..."* |
| **9** | *"prorated billing adjustments for tier upgrade"* | Billing Domain Terminology | **0.8547** | **0.8653** | **0.01639 (both)** | *"Upgrades, Downgrades, and Prorated Billing Adjustments Customers may upgrade or downg..."* |
| **10** | *"cooling-off lockout period after failed attempts"* | Security Architecture | **0.8947** | **0.8119** | **0.01639 (both)** | *"Account Lockout, Cooling-Off Periods, and Session Invalidation To thwart brute-force ..."* |

---

## 7. Detailed Analysis & Key Takeaways

### Case 1: Vocabulary Mismatch Failure in Keyword Search (Query 1, 4, 7)
- **Observation:** In queries 1, 4, and 7, Keyword Search returned **0 results**.
- **Root Cause:** In Query 4 (*"twenty percent discount"*), the source text contained `"20% discount"`. PostgreSQL `websearch_to_tsquery` does not equate the words `"twenty"` and `"percent"` with the symbol `"20%"`. Similarly, `"credentials"` does not match `"password"`, and `"stop paying"` does not match `"cancel subscription"`.
- **Hybrid Solution:** Vector Search resolved the semantic meaning immediately (similarity 0.675 to 0.791). Hybrid search fused the result with an RRF score of `0.00820`, delivering the correct answer to the user despite the keyword branch returning zero matches.

### Case 2: Exact Technical Matching & Reinforcement (Query 2, 3, 5, 8)
- **Observation:** For exact error codes (`HTTP 423 Locked`), tax identifiers (`VIES VAT`), and cryptographic headers (`X-Signature-SHA256`), both Keyword Search and Vector Search identified the target chunk.
- **Hybrid Solution:** Reciprocal Rank Fusion awarded the maximum combined score (`0.01639`), ranking the mutually validated chunk at the very top of the retrieved list with 100% confidence.

### Case 3: Source Citation and Parent Document Provenance
Every retrieved `DocumentChunk` provides:
1. `id`: The exact chunk UUID.
2. `document_id`: The parent document UUID in `documents`.
3. `chunk_index`: The sequence number within the parent document.
4. `metadata`: Inherited parent tags (`category`, `source`, `language`) combined with chunk-level attributes (`char_start`, `char_end`, `word_count`).

This enables frontend clients and RAG generation pipelines to display both the granular excerpt and a verifiable link back to the full parent source document.

---

## 8. Automated Test Suite Validation

The test suite was expanded with unit and endpoint tests covering:
- Text normalization, regex cleaning, and boundary-preserving chunking (`tests/unit/test_chunking.py`).
- Chunked ingestion, foreign key cascading, and idempotent batch processing (`tests/unit/test_chunk_search.py`).
- Vector, keyword, and hybrid RRF search algorithms with dual-dialect SQLite/PostgreSQL support.
- All existing chat, prompt, document, and search endpoints.

**Result:** **142 tests passing** in ~12.83 seconds (`uv run pytest`).
