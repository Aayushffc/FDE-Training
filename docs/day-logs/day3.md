# Day 3 — Daily Progress Log
**Date:** 19–20 November 2026  
**Track:** AI Engineering / Forward Deployed Engineer  
**Member:** Member B — AI Agents  
**Focus:** Vector Databases, Gemini 768-Dim Embeddings, Document Ingestion, Semantic Search, PostgreSQL FTS, and IR Evaluation

---

## 1. Work Completed

### Vector Database & Schema Migration
- Integrated Neon Serverless PostgreSQL with the `pgvector` extension enabled.
- Authored and applied Alembic migration (`b18e9a6256cb_create_documents_table.py`) creating the `documents` table:
  - `id`: UUID primary key.
  - `content`: Original document text.
  - `embedding`: High-dimensional vector column (`VECTOR(768)`).
  - `metadata`: JSONB dictionary for category, source, language, and tags.
  - `embedding_model`: Identifier string (`gemini-embedding-001`).
  - `created_at` and `updated_at`: Time-zone aware auditing timestamps.

### Clean Architecture & Provider Reorganization
- Relocated low-level Google GenAI SDK integration into the infrastructure layer:
  - `app/infrastructure/embedding_providers/base.py`: Provider-agnostic `BaseEmbeddingProvider` interface.
  - `app/infrastructure/embedding_providers/gemini.py`: Concrete adapter wrapping Google's modern `google-genai` SDK with `gemini-embedding-001` and `output_dimensionality=768`.
  - `app/infrastructure/embedding_providers/__init__.py`: Factory `get_embedding_provider()`.
  - `app/services/embedding_service.py`: Backward-compatible service alias.

### Prompt API Normalization
- Removed the legacy `/db` prefix across prompt endpoints.
- Standardized all prompt routes under `/v1/prompts`:
  - `GET /v1/prompts`: Retrieves all prompt categories with their active versions directly from PostgreSQL.
  - `POST /v1/prompts`: Creates or updates versioned prompts in PostgreSQL.
  - `GET /v1/prompts/{category}/{version}`: Retrieves version details.
  - `GET /v1/prompts/categories`: Dedicated endpoint for listing categories from `prompt_categories`.
  - `POST /v1/prompts/categories`: Creates a new category.
  - `GET /v1/prompts/categories/{category_name}`: Gets a category with its versions.
  - `GET /v1/prompts/categories/{category_name}/versions`: Lists versions for a specific category.
  - `GET /v1/prompts/versions/{version_id}`: Fetches prompt versions by primary key ID.

### Phase 3 — Sample Dataset Creation
- Curated a balanced dataset of exactly 30 customer-support FAQ documents in `data/sample_documents.json`:
  - **Account**: 8 documents (password reset, 2FA, lockout, email change, profile update, ownership transfer, deletion, sessions).
  - **Billing**: 8 documents (payment methods, declined cards, invoices, proration, promo codes, refunds, currencies, backup cards).
  - **Subscription**: 7 documents (plan upgrades, cancellations, downgrades, annual discount, team seats, trials, enterprise SLA).
  - **Technical support**: 7 documents (browser cache, 429 rate limits, API keys, webhooks, compatibility, data export, status page).

### Phase 4 — Document Ingestion & Idempotent Seeding
- Implemented `IngestionService` in `app/services/ingestion_service.py`:
  - Supports single document ingestion and batch ingestion.
  - Added `ingest_from_json_file(file_path, skip_duplicates=True, batch_size=10)`.
  - Built content-based deduplication to ensure rerun safety and idempotency.
- Created `scripts/seed_documents.py`:
  - Successfully seeded all 30 documents into Neon PostgreSQL with Gemini 768-dim embeddings.
  - Verified rerun idempotency: 0 inserted, 30 skipped duplicates.

### Phase 5 — Semantic Vector Search
- Built `SearchService.semantic_search` in `app/services/search_service.py`:
  - Embeds query into a 768-dim vector via Gemini.
  - Executes cosine distance similarity search via pgvector `<=>` on Neon PostgreSQL.
  - Filters candidates by JSON metadata (e.g., `{"category": "account"}`).
  - Returns ranked documents with similarity and distance scores.
- Exposed `POST /v1/search/semantic`.
- Enforced single `/v1` path normalization (unversioned `/search` returns 404).

### Phase 6 — PostgreSQL Full-Text Keyword Search
- Built `SearchService.keyword_search` in `app/services/search_service.py`:
  - Leverages native PostgreSQL `to_tsvector('english', Document.content)`.
  - Uses `websearch_to_tsquery('english', query)` to parse user input safely without query syntax crashes.
  - Matches with `@@` operator and ranks results with `ts_rank`.
  - Fully supports metadata filters (`category`).
- Exposed `POST /v1/search/keyword`.

### Phase 7 — Comparative Evaluation
- Authored benchmark runner `scripts/compare_searches.py` evaluating 10 real queries against Neon PostgreSQL.
- Compiled findings into comprehensive report `docs/search-comparison-report.md`.
- **Key finding**: Semantic search achieved a 100% success rate on conversational queries by resolving vocabulary gaps (e.g. *"credentials"* $\to$ *"password"*, *"stop paying"* $\to$ *"cancel subscription"*), while keyword search provided exact precision for technical identifiers and error codes (`X-Signature-SHA256`, `HTTP 429`).

---

## 2. Technical Decisions

- **Matryoshka Embeddings (768 Dimensions):** Configured `gemini-embedding-001` with `output_dimensionality=768`. Cuts database storage and memory bandwidth by 75% compared to the default 3072 dimensions while maintaining >98% retrieval accuracy.
- **Unified PostgreSQL with pgvector:** Avoided external standalone vector databases (Pinecone, Chroma) to maintain single-database ACID guarantees, zero ETL sync latency, and combined relational/JSONB filtering.
- **Strict `/v1` Versioning:** Eliminated unversioned routes (`/search/semantic`) in favor of uniform `/v1/search/semantic` and `/v1/search/keyword`.
- **DRY Architecture:**
  - Common `BaseSearchRequest` shared between semantic and keyword requests.
  - Centralized `_apply_metadata_filters` helper in `SearchService`.
- **Dual-Dialect Testing Strategy:** SQLite in-memory tests use a pure-Python cosine similarity fallback, while production utilizes C-accelerated pgvector and PostgreSQL FTS.

---

## 3. Testing and Validation

- **Automated Test Suite:** **125 tests passing** in ~11.89 seconds:
  - Unit tests for embedding service, ingestion service, and document schemas (`tests/unit/test_documents.py`).
  - Unit tests for prompt categories and versions CRUD (`tests/unit/test_db_prompts.py`).
  - Unit and endpoint tests for semantic search and keyword search (`tests/unit/test_search.py`).
  - Integration and chat tests (`tests/api/test_chat.py`, `tests/integration/`).
- **Live Verifications:**
  - 30 FAQ documents verified stored in Neon PostgreSQL with 768-dim vector embeddings.
  - Seeding script rerun tested for idempotency (30 duplicates skipped).
  - 10-query benchmark run live against Neon PostgreSQL.

---

## 4. Challenges and Learnings

- **SQLAlchemy Relationship Shadowing:** Defining a Python `@property def category(self)` on `PromptVersion` masked the SQLAlchemy `relationship("PromptCategory")`, causing mapper initialization failure. Resolved by removing the shadowing property and using Pydantic `@computed_field` on response schemas.
- **SQLite pgvector Operator Incompatibility:** SQLite threw syntax errors on pgvector's `<=>` operator during in-memory unit tests. Built a transparent Python cosine similarity fallback when `dialect == "sqlite"`, allowing fast hermetic tests.
- **Vocabulary Mismatch Phenomenon:** Benchmark confirmed that users rarely use technical documentation keywords when seeking help. Semantic embeddings bridged colloquial phrases (*"Can I get my money back?"*) to formal documentation (*"14-day refund policy"*), whereas keyword search returned 0 matches.

---

## 5. Day 3 Outcome

The gateway now features a production-ready vector database integration, 30 curated and embedded FAQ documents in Neon PostgreSQL, dual search capabilities (semantic vector retrieval and PostgreSQL full-text keyword search), a complete 10-query evaluation report, and 125 automated tests passing with 100% clean architecture separation.
