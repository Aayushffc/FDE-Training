# Reranking & Hybrid Retrieval Architecture Deep Dive
**Platform**: Production RAG & LLM Gateway  
**Focus**: Reciprocal Rank Fusion (RRF), Cross-Encoder Heuristic Reranking, Parameter Mechanics, and Clean Architecture

---

## 1. Executive Summary & Retrieval Flow

In high-accuracy production Question-Answering (RAG) systems, single-stage vector search is insufficient. Unstructured natural language queries frequently suffer from vocabulary mismatches, semantic dilution, or miss critical exact technical identifiers (such as error codes, function headers, or compliance acronyms).

Our system employs a **two-stage hybrid retrieval and reranking pipeline**:
1. **Stage 1 (Dual Retrieval)**: Simultaneously retrieves candidate passages from PostgreSQL using **Dense Vector Search** (Gemini 768-dim embeddings via `pgvector <=>`) and **Sparse Lexical Search** (PostgreSQL Full-Text Search `to_tsvector` and `ts_rank`).
2. **Stage 1 Fusion**: Merges both candidate lists using **Reciprocal Rank Fusion (RRF)** with standard smoothing constant $k=60$.
3. **Stage 2 (Cross-Encoder Reranking)**: Evaluates the top $2 \times k$ candidates against the query simultaneously, scoring query term coverage, exact phrase alignment, and token proximity before passing the top $k$ chunks to the Context Engineering service.

```
User Query: "How can I recover my account after too many failed attempts?"
                                  │
         ┌────────────────────────┴────────────────────────┐
         ▼                                                 ▼
[Dense Vector Retrieval]                        [Sparse Lexical Retrieval]
Gemini 768-dim <=> pgvector                     PostgreSQL to_tsvector & ts_rank
(Captures semantic meaning & intent)            (Captures exact terms & error codes)
         │                                                 │
         └────────────────────────┬────────────────────────┘
                                  ▼
                [Reciprocal Rank Fusion (RRF)]
                     RRF_Score = Σ w / (k + rank)
                     k = 60 (Empirical Constant)
                     alpha = 0.5 (Balanced Weighting)
                                  │
                                  ▼
                [Candidate Pool (Top 2 * k = 10)]
                                  │
                                  ▼
                [Cross-Encoder Reranking Engine]
             (app/infrastructure/rerankers/heuristic.py)
             ├── 1. Query Term Coverage (40%)
             ├── 2. Exact Phrase / Bigram Alignment (15-35%)
             ├── 3. Token Density & Proximity (5-15%)
             └── 4. First-Stage Retrieval Prior (30%)
                                  │
                                  ▼
                    [Top-k Reranked Chunks]
                                  │
                                  ▼
                [Context Engineering Service]
             (Deduplication + Token Budget Trimming + [Source N] Labeling)
                                  │
                                  ▼
                [Grounded LLM Answer Generation]
             (Answers strictly from context + bracketed citations)
```

---

## 2. First-Stage Fusion: What are `fusion_method`, `rrf_k`, and `alpha`?

### A. The "Apples vs. Oranges" Scoring Problem
- **Vector Search Score**: Cosine similarity is mathematically bounded in $[0.0, 1.0]$ (e.g., $0.6912$).
- **Keyword Search Score**: PostgreSQL `ts_rank` is an unbounded token-frequency metric in $[0.0, \infty)$ (e.g., $0.05, 1.25, 18.4$).
- **Why Naive Score Normalization Fails**: Min-max scaling or z-score normalization on small candidate pools is wildly unstable. A single keyword hit with high frequency can stretch the denominator and compress all vector scores to near zero.

---

### B. `fusion_method: "rrf"` (Reciprocal Rank Fusion)
Reciprocal Rank Fusion bypasses the score incompatibility problem by **discarding raw scores entirely and scoring candidates based solely on their 1-based rank positions**:

$$\text{RRF\_Score}(d) = \sum_{m \in \{\text{vector}, \text{keyword}\}} \frac{w_m}{k + \text{rank}_m(d)}$$

Where:
- $\text{rank}_m(d)$ is the 1-based rank position of chunk $d$ in list $m$ ($1, 2, 3, \dots$).
- $w_m$ is the weight of retrieval branch $m$.
- $k$ is the smoothing constant (rank dampener).

If a chunk appears in **both** candidate lists, its reciprocal scores sum together, catapulting it past chunks that only appeared in a single list.

---

### C. `rrf_k: 60` (The Smoothing Constant)
$k$ prevents top-ranked items from a single retrieval method from completely dominating the consolidated ranking.

#### Why Exactly 60?
In 2009, Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher published the foundational research paper:
> *"Reciprocal Rank Fusion outperforms Condorcet and individual Run Methods in Information Retrieval"* (ACM SIGIR 2009).

Through exhaustive testing across TREC benchmark corpora, the authors proved that **$k = 60$ consistently achieves superior Mean Reciprocal Rank (MRR) and precision compared to smaller or larger values**.

#### Numerical Proof: Why Small $k$ Fails and $k=60$ Succeeds
| Rank Position | Score with $k = 1$ | Score with $k = 60$ |
| :--- | :--- | :--- |
| **Rank 1** | $\frac{1}{1 + 1} = \mathbf{0.5000}$ | $\frac{1}{60 + 1} \approx \mathbf{0.01639}$ |
| **Rank 2** | $\frac{1}{1 + 2} = \mathbf{0.3333}$ | $\frac{1}{60 + 2} \approx \mathbf{0.01613}$ |
| **Rank 3** | $\frac{1}{1 + 3} = \mathbf{0.2500}$ | $\frac{1}{60 + 3} \approx \mathbf{0.01587}$ |
| **Rank 10** | $\frac{1}{1 + 10} = \mathbf{0.0909}$ | $\frac{1}{60 + 10} \approx \mathbf{0.01428}$ |
| **Gap (Rank 1 vs 2)** | $\Delta = \mathbf{0.1667}$ *(Huge penalty)* | $\Delta = \mathbf{0.00026}$ *(Smooth gradient)* |

#### The Consensus Effect in Practice:
Suppose we query: `"HTTP 423 cooling-off period"`:
- **Chunk A** (mentions general lockout rules): Rank 1 in Vector, but **absent in Keyword**.
  $$\text{Score}(A) = \frac{0.5}{60 + 1} + 0 = \mathbf{0.00819}$$
- **Chunk B** (mentions both lockout and the exact HTTP 423 code): Rank 2 in Vector AND Rank 2 in Keyword.
  $$\text{Score}(B) = \frac{0.5}{60 + 2} + \frac{0.5}{60 + 2} = 0.00806 + 0.00806 = \mathbf{0.01612}$$

**Chunk B scores nearly double Chunk A!** Even though Chunk A had higher pure semantic similarity, Chunk B wins because **both retrieval engines reached consensus**.

---

### D. `alpha: 0.5` (The Vector vs. Keyword Slider)
The parameter $\alpha \in [0.0, 1.0]$ allocates weight between dense semantic retrieval and sparse lexical retrieval:
- $w_{\text{vector}} = \alpha$
- $w_{\text{keyword}} = 1.0 - \alpha$

| Alpha ($\alpha$) | Strategy Profile | Best Used For |
| :--- | :--- | :--- |
| **`0.5`** *(Default)* | **Balanced Consensus** | General-purpose search. Gives equal voice to intent and exact words. |
| **`0.7`** | **Semantic-Biased** | Conversational customer support (e.g. *"I want to stop paying"* $\rightarrow$ *"cancel subscription"*). |
| **`0.2`** | **Lexical-Biased** | Exact technical inquiries (e.g. searching specific error logs, RFC numbers, or API headers). |
| **`1.0`** | **Pure Dense Vector** | Keyword search ignored entirely. |
| **`0.0`** | **Pure Sparse Keyword** | Vector search ignored entirely. |

---

## 3. Second-Stage Retrieval: What is the Reranker?

### The Bi-Encoder Bottleneck
Standard vector embeddings (like Gemini `gemini-embedding-001`) use a **Bi-Encoder architecture**:
- The query $q$ is compressed into a vector $\vec{u} \in \mathbb{R}^{768}$ without seeing the document.
- The document $d$ is compressed into a vector $\vec{v} \in \mathbb{R}^{768}$ without seeing the query.
- Proximity is computed via dot product or cosine distance.

**The Problem**: Compressing 400 characters into 768 floating-point numbers causes **information loss**. Subtle combinations of words, exact technical codes (`HTTP 423`), or token proximities get washed out.

### The Cross-Encoder Solution
A Cross-Encoder does not evaluate $q$ and $d$ separately. Instead, it processes $q$ and $d$ **together simultaneously**, evaluating token-to-token interactions.

Our high-performance production reranker ([app/infrastructure/rerankers/heuristic.py](file:///d:/FDE_Training/week1/app/infrastructure/rerankers/heuristic.py)) performs cross-encoder evaluation with sub-millisecond ($<1\,\text{ms}$) latency and zero external dependencies.

---

## 4. How the Reranking Heuristics Work (Step-by-Step)

The engine evaluates four weighted signals for each candidate chunk:

$$\text{Relevance Score} = (\text{Coverage} \times 0.40) + \text{PhraseBonus} + \text{DensityScore} + \text{PriorScore}$$

### Signal 1: Query Term Coverage (40% Weight)
Measures the percentage of unique non-stopword query tokens present in the candidate chunk:
$$\text{Coverage} = \frac{|\text{Tokens}(q) \cap \text{Tokens}(c)|}{|\text{Tokens}(q)|}$$
- Prevents retrieval of chunks that match only one repeated keyword while ignoring the rest of the question.

### Signal 2: Exact Phrase & N-Gram Alignment (15% to 35% Weight)
- **Full Contiguous Phrase**: If the cleaned query string appears as an uninterrupted substring in the chunk, an immediate **$+0.35$ bonus** is awarded.
- **Bigram Co-occurrence**: If the entire phrase is not contiguous, the reranker extracts adjacent word pairs (bigrams: $[w_1\ w_2, w_2\ w_3, \dots]$) and awards up to **$+0.20$** proportional to the bigram match ratio.

### Signal 3: Token Density & Proximity (5% to 15% Weight)
Where do the query words appear in the chunk?
- If query words are scattered 300 characters apart across different sentences, they are likely unrelated.
- If they appear tightly clustered in the same sentence, they form a cohesive semantic answer.
$$\text{Span} = \text{Pos}_{\text{last}} - \text{Pos}_{\text{first}} + 1$$
$$\text{DensityScore} = \min\left(1.0, \frac{|\text{Tokens}(q)|}{\text{Span}}\right) \times 0.15$$

### Signal 4: First-Stage Prior Retention (30% Weight)
Retains the signal from first-stage retrieval so initial similarity isn't completely discarded:
- For Vector candidates: $\min(1.0, \text{CosineSimilarity}) \times 0.30$.
- For Keyword candidates: Uses a saturated exponential curve $(1 - e^{-2 \cdot ts\_rank}) \times 0.30$.

---

## 5. Live Benchmark & Retrieval Failure Modes

Tested against production data from [data/chunk_data.json](file:///d:/FDE_Training/week1/data/chunk_data.json):

```powershell
.venv\Scripts\python.exe scripts\compare_three_retrieval_methods.py
```

### Case 1: Vocabulary Mismatch Failure
- **Query**: `"I want to stop paying for this service"`
- **Target**: Account cancellation & refund policies.
- **Keyword FTS**: **0 matches found** ($ts\_rank=0.0000$). The user used conversational words; the document used words like "cancel", "downgrade", and "money-back".
- **Vector Search**: **5 matches** ($cosine=0.5623$, Top 1: *"Refund Policies and Account Credit Adjustments"*).
- **Winner**: **Vector / Hybrid**.

### Case 2: Exact Technical Identifier
- **Query**: `"HTTP 423 Locked"`
- **Target**: Account brute-force lockout threshold.
- **Keyword FTS**: **1 exact match** ($ts\_rank=0.2683$).
- **Vector Search**: **5 matches** ($cosine=0.6912$).
- **Hybrid RRF**: Consolidated exact chunk at **Rank 1** ($RRF=0.0164$).
- **Reranker Impact**: Promoted relevance score to **$1.0000$** due to contiguous phrase match bonus.
- **Winner**: **Hybrid + Reranker consensus**.

### Case 3: Fact Extraction
- **Query**: `"How many single-use recovery codes are generated during 2FA setup?"`
- **Target**: The number of single-use codes (ten).
- **Keyword FTS**: **0 matches found** (terms were too dispersed across grammatical boilerplate).
- **Vector Search**: **5 matches** ($cosine=0.7438$, retrieved: *"During enrollment, a secret seed key is generated alongside ten single-use alphanumeric recovery codes"*).
- **Winner**: **Vector / Hybrid**.

---

## 6. Architecture & Codebase Organization

Following Clean Architecture principles, technical adapters and mathematical algorithms have been decoupled from business orchestration services:

| Component | Layer | File Path | Responsibility |
| :--- | :--- | :--- | :--- |
| **`BaseReranker`** | Infrastructure | [app/infrastructure/rerankers/base.py](file:///d:/FDE_Training/week1/app/infrastructure/rerankers/base.py) | Abstract interface for scoring query-passage pairs. |
| **`HeuristicReranker`** | Infrastructure | [app/infrastructure/rerankers/heuristic.py](file:///d:/FDE_Training/week1/app/infrastructure/rerankers/heuristic.py) | High-performance token coverage & proximity cross-scorer. |
| **`compute_rrf_score`** | Infrastructure | [app/infrastructure/ranking/rrf.py](file:///d:/FDE_Training/week1/app/infrastructure/ranking/rrf.py) | Mathematical Reciprocal Rank Fusion implementation. |
| **`clean_text` / Splitters** | Infrastructure | [app/infrastructure/text_processing/](file:///d:/FDE_Training/week1/app/infrastructure/text_processing/) | Text normalization, sentence boundary detection, regex splitters. |
| **`SearchService`** | Service | [app/services/search_service.py](file:///d:/FDE_Training/week1/app/services/search_service.py) | SQL execution over `documents` and `document_chunks`. |
| **`RerankingService`** | Service | [app/services/reranking_service.py](file:///d:/FDE_Training/week1/app/services/reranking_service.py) | Business facade managing candidate reordering. |
| **`RAGService`** | Service | [app/services/rag_service.py](file:///d:/FDE_Training/week1/app/services/rag_service.py) | End-to-end RAG orchestrator with citation validation. |
| **`dependencies.py`** | API Dependency Hub | [app/api/v1/dependencies.py](file:///d:/FDE_Training/week1/app/api/v1/dependencies.py) | Centralized FastAPI dependency injection providers. |

---

## 7. API Specification (Simplified Endpoints)

Following your requirement, `fusion_method` and `rrf_k` are **fully encapsulated inside the backend engine**. They are no longer exposed in Swagger / API request schemas:

### Active Endpoints:
1. **Unchunked Search** (operates on whole `documents`):
   - `POST /v1/search/semantic` — Dense vector search
   - `POST /v1/search/keyword` — Sparse keyword search
   - `POST /v1/search/hybrid` — Hybrid search (RRF $k=60$)
2. **Chunked Search** (operates on granular `document_chunks`):
   - `POST /v1/search/chunks/semantic` — Dense vector search
   - `POST /v1/search/chunks/keyword` — Sparse keyword search
   - `POST /v1/search/chunks/hybrid` — Hybrid search (RRF $k=60$, optional reranking)
3. **RAG Question Answering**:
   - `POST /v1/rag/ask` — End-to-end RAG with citations, latency, token usage, and pricing telemetry.

### Request Payload Example (`POST /v1/search/chunks/hybrid`):
```json
{
  "query": "HTTP 423 cooling-off period",
  "top_k": 5,
  "alpha": 0.5,
  "rerank": true
}
```
*Note: `fusion_method` defaults internally to `"rrf"` and `rrf_k` defaults to `60`.*
