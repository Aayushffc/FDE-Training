# LangChain Teach-Back Guide: Retrievers, Runnables, and Plain Python RAG Architecture

**Sprint:** AI Engineering / Forward Deployed Engineer (FDE)  
**Session Duration:** 45-Minute Technical Teach-Back  
**Audience:** Senior Engineers, Technical Leads, and System Architects  
**Topic:** Deep-Dive Comparison of LangChain Abstractions vs. Native Plain Python Clean Architecture in Production RAG Systems  

---

## Executive Summary

As retrieval-augmented generation (RAG) matures from quick hackathon prototypes to mission-critical enterprise infrastructure, engineering teams face a fundamental architectural choice:
1. **Framework-Centric Approach:** Build upon higher-level frameworks such as **LangChain / LCEL (LangChain Expression Language)**.
2. **First-Principles Plain Python Approach:** Build decoupled, observable, and modular pipelines adhering to **Clean Architecture / Hexagonal Architecture** using native Python, standard ORMs (SQLAlchemy, pgvector), and direct SDK clients.

This teach-back dissects the core abstractions of LangChain—specifically **Retrievers** and **Runnables**—and provides an exact, side-by-side comparison with the plain Python implementation built in our Day 4 sprint codebase.

---

## Part 1: LangChain Retrievers Demystified

### 1.1 What is a LangChain Retriever?

In LangChain, a **Retriever** is an abstraction defined by the abstract base class `BaseRetriever` (`langchain_core.retrievers.BaseRetriever`). Unlike a Vector Store, which exposes low-level CRUD operations (`add_texts`, `similarity_search_with_score`), a Retriever is a unified read-only query interface that takes an unstructured query string and returns a list of `Document` objects:

$$\text{Retriever}: \text{str} \longrightarrow \text{List}[\text{Document}]$$

Every `BaseRetriever` implements:
- `get_relevant_documents(query: str) -> List[Document]` (synchronous)
- `aget_relevant_documents(query: str) -> Coroutine[List[Document]]` (asynchronous)

### 1.2 Common LangChain Retriever Variants

```
                     ┌───────────────────────┐
                     │     BaseRetriever     │
                     └───────────┬───────────┘
                                 │
         ┌───────────────────────┼───────────────────────┐
         ▼                       ▼                       ▼
┌──────────────────┐   ┌──────────────────┐   ┌───────────────────────┐
│ VectorStore-     │   │ Ensemble-        │   │ Contextual-           │
│ Retriever        │   │ Retriever        │   │ CompressionRetriever  │
│ (Similarity/MMR) │   │ (RRF / Weighted) │   │ (Reranking/Filtering) │
└──────────────────┘   └──────────────────┘   └───────────────────────┘
```

1. **`VectorStoreRetriever`:**
   Wraps a vector store (e.g., PGVector, Chroma, Pinecone). Configurable with search types:
   - `similarity`: standard top-k cosine / inner product search.
   - `similarity_score_threshold`: filters candidates below a minimum similarity cutoff.
   - `mmr` (Maximal Marginal Relevance): balances relevance with diversity to reduce redundancy.
2. **`EnsembleRetriever`:**
   Orchestrates multiple underlying retrievers (e.g., BM25 keyword retriever + VectorStoreRetriever) and merges candidate lists using **Reciprocal Rank Fusion (RRF)**:
   $$\text{RRF\_Score}(d) = \sum_{r \in \text{retrievers}} \frac{w_r}{k + \text{rank}_r(d)}$$
3. **`ContextualCompressionRetriever`:**
   A two-stage retriever that wraps a base retriever with a compressor (e.g., Cohere Reranker, Cross-Encoder, or LLM-based extractor). The compressor trims, reorders, or drops irrelevant chunks before returning them to the caller.
4. **`ParentDocumentRetriever` / `MultiVectorRetriever`:**
   Splits large documents into smaller chunks for high-granularity vector search, but returns the larger parent document (or a window of surrounding context) during retrieval.

---

## Part 2: LangChain Runnables & LCEL (LangChain Expression Language)

### 2.1 The Runnable Protocol

In LangChain 0.1+, the entire framework is unified under the **`Runnable`** interface (`langchain_core.runnables.Runnable`). Any component—prompt templates, chat models, retrievers, output parsers, and custom functions—is a `Runnable` supporting a standard lifecycle:

| Method | Execution Mode | Purpose |
| :--- | :--- | :--- |
| `invoke(input)` | Sync | Single-item synchronous execution |
| `ainvoke(input)` | Async | Single-item asynchronous non-blocking coroutine |
| `batch(inputs)` | Sync | Parallel execution over an iterable of inputs |
| `abatch(inputs)` | Async | Asynchronous batched execution with concurrency limits |
| `stream(input)` | Sync | Yields output chunks as they become available |
| `astream(input)` | Async | Asynchronous generator yielding streaming tokens |

### 2.2 LCEL Composition and Key Runnables

LCEL overloads the Python bitwise OR operator (`|`) to create declarative computational directed acyclic graphs (DAGs), inspired by Unix pipes:

$$\text{chain} = \text{input} \mid \text{step}_1 \mid \text{step}_2 \mid \text{output}$$

Key building blocks include:
- **`RunnablePassthrough`:** Forwards incoming data untouched, or injects extra keys via `RunnablePassthrough.assign(...)`.
- **`RunnableParallel` (or dict syntax):** Executes multiple sub-runnables concurrently over the same input (e.g., retrieving context while passing the original query through).
- **`RunnableLambda`:** Wraps any plain Python function `Callable[[A], B]` into a first-class `Runnable`.

### 2.3 The Canonical LangChain LCEL RAG Implementation

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.vectorstores import PGVector
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# 1. Setup Retriever
embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001")
vectorstore = PGVector(connection_string=DB_URL, embedding_function=embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# 2. Define Prompt
template = """Answer the question based only on the following context.
If you cannot answer, say "I don't know based on the provided context."

Context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

# 3. Format Context Helper
def format_docs(docs):
    return "\n\n".join(f"[Source {i+1}] {doc.page_content}" for i, doc in enumerate(docs))

# 4. Construct Declarative LCEL Pipeline
rag_chain = (
    RunnableParallel({
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    })
    | prompt
    | ChatGroq(model="llama-3.3-70b-versatile", temperature=0.0)
    | StrOutputParser()
)

# 5. Execute
response = await rag_chain.ainvoke("What discount is offered for annual billing?")
```

---

## Part 3: Plain Python Clean Architecture Implementation (Our Approach)

In our Day 4 codebase, we deliberately implemented the entire pipeline using **plain Python** and standard enterprise patterns without LangChain dependencies.

### 3.1 Architectural Decomposition

Instead of hiding state and database transactions behind nested class wrappers, we separated concerns across single-responsibility domain services:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        FastAPI REST Layer                              │
│                    POST /v1/rag/ask                                    │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│                      RAGService (Orchestrator)                         │
│   • Coordinates Retrieval -> Reranking -> Context Assembly -> LLM      │
└──────────┬────────────────────────┬──────────────────────┬─────────────┘
           │                        │                      │
           ▼                        ▼                      ▼
┌──────────────────────┐ ┌──────────────────────┐ ┌──────────────────────┐
│    SearchService     │ │   RerankingService   │ │ContextEngineeringSvc │
│  • pgvector Cosine   │ │  • Cross-Encoder     │ │  • Deduplication     │
│  • Postgres tsvector │ │  • Token Coverage    │ │  • Budget Trimming   │
│  • RRF Fusion (k=60) │ │  • Phrase Proximity  │ │  • [Source N] Label  │
└──────────────────────┘ └──────────────────────┘ └──────────────────────┘
```

### 3.2 Side-by-Side Code Comparison

#### Feature A: Hybrid Retrieval with Reciprocal Rank Fusion

* **LangChain Way:**
  ```python
  from langchain.retrievers import EnsembleRetriever
  from langchain_community.retrievers import BM25Retriever
  
  # Requires loading all documents into memory to initialize BM25
  bm25_retriever = BM25Retriever.from_texts(all_chunk_texts)
  ensemble_retriever = EnsembleRetriever(
      retrievers=[bm25_retriever, vector_retriever],
      weights=[0.5, 0.5],
      c=60,
  )
  docs = ensemble_retriever.get_relevant_documents("password lockout")
  ```
  *Flaw:* `BM25Retriever` in LangChain typically runs in-memory Python dictionaries. If you have 1,000,000 chunks, in-memory BM25 crashes the application server on startup.

* **Our Plain Python Way (`SearchService.chunk_hybrid_search`):**
  ```python
  # Offloads BM25/keyword indexing directly to Neon PostgreSQL full-text index!
  vector_results = self.chunk_vector_search(query=query, top_k=fetch_k, filters=filters)
  keyword_results = self.chunk_keyword_search(query=query, top_k=fetch_k, filters=filters)
  
  # Pure deterministic Reciprocal Rank Fusion (RRF) math in 15 lines of pure Python:
  scores: Dict[uuid.UUID, float] = {}
  for rank, item in enumerate(vector_results.results, start=1):
      scores[item.id] = scores.get(item.id, 0.0) + (alpha / (rrf_k + rank))
  for rank, item in enumerate(keyword_results.results, start=1):
      scores[item.id] = scores.get(item.id, 0.0) + ((1.0 - alpha) / (rrf_k + rank))
  ```
  *Advantage:* Leverages native PostgreSQL GIN indexes and pgvector HNSW indexes without duplicate memory footprints.

---

#### Feature B: Context Assembly, Deduplication, and Prompt Grounding

* **LangChain Way:**
  Often requires multiple chained Runnables (`RunnablePassthrough.assign`, custom lambda functions, or third-party compressors).
  ```python
  def deduplicate(docs):
      # Custom lambda inserted into pipe
      ...
  ```
* **Our Plain Python Way (`ContextEngineeringService`):**
  Explicit, unit-testable methods with zero magic:
  - `deduplicate_chunks(items, overlap_threshold=0.80)`: Jaccard set-intersection deduplication.
  - `trim_context(items, max_chars=4000)`: Hard character budget enforcement.
  - `assemble_labeled_context(items)`: Attaches structured source headers and builds source metadata manifests.
  - `build_rag_prompt(question, items)`: Builds system prompt with strict `"I don't know"` contract.

---

#### Feature C: RAG Pipeline Execution

* **LangChain Way (LCEL Pipe):**
  ```python
  chain = {"context": retriever | format_docs, "question": RunnablePassthrough()} | prompt | model | parser
  result = await chain.ainvoke(question)
  # Citations and chunk IDs are lost unless custom callbacks or Document objects are preserved in complex dicts.
  ```
* **Our Plain Python Way (`RAGService.generate_rag_answer`):**
  ```python
  # 1. Candidate retrieval
  candidates = self.retrieve_candidates(payload.question, strategy=strat, top_k=fetch_k)
  
  # 2. Cross-encoder reranking
  if payload.rerank:
      candidates = self.reranker.rerank(payload.question, candidates, top_k=payload.top_k)
  
  # 3. Context assembly
  system_prompt, user_prompt, manifest = self.context_service.build_rag_prompt(
      question=payload.question,
      items=candidates,
  )
  
  # 4. LLM completion
  llm_resp = await provider.generate_chat_completion(
      payload_or_message=user_prompt,
      system_prompt=system_prompt,
  )
  
  # 5. Grounding & citation parsing
  cited_tags = re.findall(r"\[(Source \d+)\]", llm_resp.reply)
  return RAGResponse(
      answer=llm_resp.reply,
      sources=manifest,
      citations=cited_tags,
      is_grounded=not is_unknown,
  )
  ```
  *Advantage:* 100% visible control flow. Standard Python debuggers (`pdb`, IDE breakpoints) inspect every variable directly without stepping through 45 layers of LangChain internal dispatcher frames.

---

## Part 4: Production Trade-Off Matrix

| Evaluation Dimension | LangChain / LCEL Framework | Plain Python Clean Architecture (Our System) |
| :--- | :--- | :--- |
| **Speed to Initial MVP** | **High** (Pre-built adapters allow 20-line demos in 30 minutes). | **Medium** (Requires writing data models, schemas, and SQL queries). |
| **Debuggability & Stack Traces** | **Poor** (Stack traces span 30+ framework files across `langchain_core`, `pydantic`, and `runnables`). | **Exceptional** (Stack trace directly highlights the failing line in your service or SQL query). |
| **Observability & Profiling** | Requires LangSmith / Langfuse callback managers or heavy tracing hooks. | Native per-request middleware, Prometheus metrics, and structured JSON logs. |
| **Dependency Weight** | **Heavy** (Dozens of transitive packages, frequent breaking changes between minor versions). | **Minimal** (FastAPI, SQLAlchemy, pgvector, httpx, and official SDKs). |
| **Database Scalability** | Often defaults to naive client-side operations (e.g., in-memory BM25, client-side filtering). | Fully offloaded to database engine (PostgreSQL GIN indexes, pgvector HNSW/IVFFlat). |
| **Custom Business Logic** | Rigid (Fitting custom multi-tenant authorization or dynamic database transactions into LCEL pipes is awkward). | Frictionless (Standard dependency injection and explicit database sessions via FastAPI `Depends`). |
| **Citation Lineage** | Difficult (Standard LCEL chains strip document metadata unless wrapped in complex container objects). | Native (Document IDs, chunk UUIDs, and character offsets remain preserved in domain schemas). |

---

## Part 5: When Should You Use Which?

### Use LangChain When:
1. **Rapid Hackathons / POCs:** You need to validate a concept in 48 hours and need pre-built connectors for exotic vector databases or document loaders (e.g., parsing 10 different obscure file formats).
2. **Standard Non-Critical Workflows:** Where deep performance optimization, strict latency budgets (<100ms), and custom SQL execution are not primary concerns.
3. **Teams Heavily Invested in LangSmith:** If the organization relies exclusively on LangSmith for evaluation and agent tracing.

### Use Plain Python When:
1. **Production Enterprise Systems (FDE Track):** You require high throughput, low latency, robust database transactions, and zero mystery abstractions.
2. **Complex Database Architectures:** Systems requiring hybrid search, PostgreSQL RRF, row-level security (RLS), multi-tenant partitioning, or audit logs.
3. **Long-Term Maintainability:** Codebases where stability across 3–5 years is required without dependency churn or breaking framework rewrites.
4. **Verifiable Citations & Grounding:** Systems where safety, provenance tracing, and auditability are non-negotiable.

---

## 45-Minute Teach-Back Presentation Outline

1. **Minutes 00–10: The Retriever Concept**
   - Why not just `vectorstore.search()`?
   - The `BaseRetriever` contract.
   - Evolution: Vector $\to$ Ensemble (RRF) $\to$ Contextual Compression (Reranking).
2. **Minutes 10–20: LangChain Runnables & LCEL**
   - The `Runnable` lifecycle (`invoke`, `batch`, `stream`).
   - How `|` creates execution DAGs.
   - Code walkthrough of a standard LCEL chain.
3. **Minutes 20–35: Deep Dive into Our Plain Python Architecture**
   - Walk through `SearchService` (pgvector + PostgreSQL full-text search).
   - Walk through `ContextEngineeringService` (deduplication, budget trimming, source formatting).
   - Walk through `RAGService` (orchestration, citation validation, "I don't know" handling).
4. **Minutes 35–45: Production Trade-Offs, Live Q&A, and Decision Matrix**
   - Debugging nightmare: LangChain call stack vs. Python call stack.
   - Discussion: When should our team choose one over the other?
