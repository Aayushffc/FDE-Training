# Full-Text Keyword Search vs. Semantic Vector Search Evaluation Report

This study evaluates two core information retrieval approaches implemented over a 30-document customer support FAQ dataset hosted on **Neon PostgreSQL**:
1. **Keyword Full-Text Search (FTS)**: Built natively using PostgreSQL `to_tsvector('english', ...)`, `websearch_to_tsquery('english', ...)`, and ranked via `ts_rank`.
2. **Semantic Vector Search**: Powered by Google Gemini (`gemini-embedding-001`, 768 output dimensions) and evaluated via pgvector cosine distance (`<=>`).

---

## 1. Technical Architecture Overview

| Dimension | Keyword Full-Text Search (FTS) | Semantic Vector Search (Gemini + pgvector) |
|---|---|---|
| **Mechanism** | Lexical tokenization, stemming (Snowball English stemmer), lexeme matching | Dense representation of meaning in 768-dimensional latent vector space |
| **PostgreSQL Features** | `tsvector`, `tsquery`, `websearch_to_tsquery`, `@@`, `ts_rank` | `vector(768)`, `<=>` (cosine distance), HNSW / IVFFlat indexes |
| **Endpoint** | `POST /v1/search/keyword` | `POST /v1/search/semantic` |
| **Computational Cost** | Very low CPU overhead; direct index lookup | Inference cost for query embedding via Gemini API + vector distance calc |
| **Key Strength** | Exact code/token matching, acronyms, product IDs, high precision | Conceptual reasoning, paraphrasing, synonyms, vocabulary gap bridging |

---

## 2. 10-Query Benchmark Results

Each of the following 10 queries was executed directly against the live **Neon PostgreSQL** database populated with our 30 curated FAQ documents.

| # | Query | Keyword Search Result | Semantic Search Result | Better Approach | Technical Observation |
|:--|:---|:---|:---|:---|:---|
| **1** | `"Forgot my login credentials"` | **No match** (rank: 0.0) | **Password Reset FAQ** (similarity: `0.6880`) | **Semantic** | **Vocabulary Gap:** Lexical search fails because the document uses *"reset your forgotten password"* and *"Forgot Password"*, rather than the word *"credentials"*. Vector search recognizes the conceptual equivalence between *"login credentials"* and *"password reset"*. |
| **2** | `"X-Signature-SHA256"` | **Webhook Security FAQ** (rank: `0.7183`) | **Webhook Security FAQ** (similarity: `0.7286`) | **Keyword / Both** | **Exact Token Precision:** Both find the document, but Keyword search is ideal here: exact alphanumeric identifiers and HTTP headers should never be fuzzy-matched or drifted by embedding proximity. |
| **3** | `"I want to stop paying for this service"` | **No match** (rank: 0.0) | **Subscription Cancellation FAQ** (similarity: `0.6834`) | **Semantic** | **Intent vs. Lexicon:** User expresses an outcome without using domain jargon (*"cancel"*, *"subscription"*, *"downgrade"*). FTS finds zero hits; vector embeddings map *"stop paying"* directly to cancellation semantics. |
| **4** | `"Credit card payment was declined"` | **Backup Payment FAQ** (rank: `0.4403`) | **Payment Failure & Retry Engine FAQ** (similarity: `0.5965`) | **Semantic** | **Context Relevance:** Keyword matched a secondary card FAQ simply because the word *"declined"* was present. Semantic search correctly prioritized the dedicated *"automated billing retry engine"* document addressing the root issue. |
| **5** | `"How do I configure Google Authenticator?"` | **2FA Security FAQ** (rank: `0.2808`) | **2FA Security FAQ** (similarity: `0.7548`) | **Both** | **Named Entity Overlap:** The document explicitly mentions *"Google Authenticator"*, so keyword search matches cleanly. Semantic search yields a strong score (`0.7548`) by understanding the setup guide intent. |
| **6** | `"HTTP 429 Too Many Requests"` | **API Rate Limit FAQ** (rank: `0.5807`) | **API Rate Limit FAQ** (similarity: `0.7115`) | **Keyword / Both** | **Exact Error Code:** Keyword matching is instant and accurate for verbatim HTTP status codes. Vector search also performs well due to dense semantic clustering around rate limit terminology. |
| **7** | `"Can I get my money back?"` | **No match** (rank: 0.0) | **14-day Refund Policy FAQ** (similarity: `0.6321`) | **Semantic** | **Phrasal Synonymy:** PostgreSQL FTS treats *"can"*, *"i"*, *"get"*, *"my"* as stopwords and fails on hyphenated *"money-back"*. Gemini's vector model understands colloquial requests for refunds effortlessly. |
| **8** | `"Exporting data to JSON or CSV"` | **Data Export FAQ** (rank: `0.0718`) | **Data Export FAQ** (similarity: `0.6916`) | **Both** | **Multi-Token Literal:** Both approaches successfully retrieve the workspace export FAQ because both the action (*"exporting"*) and format acronyms (*"JSON"*, *"CSV"*) are present in the text. |
| **9** | `"My team needs 5 more licenses"` | **No match** (rank: 0.0) | **Team Seat Management FAQ** (similarity: `0.6395`) | **Semantic** | **Terminology Mismatch:** User asks for *"licenses"*, whereas the product and FAQ consistently use *"seats"*. Keyword FTS fails completely; Gemini bridges the semantic equivalence between team licenses and seats. |
| **10** | `"Why is my account blocked after too many attempts?"` | **No match** (rank: 0.0) | **Account Lockout Protocol FAQ** (similarity: `0.7129`) | **Semantic** | **Paraphrased Incident:** The FAQ text uses *"temporarily locked after five consecutive incorrect password attempts"*. Keyword search misses due to word mismatch (*"blocked"* vs *"locked"*, *"attempts"* count). Semantic search identifies the exact situation. |

---

## 3. Comparative Scorecard

| Metric | Keyword Search (FTS) | Semantic Vector Search (Gemini) |
|---|---|---|
| **Queries with Relevant Matches** | 5 / 10 (50%) | 10 / 10 (100%) |
| **Vocabulary Gap Resilience** | Poor (requires verbatim or stemmed root) | Excellent (handles colloquialisms & synonyms) |
| **Exact Token & Identifier Precision** | Excellent (exact match on code/headers) | Good (may retrieve conceptually adjacent docs) |
| **Handling of Conversational Queries** | Poor (stops words + grammatical variance) | Excellent (trained on conversational intent) |
| **Average Query Latency** | ~2–5 ms (local PostgreSQL index) | ~150–300 ms (Gemini API embedding call + pgvector scan) |

---

## 4. Key Takeaways & Recommendations

1. **Why Semantic Search Won on Conversational Queries:**
   Customers rarely phrase support questions using exact documentation terminology. They ask *"Why am I blocked?"* instead of *"What is the account lockout policy?"*, and *"Can I get my money back?"* instead of *"14-day refund policy guarantee"*. Semantic search bridges this **vocabulary mismatch** that causes traditional keyword searches to return zero results.

2. **When Keyword Search Remains Superior:**
   When searching for **exact identifiers**, such as error codes (`HTTP 429`, `ERR_CONNECTION_TIMED_OUT`), configuration flags, variable names, or cryptographic headers (`X-Signature-SHA256`), keyword search provides 100% precision with sub-5ms response times.

3. **Production Recommendation: Hybrid Search (RRF):**
   In enterprise production architectures, combining both approaches via **Reciprocal Rank Fusion (RRF)** offers the optimal tradeoff:
   $$\text{RRF Score}(d) = \sum_{m \in \{\text{keyword}, \text{semantic}\}} \frac{1}{k + \text{rank}_m(d)}$$
   This guarantees that exact part numbers and codes are never lost, while natural-language and conversational questions always find their intended answer.
