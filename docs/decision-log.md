# Architecture Decision Records (ADR) & Technical Decision Log

This document records the architectural and technical decisions made during the design and implementation of the `fde-llm-gateway`, detailing context, alternatives considered, trade-offs, and final rationale.

---

## ADR-001: Server-Sent Events (SSE) for LLM Streaming

### Context
Users require incremental token streaming from LLM providers (OpenAI, Anthropic, Groq) to minimize perceived latency (Time to First Token) rather than waiting for multi-second complete responses.

### Alternatives Considered
1. **WebSockets (`ws://`):** Full-duplex bidirectional protocol over a persistent TCP connection.
   - *Pros:* Bidirectional messaging; supports client interrupts mid-stream.
   - *Cons:* Heavy protocol overhead; bypasses standard HTTP caching; breaks edge load balancers/CDNs; complicates HTTP header authentication and middleware logging.
2. **Raw Chunked HTTP / NDJSON (`Transfer-Encoding: chunked`):** Emitting newline-delimited JSON objects over a standard HTTP response.
   - *Pros:* Simple text stream.
   - *Cons:* No standardized event typing, retry IDs, or automatic client-side reconnection.
3. **Server-Sent Events (`text/event-stream`) [SELECTED]:** Unidirectional HTTP streaming protocol standardized in HTML5.
   - *Pros:* Built on standard HTTP; works through proxies and firewalls without special upgrades; client `EventSource` handles reconnection natively; mirrors native provider streaming formats (`data: {...}`).

### Decision & Rationale
We adopted **Server-Sent Events (`text/event-stream`)** exposed via `StreamingResponse` in FastAPI. It aligns directly with the upstream streaming protocols of OpenAI, Groq, and Anthropic, preserves our HTTP middleware logging, and requires zero additional transport infrastructure.

---

## ADR-002: Pydantic Schema-Driven Function Calling vs. Raw JSON Schemas

### Context
The LLM gateway needs to support autonomous tool execution (e.g., `get_order_status(order_id)`). When the model invokes a tool, its arguments must be validated and executed safely on the backend.

### Alternatives Considered
1. **Raw Python Dictionaries & Manual Parsing:**
   - *Pros:* Zero dependencies.
   - *Cons:* Highly brittle, boilerplate-heavy, susceptible to `KeyError`, and prone to silent type coercion bugs.
2. **`jsonschema` Validation Library:**
   - *Pros:* Standard JSON schema validation.
   - *Cons:* Separate schema definitions required for Python type hints and validation; verbose error formatting.
3. **Pydantic Model Schema Generation & Validation [SELECTED]:**
   - *Pros:* Single source of truth. Models declare typing with `pydantic.BaseModel` and automatically generate standard OpenAI function schemas via `.model_json_schema()`. Validation errors return structured diagnostics.

### Decision & Rationale
We selected **Pydantic v2 Models**. Each agent tool defines an explicit arguments schema (e.g. `OrderStatusArgs`). The tool loop generates the OpenAPI-compliant tool specification for the provider and validates the LLM's arguments before execution, returning validation errors back to the model for self-correction.

---

## ADR-003: Clean Architecture Layer Separation (API vs Modules vs Infrastructure)

### Context
Earlier iterations of the codebase mingled FastAPI route handling, provider API key resolution, prompt formatting, and SDK invocation in flat service files.

### Alternatives Considered
1. **Flat `services/` Structure:**
   - *Pros:* Quick to scaffold.
   - *Cons:* High coupling; chat business logic directly imports third-party SDK clients (`openai`, `anthropic`, `groq`), violating the Dependency Inversion Principle.
2. **Full Hexagonal / Ports & Adapters Architecture:**
   - *Pros:* Complete isolation of domain core.
   - *Cons:* Excessive boilerplate (interfaces, DTOs, factories) for a lightweight gateway.
3. **Pragmatic Clean Architecture (`api/`, `modules/`, `infrastructure/`, `core/`) [SELECTED]:**
   - *Pros:*
     - `app/api/v1/`: Presentation layer (HTTP routes, FastAPI dependencies, status codes).
     - `app/modules/`: Business capabilities (`chat/`, `prompts/`, `agents/`).
     - `app/infrastructure/`: Adapters for external dependencies (`llm_providers/`, `observability/`, `persistence/`).
     - `app/core/`: Cross-cutting concerns (config, logging, base exceptions).

### Decision & Rationale
Adopted the **Pragmatic Clean Architecture**. `ChatService` in `modules/chat` depends on the `LLMProvider` abstract contract and `ProviderRegistry` from `infrastructure/llm_providers/`. It never directly instantiates SDK clients. Swapping or mocking a provider requires zero changes to route handlers or business logic.

---

## ADR-004: Versioned Template Prompt Library vs. In-Code String Constants

### Context
Prompt engineering experiments require comparing multiple prompt variants (V1 Basic, V2 Detailed Instructions, V3 Few-Shot Examples) and versioning prompts alongside application code.

### Alternatives Considered
1. **Hardcoded Python String Literals:**
   - *Pros:* Convenient for prototyping.
   - *Cons:* Clutters business logic, violates DRY, requires code deployment to change a prompt word, and prevents non-developer auditing.
2. **Database-Driven Prompt Storage:**
   - *Pros:* Dynamic updates without code restart.
   - *Cons:* Adds database operational overhead, lacks Git-based version history, and complicates local testing.
3. **File-Based Versioned Prompt Library (`templates/{category}/{version}.txt`) [SELECTED]:**
   - *Pros:* Versioned in Git; plain text files readable by domain experts; loaded dynamically by `PromptService`; supports category namespaces; cached in memory with file modification timestamps.

### Decision & Rationale
Adopted **File-Based Versioned Prompts** rooted in `app/modules/prompts/templates/`. The `PromptService` provides automatic fallback to default versions, lists available categories/versions, and ensures zero hardcoded prompt strings exist in Python code.

---

## ADR-005: Upstream Error Normalization and Secret Redaction

### Context
When upstream LLM providers fail (e.g. invalid API keys, rate limits, server crashes), their SDKs raise provider-specific exceptions (`openai.RateLimitError`, `anthropic.AuthenticationError`, `groq.InternalServerError`) that can leak sensitive tokens, internal URLs, or raw tracebacks in HTTP error bodies.

### Alternatives Considered
1. **Catch-All 500 Handler:** Return generic `"Internal Server Error"` for everything.
   - *Cons:* Hides actionable feedback (e.g., client 429 rate limit or 422 schema error) from legitimate callers.
2. **Pass Upstream Exceptions Directly:**
   - *Cons:* Leaks bearer tokens, internal endpoints, and creates inconsistent JSON schemas across providers.
3. **Domain Exception Translation + Regex Secret Redaction [SELECTED]:**
   - *Pros:* Each provider adapter maps upstream exceptions to unified gateway errors (`LLMProviderError`, `RateLimitError`, `AuthenticationError`, `ProviderTimeoutError`). A centralized exception handler scrubs any potential secrets (`sk-...`, `gsk_...`) from error messages before emitting standard JSON payloads.

### Decision & Rationale
Adopted **Centralized Domain Exception Mapping with Secret Redaction** in `app/core/exceptions.py`. Callers receive uniform `{ "detail": "...", "provider": "...", "status_code": ... }` responses with HTTP status codes matching RFC semantics without security exposure.

---

## ADR-006: Strict API Versioning (`/v1`) without Legacy Route Fallbacks

### Context
Initial implementations mounted both `/chat` and `/v1/chat` to maintain backward compatibility.

### Alternatives Considered
1. **Dual Mount (`/chat` + `/v1/chat`):**
   - *Cons:* Violates single source of truth; prevents independent evolution of v2 contracts; duplicates OpenAPI schema endpoints.
2. **Strict `/v1` Prefix [SELECTED]:**
   - *Pros:* Explicit contract versioning; root `/` cleanly redirects to `/docs`; endpoints are strictly grouped under `/v1/chat`, `/v1/health`, etc.

### Decision & Rationale
All active endpoints are mounted strictly under `/v1`. Legacy root endpoints return 404, preventing silent regressions and maintaining API cleanliness.
