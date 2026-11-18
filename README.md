# FDE Sprint LLM Gateway

A modular, production-grade **FastAPI** gateway wrapping **OpenAI**, **Anthropic**, and **Groq** behind a single normalized contract. Built with **Clean Architecture** separation of concerns, Dependency Inversion, real-time Server-Sent Events (SSE) streaming, autonomous tool calling, versioned prompt management, per-request observability, deliberate HTTP status mapping, secret redaction, and a 4-level test suite.

---

## Architecture & Directory Structure

The project structure strictly follows Clean Architecture principles: **The presentation and application layers define what they need; infrastructure implements it.**

```text
📁 fde-llm-gateway/
  📁 app/                           # Application source
    📁 api/                         # Presentation Layer (FastAPI routes & controllers)
      📁 v1/
        endpoints/chat.py           # Chat, streaming, and tool execution routes
        endpoints/health.py         # Health check endpoint
        router.py                   # Aggregates endpoints under api_v1_router (/v1)
        dependencies.py             # FastAPI dependency injection providers
    📁 modules/                     # Application & Business Capabilities
      📁 chat/
        schemas.py                  # ChatRequest and ChatResponse schemas
        service.py                  # ChatService business orchestration
        exceptions.py               # Domain chat exceptions
      📁 prompts/
        schemas.py                  # PromptTemplate models
        service.py                  # Dynamic versioned prompt loader & cache
        templates/                  # Versioned prompt library (customer_support v1/v2/v3)
      📁 agents/                    # Agent workflows & tool orchestration
        schemas.py                  # Tool definitions & execution result schemas
        service.py                  # AgentToolService provider-agnostic tool loop
        tools.py                    # Tool declarations & Pydantic argument schemas
    📁 infrastructure/              # External Integrations & Adapters
      📁 llm_providers/             # LLM provider implementations
        base.py                     # BaseLLMProvider abstract contract & DRY utilities
        openai.py                   # OpenAI AsyncOpenAI adapter
        anthropic.py                # Anthropic AsyncAnthropic adapter
        groq.py                     # Groq AsyncGroq adapter
        pricing.py                  # Per-model pricing catalog & USD calculator
        registry.py                 # LLMProviderRegistry (Dependency Inversion)
      📁 observability/             # Metrics and tracing adapters
      📁 persistence/               # Database and session storage adapters (when needed)
    📁 core/                        # Cross-Cutting Concerns
      config.py                     # Pydantic Settings & environment variables
      exceptions.py                 # Exception hierarchy, handlers & secret redactor
      logging.py                    # Structured JSON logging, ANSI colors & middleware
    main.py                         # FastAPI app bootstrap, middleware & /v1 mount
  📁 tests/
    📁 unit/                        # Schemas, exceptions, prompt library, registry, tools
    📁 api/                         # HTTP routes (/health, /chat, /stream, /tools)
    📁 integration/                 # Mocked provider adapters (OpenAI, Anthropic, Groq)
    📁 smoke/                       # Live network smoke tests (@pytest.mark.live)
  📁 docs/                          # Architecture, reliability, evaluations & day logs
  pyproject.toml                    # Project configuration, dependencies & CLI scripts
  uv.lock                           # Pinned reproducible dependency graph
  .env.example                      # Environment configuration template
  .gitignore                        # Git ignore patterns
  README.md                         # Project documentation
```

### Why These Folders Exist

#### A. `api/` — Presentation Layer
This is where FastAPI-specific code lives.
- **Responsibilities:**
  - Receive HTTP requests and parse route parameters.
  - Validate request payloads via Pydantic.
  - Call application logic through dependency injection.
  - Return HTTP responses and appropriate HTTP status codes.
  - Handle HTTP-specific concerns (content negotiation, CORS, response headers).
- **Boundary:** Contains **no** OpenAI/Anthropic/Groq SDK calls, prompt engineering strings, or business decisions. Think of this as the equivalent of your Node.js controllers and routes.

#### B. `modules/` — Application and Business Capabilities
This is where actual application features and business use cases live.
- `chat/`: Owns the chat completion use case, input validation, and provider dispatch orchestration.
- `prompts/`: Owns prompt management, dynamic template loading, category namespaces, and in-memory template caching.
- `agents/`: Owns autonomous tool calling, multi-turn LLM agent execution loops, and backend function dispatch.
- **Boundary:** More cohesive and scalable than dumping every business operation into a generic `services/` folder.

#### C. `infrastructure/` — External-World Integrations
One of the most important architectural boundaries.
- Adapters translate between the application's expected interface (`BaseLLMProvider`) and external vendor APIs (OpenAI, Anthropic, Groq).
- If OpenAI changes their SDK or if you swap a provider, application logic inside `modules/` remains untouched.
- Provider-specific pricing logic lives alongside provider infrastructure rather than polluting core.

#### D. `core/` — Cross-Cutting Concerns
Shared capabilities used across multiple modules:
- Configuration and environment management (`config.py`).
- Centralized exception definitions, global handlers, and secret redaction (`exceptions.py`).
- Structured logging setup, console formatters, and request middleware (`logging.py`).

---

## Core Capabilities

### 1. Unified Multi-Provider Gateway
Seamlessly route requests across `openai`, `anthropic`, and `groq` through `POST /v1/chat`. Normalized response includes generated text, token usage, latency, and estimated cost in USD.

### 2. Real-Time Streaming Support (SSE)
Incremental token streaming via Server-Sent Events (`text/event-stream`) through `POST /v1/chat/stream` or `POST /v1/chat` with `"stream": true`. Emits standardized event chunks (`data: {"content": "...", "is_complete": false}`) with full token accounting upon completion.

### 3. Autonomous Tool Calling (Function Calling)
Enables LLMs to query backend capabilities (e.g. `get_order_status`) via `POST /v1/chat/tools`. The gateway exposes OpenAPI schemas, validates LLM arguments using Pydantic, executes backend functions, feeds results back to the LLM, and returns the grounded answer.

### 4. Versioned Prompt Library
Dynamic template loading from `app/modules/prompts/templates/customer_support/` (`v1.txt`, `v2.txt`, `v3.txt`). Supports on-the-fly prompt selection via `"prompt_version": "v2"` and discovery via `GET /v1/chat/prompts`.

### 5. Production Observability
Structured JSON logging in `logs/app.log` tracking provider, model, input/output tokens, latency, cost, and request status across standard, streaming, and tool calls.

---

## Quickstart & Setup

### Prerequisites
- **Python** `>= 3.11`
- **[`uv`](https://docs.astral.sh/uv/)** package manager

### 1. Install Dependencies
```powershell
uv sync
```

### 2. Configure Environment
Copy `.env.example` to `.env` and set your API key(s):
```powershell
Copy-Item .env.example .env
```

| Variable | Required | Default | Description |
| :--- | :---: | :--- | :--- |
| `GROQ_API_KEY` | For `groq` | `None` | Groq API key ([console.groq.com](https://console.groq.com/keys)) |
| `OPENAI_API_KEY` | For `openai` | `None` | OpenAI API key |
| `ANTHROPIC_API_KEY` | For `anthropic` | `None` | Anthropic API key |
| `GROQ_MODEL` | No | `openai/gpt-oss-120b` | Default Groq model |
| `OPENAI_MODEL` | No | `gpt-6-luna` | Default OpenAI model |
| `ANTHROPIC_MODEL` | No | `claude-haiku-5-5` | Default Anthropic model |
| `REQUEST_TIMEOUT` | No | `30` | Upstream request timeout (seconds) |

### 3. Start the Server
```powershell
# Development mode with hot-reload (127.0.0.1:8000)
uv run dev

# Production mode (0.0.0.0:8000)
uv run start
```
Interactive Swagger documentation is available at **`http://127.0.0.1:8000/docs`** (root `/` automatically redirects here).

---

## CLI & Demo Commands

All workflows are pre-configured as runnable commands in `pyproject.toml`:

| Command | Action | Description |
| :--- | :--- | :--- |
| `uv run dev` | Server | Start Uvicorn development server with hot-reload |
| `uv run start` | Server | Start Uvicorn production server |
| `uv run test` | Testing | Run all 92 automated tests (unit, API, integration) |
| `uv run test-live` | Testing | Run live API smoke tests against real provider endpoints |
| `uv run test-all` | Testing | Run all 95 tests (automated + live smoke tests) |
| `uv run stream-demo` | Demo | Live demonstration of Server-Sent Events (SSE) streaming |
| `uv run tool-demo` | Demo | Live demonstration of 5-step autonomous tool execution loop |
| `uv run run-experiments` | Experiments | Execute 30-run prompt engineering benchmark on Groq |
| `uv run analyze-experiments` | Experiments | Generate comparative report from prompt benchmark data |

---

## API Reference

### Health Check
- **`GET /v1/health`**
```json
{
  "status": "ok"
}
```

### Chat Completion
- **`POST /v1/chat`**
```json
{
  "message": "My order #12345 arrived damaged, please refund me.",
  "provider": "groq",
  "prompt_version": "v2"
}
```
**Response (`200 OK`):**
```json
{
  "provider": "groq",
  "model": "openai/gpt-oss-120b",
  "reply": "{\n  \"category\": \"Billing & Payments\",\n  \"urgency\": \"High\",\n  \"sentiment\": \"Frustrated\",\n  \"churn_risk\": false,\n  \"key_entities\": [\"#12345\"],\n  \"summary\": \"Customer requests a refund for damaged order #12345.\",\n  \"recommended_action\": \"Initiate replacement or refund procedure.\"\n}",
  "input_tokens": 583,
  "output_tokens": 78,
  "latency_ms": 1535.14,
  "estimated_cost_usd": 0.00029595,
  "prompt_version": "customer_support/v2"
}
```

### Real-Time Streaming (SSE)
- **`POST /v1/chat/stream`** (or `POST /v1/chat` with `"stream": true`)
```json
{
  "message": "Count from 1 to 5.",
  "provider": "groq",
  "stream": true
}
```
**Event Stream Output:**
```text
data: {"content": "1", "is_complete": false}

data: {"content": " 2", "is_complete": false}

data: {"content": " 3", "is_complete": false}

data: {"content": "", "is_complete": true, "provider": "groq", "latency_ms": 312.4}

data: [DONE]
```

### Tool Calling Endpoint
- **`POST /v1/chat/tools`**
```json
{
  "message": "What is the status of my order ORD123?",
  "provider": "groq"
}
```
**Response (`200 OK`):**
```json
{
  "reply": "Order ORD123 has been Shipped via FedEx (Tracking: FX-992817203). Estimated delivery is 2026-11-20.",
  "provider": "groq",
  "model": "openai/gpt-oss-120b",
  "tool_calls": [
    {
      "tool_name": "get_order_status",
      "arguments": { "order_id": "ORD123" },
      "result": {
        "order_id": "ORD123",
        "status": "Shipped",
        "carrier": "FedEx",
        "tracking_number": "FX-992817203",
        "estimated_delivery": "2026-11-20"
      },
      "execution_latency_ms": 0.07
    }
  ],
  "total_latency_ms": 1474.48,
  "estimated_cost_usd": 0.000192
}
```
- **`GET /v1/chat/tools`** — Lists declared tools and schemas available to LLMs.
- **`GET /v1/chat/prompts`** — Lists available prompt versions and categories.

---

## Testing & Verification

The test suite covers unit logic, schema contracts, HTTP status code translation, exception mapping, secret redaction, and live provider smoke tests:

```powershell
uv run test
```

```text
============================= test session starts =============================
tests/api/test_chat.py .....................                            [ 22%]
tests/integration/test_anthropic_service.py .......                     [ 30%]
tests/integration/test_groq_service.py .......                          [ 38%]
tests/integration/test_openai_service.py .......                        [ 45%]
tests/unit/test_agent_tools.py .........                                [ 55%]
tests/unit/test_exceptions.py ........                                  [ 64%]
tests/unit/test_logging.py ........                                     [ 71%]
tests/unit/test_modules_and_infrastructure.py ............              [ 84%]
tests/unit/test_schemas.py ...............                             [100%]
====================== 92 passed, 3 deselected in 0.70s =======================
```

---

## Documentation Index

Comprehensive guides and architectural records are located in [`docs/`](docs/):

- **[API Reliability & Fundamentals (`docs/api-reliability.md`)](docs/api-reliability.md)** — HTTP status codes (`400`, `401`, `403`, `404`, `422`, `429`, `500`, `502`, `503`), Rate limit dimensions (TPM/RPM/RPD), Retries and idempotency, Exponential backoff with Full Jitter, Multi-phase timeouts, and OAuth 2.0 / Enterprise IAM.
- **[Security, Hallucinations & Injections (`docs/security-and-evaluations.md`)](docs/security-and-evaluations.md)** — Concrete examples, root causes, and mitigations for Hallucinations, Ungrounded Answers, Direct Prompt Injections (jailbreaks), and Indirect Prompt Injections (poisoned external context).
- **[Architecture Decision Records (`docs/decision-log.md`)](docs/decision-log.md)** — ADR-001 (Server-Sent Events), ADR-002 (Pydantic Tool Calling Schemas), ADR-003 (Pragmatic Clean Architecture), ADR-004 (Versioned Prompt Library), ADR-005 (Upstream Error Mapping & Secret Redaction), and ADR-006 (Strict `/v1` Route Versioning).
- **[Prompt Engineering Experiments (`docs/promt-experiment/prompt-experiments.md`)](docs/promt-experiment/prompt-experiments.md)** — Evaluation of 3 prompt versions across 10 identical benchmark tickets (30 live runs evaluated for correctness, consistency, hallucinations, and token cost).
- **[Provider Comparison Matrix (`docs/provider-comparison.md`)](docs/provider-comparison.md)** — Comparison of OpenAI, Anthropic, and Groq across pricing, latency, SDK ergonomics, context windows, and rate limits.
- **Daily Progress Logs:**
  - **[Day 1 Log (`docs/day-logs/day1.md`)](docs/day-logs/day1.md)** — Initial multi-provider gateway setup, testing, and observability.
  - **[Day 2 Log (`docs/day-logs/day2.md`)](docs/day-logs/day2.md)** — Clean Architecture refactor, SSE streaming, tool calling, prompt experiments, and reliability documentation.
