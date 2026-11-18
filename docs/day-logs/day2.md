# Day 2 — Daily Progress Log
**Date:** 18 November 2026  
**Track:** AI Engineering / Forward Deployed Engineer  
**Member:** Member B — AI Agents  
**Focus:** Clean Architecture, Prompt Engineering, Streaming, Tool Calling, and Gateway Reliability

---

## 1. Work Completed

### Clean Architecture Refactoring
- Restructured the codebase according to Clean Architecture boundaries:
  - `app/api/v1/`: Presentation layer (FastAPI endpoints, router, dependencies).
  - `app/modules/`: Application business capabilities (`chat/`, `prompts/`, `agents/`).
  - `app/infrastructure/`: External integrations (`llm_providers/` with provider registry, base class, and adapter implementations for OpenAI, Anthropic, Groq).
  - `app/core/`: Cross-cutting concerns (`config.py`, `exceptions.py`, `logging.py`).
- Eliminated duplicate code and legacy fallback routes. Removed dual-mounting in favor of strictly versioned `/v1` routes (`/v1/chat`, `/v1/health`, etc.).
- Refactored `BaseLLMProvider` to adhere to DRY principles by centralizing API key validation, input normalization, and observability logging.

### Prompt Engineering & Versioned Prompt Library
- Implemented a versioned prompt library under `app/modules/prompts/templates/customer_support/`:
  - `v1.txt`: Baseline system instructions.
  - `v2.txt`: Structured instructions with explicit JSON output requirements and category tags.
  - `v3.txt`: Comprehensive system instructions with few-shot exemplars and edge case handling.
- Conducted experiments evaluating 10 identical customer support tickets across all 3 prompt versions (30 total inferences).
- Automated execution and evaluation using `scripts/run_prompt_experiments.py` and `scripts/analyze_results.py`.
- Documented findings in `docs/promt-experiment/prompt-experiments.md`.

### Real-Time Streaming Support (Server-Sent Events)
- Built streaming completion support using Server-Sent Events (`text/event-stream`).
- Implemented async chunk generators for `GroqProvider`, `OpenAIProvider`, and `AnthropicProvider`.
- Added dedicated endpoints:
  - `POST /v1/chat/stream`: Direct SSE streaming endpoint.
  - `POST /v1/chat` (`stream: true`): Dual-mode endpoint that switches dynamically between JSON response and SSE stream.
- Handled usage and token accumulation during streaming runs, logging token counts and estimated costs upon stream completion.
- Verified live streaming with `scripts/demonstrate_streaming.py` (`uv run stream-demo`).

### Autonomous Tool Calling (Agent Capabilities)
- Created `app/modules/agents/` containing tool definitions, Pydantic argument schemas, and an execution loop.
- Implemented `get_order_status` tool with strict argument validation via Pydantic model `OrderStatusArgs`.
- Built provider-agnostic multi-turn execution loop:
  1. LLM detects need for external data.
  2. Model requests tool call with JSON arguments.
  3. Gateway validates arguments with Pydantic and executes backend Python function.
  4. Tool result injected into conversation history.
  5. LLM generates final natural language answer grounded in tool output.
- Exposed `POST /v1/chat/tools` and `GET /v1/chat/tools` (schema discovery).
- Verified live tool calling with `scripts/demonstrate_tool_calling.py` (`uv run tool-demo`).

### Observability, Reliability & Security Documentation
- Configured structured JSON logging in `logs/app.log` capturing provider, model, input/output tokens, cost, latency, and request status for standard, streaming, and tool calls.
- Authored comprehensive engineering guides:
  - `docs/api-reliability.md`: HTTP status codes (400-503), rate limits, backoff with jitter, timeouts, and enterprise OAuth.
  - `docs/security-and-evaluations.md`: Hallucinations, ungrounded answers, direct prompt injection, and indirect prompt injection with root causes and mitigations.
  - `docs/decision-log.md`: Architecture Decision Records (ADR-001 through ADR-006).

---

## 2. Technical Decisions

- **Clean Architecture & Dependency Direction:** Application business logic in `app/modules/` depends on the abstract `LLMProvider` interface rather than concrete provider SDKs.
- **Server-Sent Events (SSE):** Selected over WebSockets for streaming to preserve HTTP infrastructure compatibility and match upstream LLM streaming standards.
- **Pydantic Tool Calling Schemas:** Used Pydantic v2 to auto-generate JSON schemas for tool definitions and validate LLM-generated arguments before execution.
- **File-Based Versioned Prompts:** Replaced in-code string constants with file-based templates versioned in Git.
- **Strict API Versioning:** Removed legacy root route fallbacks (`/chat`) in favor of strict `/v1` prefix.

---

## 3. Testing and Validation

- **Automated Test Suite:** 92 unit, API, and integration tests passing in ~0.70 seconds (`uv run test`).
- **Live Verifications:**
  - Streaming demo executed against live Groq API (`openai/gpt-oss-120b`).
  - Tool execution demo executed against live Groq API verifying tool dispatch and response synthesis.
  - Prompt experiment script executed across all 3 versions with token and latency tracking.

---

## 4. Challenges and Learnings

- **Streaming Token Accounting:** Different providers surface token counts during streaming differently (OpenAI requires `stream_options={"include_usage": True}`, Groq surfaces `chunk.usage` or `chunk.x_groq.usage`, and Anthropic provides usage on stream completion). Built provider-specific token extractors with deterministic fallback estimation.
- **Multi-Turn Tool Loops:** Ensured conversation context preserves both tool calls and tool role responses across consecutive API roundtrips.
- **Testing Dependency Isolation:** Isolated configuration settings in Pytest fixtures to prevent test suites from requiring active third-party API keys during unit and API tests.

---

## 5. Day 2 Outcome

The gateway is now fully modularized under Clean Architecture with production-ready streaming, schema-validated tool calling, file-based prompt management, complete observability, and 92 automated tests verifying API resilience and behavior.
