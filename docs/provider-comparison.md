# LLM Provider Comparison

Comparison of the three LLM providers integrated into the **FDE Sprint LLM Gateway** (`openai`, `anthropic`, and `groq`).

---

## 1. Provider Overview

| Dimension | OpenAI | Anthropic | Groq |
| :--- | :--- | :--- | :--- |
| **Provider Key** | `"openai"` | `"anthropic"` | `"groq"` |
| **SDK / Protocol** | `AsyncOpenAI` (OpenAI Chat Completions API) | `AsyncAnthropic` (Anthropic Messages API) | `AsyncOpenAI` pointed at `https://api.groq.com/openai/v1` (OpenAI-compatible) |
| **API Key Env Var** | `OPENAI_API_KEY` | `ANTHROPIC_API_KEY` | `GROQ_API_KEY` |
| **Model Env Var** | `OPENAI_MODEL` | `ANTHROPIC_MODEL` | `GROQ_MODEL` |
| **Default Model** | `gpt-6-luna` | `claude-haiku-5-5` | `llama-3.3-70b-versatile` |
| **System Prompt Field** | `messages[0]` with `{"role": "system", ...}` | Dedicated top-level `system=` parameter | `messages[0]` with `{"role": "system", ...}` |
| **Max Tokens Required?** | Optional | **Required** (`max_tokens=1024`) | Optional |
| **Response Content Shape** | `response.choices[0].message.content` | List of content blocks (`block.text` where `block.type == "text"`) | `response.choices[0].message.content` |
| **Token Usage Fields** | `prompt_tokens` / `completion_tokens` | `input_tokens` / `output_tokens` | `prompt_tokens` / `completion_tokens` |
| **Free Tier** | Paid credits required | Paid credits required | Free developer tier (rate-limited) |
| **Primary Strength** | Broad ecosystem & tool-calling maturity | Strong instruction following & long-context reasoning | Ultra-low latency / high tokens-per-second inference on LPU hardware |

---

## 2. Configured Model Pricing (`app/core/pricing.py`)

Costs are defined per **1,000,000 tokens** in USD. The example column estimates a typical **1,000 input / 500 output token** request.

| Provider | Model | Input (USD / 1M) | Output (USD / 1M) | Est. Cost (1k in / 500 out) |
| :--- | :--- | :---: | :---: | :---: |
| OpenAI | `gpt-6-luna` | $0.10 | $0.50 | $0.00035 |
| OpenAI | `gpt-4o-mini` | $0.15 | $0.60 | $0.00045 |
| Anthropic | `claude-haiku-5-5` | $0.10 | $0.50 | $0.00035 |
| Anthropic | `claude-haiku-4-5` | $1.00 | $5.00 | $0.00350 |
| Groq | `llama-3.3-70b-versatile` | $0.59 | $0.79 | $0.00099 |
| Groq | `llama-3.1-8b-instant` | $0.05 | $0.08 | $0.00009 |
| Groq | `openai/gpt-oss-120b` | $0.15 | $0.60 | $0.00045 |
| Groq | `openai/gpt-oss-20b` | $0.075 | $0.30 | $0.00023 |

> **Note:** If a model configured in `.env` is not present in `MODEL_PRICING_PER_1M` (`app/core/pricing.py`), `calculate_cost_usd()` logs a warning and returns `0.0` rather than failing the request.

---

## 3. Normalized Gateway Contract

Regardless of the underlying provider differences above, `POST /chat` exposes a single normalized request and response shape:

### Request (`POST /chat`)
Models and system instructions are controlled centrally via `.env` and `app/core/prompts.py` (not exposed as per-request overrides):

```json
{
  "message": "Explain idempotency in REST APIs in two sentences.",
  "provider": "groq"
}
```

### Response (`200 OK`)
```json
{
  "provider": "groq",
  "model": "openai/gpt-oss-120b",
  "reply": "...",
  "input_tokens": 95,
  "output_tokens": 40,
  "latency_ms": 604.45,
  "estimated_cost_usd": 0.00003825
}
```

---

## 4. Upstream Error Mapping

Both the OpenAI/Groq SDK and Anthropic SDK share Stainless-based exception hierarchies, which `LLMProviderError.from_upstream()` translates deliberately into consistent HTTP status codes:

| Upstream Condition | SDK Exception | Gateway Status | JSON `error` |
| :--- | :--- | :---: | :--- |
| Missing API key in `.env` | Config check | `503` | `llm_provider_error` |
| Request exceeded `REQUEST_TIMEOUT` | `APITimeoutError` / `TimeoutError` | `504` | `llm_provider_error` |
| Rate limit / quota exceeded | `RateLimitError` | `429` | `llm_provider_error` |
| Invalid or expired API key | `AuthenticationError` | `401` | `llm_provider_error` |
| Account / model access denied | `PermissionDeniedError` | `403` | `llm_provider_error` |
| Invalid parameters / context too long | `BadRequestError` | `400` | `llm_provider_error` |
| Upstream 5xx, network drop, empty reply | `InternalServerError` / `APIConnectionError` | `502` | `llm_provider_error` |
| Unhandled internal bug (e.g. `TypeError`) | `Exception` | `500` | `internal_error` |

---

## 5. When to Choose Each Provider

- **Groq** — Best for **rapid prototyping (free developer tier)** and **latency-sensitive workflows** (sub-second time-to-first-token on `llama-3.1-8b-instant`, `llama-3.3-70b-versatile`, and `openai/gpt-oss-120b`). Watch out for tighter free-tier requests-per-minute / tokens-per-minute limits.
- **OpenAI** — Best default for **general-purpose tasks, structured JSON outputs, and broad SDK compatibility** at low per-token cost (`gpt-4o-mini`, `gpt-6-luna`).
- **Anthropic** — Best for **nuanced instruction following, technical writing, and complex reasoning** where response quality outweighs raw token speed (`claude-haiku-5-5`, `claude-haiku-4-5`).

---

## 6. Local Benchmarking & Observability

Every HTTP request and upstream LLM call is recorded in `logs/app.log` (single rotating log file, 5 MB max):
- `api.access` logs end-to-end HTTP method, path, status code, and duration.
- `llm_observability` logs structured `LLM_CALL` JSON entries with `provider`, `model`, `latency_ms`, `input_tokens`, `output_tokens`, `estimated_cost_usd`, `status`, and `error`.
