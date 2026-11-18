# API Reliability and Fundamentals for LLM Gateways

This document provides an engineering guide to API reliability, error handling, rate limiting, retry semantics, timeouts, and enterprise authorization when integrating with external Large Language Model (LLM) providers (OpenAI, Anthropic, Groq, Azure OpenAI, Google Vertex AI).

---

## 1. HTTP Status Codes in LLM Provider Integrations

LLM APIs exhibit distinct status code semantics compared to traditional CRUD REST APIs. Below is the operational mapping across the gateway and upstream providers:

| Status Code | Meaning | Context in LLM APIs | Gateway Action & Client Exposure |
|---|---|---|---|
| **400 Bad Request** | Malformed Request | Invalid JSON body, missing required parameters, unsupported sampling parameters (e.g., passing both `temperature=0` and unsupported top-p combinations), or invalid tool schema. | **Do NOT retry.** Return structured 400 with provider details to the caller for client-side correction. |
| **401 Unauthorized** | Authentication Failed | Missing API key, expired token, revoked credential, or whitespace in the authorization header. | **Do NOT retry.** Raise an alert immediately; redact the invalid secret and return 401 or 503 (service misconfiguration) to avoid leaking credentials. |
| **403 Forbidden** | Authorization / Access Denied | Account suspended, organization disabled, access to a gated model denied (e.g., GPT-4-32k without enterprise tier), or request originated from a geo-blocked region. | **Do NOT retry.** Surface domain error indicating permission restriction. |
| **404 Not Found** | Model or Resource Missing | Model ID does not exist, has been decommissioned/deprecated (e.g., `gpt-3.5-turbo-0613`), or endpoint URL typo. | **Do NOT retry.** Fail fast; caller must verify supported model catalog. |
| **422 Unprocessable Entity** | Semantic Validation Failure | Schema validation failed: message length exceeds context window prior to inference, Pydantic type mismatch, or invalid JSON schema passed in `tools` declaration. | **Do NOT retry.** Return field-level validation errors from Pydantic schema parser. |
| **429 Too Many Requests** | Rate Limit or Quota Exhausted | **Two distinct root causes:**<br>1. *Rate Limit:* Exceeded Requests Per Minute (RPM) or Tokens Per Minute (TPM).<br>2. *Quota Exhaustion:* Hard spending budget reached or billing balance empty. | **Conditional:**<br>- *Rate Limit:* Retry with Exponential Backoff + Jitter honoring `Retry-After`.<br>- *Quota Exhausted:* **Do NOT retry.** Alert ops; notify billing admin. |
| **500 Internal Server Error** | Upstream Inference Crash | Cluster-level GPU node panic, internal distributed memory allocation error, or upstream service fault. | **Retryable** with exponential backoff up to max attempts (e.g., 3 retries). |
| **502 Bad Gateway** | Edge Proxy / CDN Failure | Cloudflare or AWS API Gateway in front of provider dropped connection or upstream load balancer timed out. | **Retryable** with exponential backoff. |
| **503 Service Unavailable** | Model Overloaded | Provider cluster under extreme demand (e.g., Anthropic `overloaded_error`, Groq queue saturation). | **Retryable** with exponential backoff + jitter. If retries exhaust, return 503 to consumer. |
| **504 Gateway Timeout** | Inference / Response Timeout | Provider backend failed to return the first token or complete generation before HTTP socket timeout. | **Retryable** once or fail with fallback provider (e.g., Groq $\to$ OpenAI). |

---

## 2. Rate Limits & Quota Mechanics

Upstream providers enforce throttling across multiple dimensions concurrently:

### Throttling Dimensions
1. **Requests Per Minute (RPM):** Maximum HTTP transactions per 60-second sliding window.
2. **Tokens Per Minute (TPM):** Sum of input prompt tokens + output generated tokens in a sliding 60-second window. High-context prompts (e.g. 50k tokens) can trip TPM limits on a single request despite remaining under RPM limits.
3. **Requests Per Day (RPD):** Daily cap applied primarily to free or tier-1 developer accounts.
4. **Concurrent Streaming Connections:** Number of open HTTP SSE sockets allowed simultaneously.

### Provider Response Headers
LLM providers communicate quota state via HTTP response headers:
```http
x-ratelimit-limit-requests: 5000
x-ratelimit-remaining-requests: 4982
x-ratelimit-reset-requests: 216ms
x-ratelimit-limit-tokens: 2000000
x-ratelimit-remaining-tokens: 1845120
x-ratelimit-reset-tokens: 4.64s
retry-after: 5
```

### Rate Limit vs. Quota Exhaustion
* **Rate Limit (Transient):** Upstream returns 429 with `Retry-After: 5` or `x-ratelimit-reset-tokens: 4.2s`. Handled automatically by retry mechanisms.
* **Quota Exhaustion (Permanent until billed):** Upstream returns 429 with error type `insufficient_quota` (OpenAI) or `billing_error`. Retrying will **never** succeed and burns gateway resources. The gateway checks the error payload type before deciding to retry.

---

## 3. Retries and Idempotency

### Retry Decision Matrix
| Failure Mode | Retryable? | Strategy |
|---|---|---|
| HTTP 400, 401, 403, 404, 422 | ❌ No | Immediate rejection; return error to client. |
| HTTP 429 (Rate Limit: RPM/TPM) | ✅ Yes | Exponential backoff using `Retry-After` header value. |
| HTTP 429 (`insufficient_quota`) | ❌ No | Mark provider degraded; route to secondary provider. |
| HTTP 500, 502, 503 | ✅ Yes | Exponential backoff with jitter (max 3 attempts). |
| TCP Connect Timeout | ✅ Yes | Immediate retry on alternate provider IP / endpoint. |
| Mid-stream socket disconnection | ⚠️ Caution | Only retry if operation is idempotent and client disconnected cleanly. |

### Idempotency in LLM Generation
LLM completions are non-deterministic by nature unless pinned by a `seed` parameter. If a network blip occurs after the upstream model generated tokens but before the gateway receives the final byte:
- A blind retry may charge the account twice for the same user prompt.
- **Mitigation:**
  1. Pass client-generated `idempotency-key` in upstream request headers where supported (e.g., `Idempotency-Key` header).
  2. Pin `seed: 42` during regression or evaluation runs.
  3. Maintain an in-flight cache of request hashes (`sha256(provider + model + messages)`) for duplicate detection.

---

## 4. Exponential Backoff and Jitter

When a provider encounters a transient outage or load spike, thousands of client instances retrying simultaneously produce a **thundering herd** (synchronous retry spikes) that prolongs the outage.

### Standard Truncated Exponential Backoff
$$t_{\text{wait}} = \min(t_{\text{max}}, t_{\text{base}} \times 2^{\text{attempt}})$$

### Full Jitter Algorithm (Recommended)
To break the synchronization between competing workers, introduce random uniform jitter between $0$ and $t_{\text{wait}}$:
$$t_{\text{sleep}} = \text{random}(0, \min(t_{\text{max}}, t_{\text{base}} \times 2^{\text{attempt}}))$$

### Decorrelated Jitter Algorithm (Optimal for High Contention)
$$t_i = \min(t_{\text{max}}, \text{random}(t_{\text{base}}, t_{i-1} \times 3))$$

```python
import asyncio
import random

async def execute_with_jitter(fn, max_retries=3, base_delay=1.0, max_delay=16.0):
    delay = base_delay
    for attempt in range(max_retries):
        try:
            return await fn()
        except RetryableProviderError as exc:
            if attempt == max_retries - 1:
                raise
            # Full Jitter backoff calculation
            jittered_delay = random.uniform(0, min(max_delay, base_delay * (2 ** attempt)))
            await asyncio.sleep(jittered_delay)
```

---

## 5. Timeouts

LLM workloads have radically different latency profiles than traditional database queries. A single response may take 100ms or 60,000ms depending on token output size and provider queuing.

### Multi-Phase Timeout Architecture
1. **Connect Timeout (3 – 5 seconds):** Max time to establish TCP connection and complete TLS handshake. Prevents hanging when a provider host is down at the routing layer.
2. **Time to First Token (TTFT) / Read Timeout (15 – 30 seconds):** Max time waiting for upstream scheduling, prompt processing, and outputting the initial token.
3. **Inter-Chunk / Streaming Read Timeout (5 – 10 seconds):** During Server-Sent Events (SSE) streaming, if no byte is received for 10 seconds, the socket is presumed dead/stalled and severed.
4. **Total Request Ceiling (60 – 120 seconds):** Hard upper limit for the entire inference transaction to prevent worker starvation.

```python
import httpx

# Example configuration for production LLM client
transport = httpx.AsyncHTTPTransport(retries=0)
timeout = httpx.Timeout(
    connect=5.0,     # 5s TCP handshake limit
    read=30.0,       # 30s TTFT wait
    write=10.0,      # 10s request upload
    pool=10.0,       # 10s connection acquisition
)
```

---

## 6. OAuth 2.0 & Enterprise LLM Authorization

While consumer and developer endpoints primarily use static Bearer API keys (`Authorization: Bearer sk-...`), enterprise deployments (AWS Bedrock, Azure OpenAI, Google Cloud Vertex AI) mandate standard OAuth 2.0 and OpenID Connect (OIDC) identity flows.

### Enterprise Identity Comparison
| Pattern | Mechanism | Pros | Cons / Risks |
|---|---|---|---|
| **Static API Keys** | Shared secret string in `.env` | Simplest setup, zero handshake latency. | High blast radius on leak, manual rotation required, no fine-grained role scopes. |
| **OAuth 2.0 Client Credentials** | `POST /oauth/v2/token` exchanging `client_id` + `client_secret` for short-lived JWT (1 hour). | Auto-expiring tokens, granular scopes (`llm:read`, `llm:chat`), enterprise credential vaulting. | Requires token caching and refresh lifecycle management. |
| **Workload Identity Federation (OIDC)** | Service exchange of Kubernetes/AWS OIDC tokens for cloud IAM access tokens without long-lived keys. | Zero stored secrets; eliminates secret rotation entirely. | Requires Cloud IAM infrastructure and trust domain configuration. |
| **OAuth 2.0 Token Exchange (RFC 8693)** | Downstream gateway exchanges incoming user token for scoped LLM invocation token. | Preserves caller identity and audit trail down to LLM provider. | Higher authorization latency and token parsing overhead. |

### Token Lifecycle & Refresh Pattern
Enterprise gateways maintain an in-memory cached bearer token with a preemptive refresh worker:
- **Lifetime:** Standard JWT validity is 3600 seconds (1 hour).
- **Proactive Refresh:** Refresh token at $T - 300\text{s}$ (5 minutes prior to expiry) so active inference calls never hit an expired token during long generation runs.
