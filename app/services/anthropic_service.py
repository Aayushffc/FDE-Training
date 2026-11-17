from time import perf_counter

from anthropic import APIError, AsyncAnthropic

from app.config import settings
from app.core.exceptions import LLMProviderError
from app.core.logging import log_llm_call
from app.schemas.chat import DEFAULT_SYSTEM_PROMPT, ChatRequest, ChatResponse


async def generate_chat_completion(payload: ChatRequest) -> ChatResponse:
    provider = "anthropic"
    api_key = settings.anthropic_api_key
    if api_key is None:
        raise LLMProviderError(provider, "ANTHROPIC_API_KEY environment variable is not set.", status_code=503)

    model = payload.model or settings.default_anthropic_model
    system_prompt = payload.system_prompt or DEFAULT_SYSTEM_PROMPT
    started_at = perf_counter()

    try:
        async with AsyncAnthropic(
            api_key=api_key,
            timeout=settings.request_timeout_seconds,
        ) as client:
            response = await client.messages.create(
                model=model,
                max_tokens=1024,
                system=system_prompt,
                messages=[{"role": "user", "content": payload.message}],
            )
    except (APIError, TimeoutError) as exc:
        latency_ms = (perf_counter() - started_at) * 1000
        log_llm_call(
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            status="error",
            error=type(exc).__name__,
        )
        raise LLMProviderError.from_upstream(provider, exc) from exc

    latency_ms = (perf_counter() - started_at) * 1000
    reply = "\n".join(block.text for block in response.content if block.type == "text")
    if not reply:
        log_llm_call(
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            status="error",
            error="empty_response",
        )
        raise LLMProviderError(provider, "Anthropic returned an empty response.", status_code=502)

    input_tokens = response.usage.input_tokens
    output_tokens = response.usage.output_tokens
    cost = log_llm_call(
        provider=provider,
        model=model,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    return ChatResponse(
        provider=provider,
        model=response.model,
        reply=reply,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        latency_ms=latency_ms,
        estimated_cost_usd=cost,
    )
