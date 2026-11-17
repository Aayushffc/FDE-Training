from time import perf_counter

from groq import APIError, AsyncGroq

from app.config import settings
from app.core.exceptions import LLMProviderError
from app.core.logging import log_llm_call
from app.core.prompts import SYSTEM_PROMPT
from app.schemas.chat import ChatRequest, ChatResponse


async def generate_chat_completion(payload: ChatRequest) -> ChatResponse:
    provider = "groq"
    api_key = settings.groq_api_key
    if api_key is None:
        raise LLMProviderError(provider, "GROQ_API_KEY environment variable is not set.", status_code=503)

    model = settings.default_groq_model
    started_at = perf_counter()

    try:
        async with AsyncGroq(
            api_key=api_key,
            timeout=settings.request_timeout_seconds,
        ) as client:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": payload.message},
                ],
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
    if not response.choices:
        log_llm_call(
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            status="error",
            error="empty_response",
        )
        raise LLMProviderError(provider, "Groq returned an empty response.", status_code=502)

    choice = response.choices[0]
    reply = choice.message.content or ""
    if not reply:
        log_llm_call(
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            status="error",
            error="empty_response",
        )
        raise LLMProviderError(provider, "Groq returned an empty response.", status_code=502)

    input_tokens = response.usage.prompt_tokens if response.usage else 0
    output_tokens = response.usage.completion_tokens if response.usage else 0
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
