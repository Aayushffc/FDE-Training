from time import perf_counter

from openai import APIError, AsyncOpenAI

from app.config import settings
from app.core.exceptions import LLMProviderError
from app.core.logging import log_llm_call
from app.schemas.chat import DEFAULT_SYSTEM_PROMPT, ChatRequest, ChatResponse


async def generate_chat_completion(payload: ChatRequest) -> ChatResponse:
    provider = "openai"
    api_key = settings.openai_api_key
    if api_key is None:
        raise LLMProviderError(provider, "OPENAI_API_KEY environment variable is not set.", 503)

    model = payload.model or settings.default_openai_model
    system_prompt = payload.system_prompt or DEFAULT_SYSTEM_PROMPT
    started_at = perf_counter()

    try:
        async with AsyncOpenAI(
            api_key=api_key,
            timeout=settings.request_timeout_seconds,
        ) as client:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": payload.message},
                ],
            )
    except APIError as exc:
        latency_ms = (perf_counter() - started_at) * 1000
        log_llm_call(
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            status="error",
            error=type(exc).__name__,
        )
        raise LLMProviderError(provider, f"OpenAI request failed: {exc}") from exc

    latency_ms = (perf_counter() - started_at) * 1000
    if not response.choices:
        log_llm_call(
            provider=provider,
            model=model,
            latency_ms=latency_ms,
            status="error",
            error="empty_response",
        )
        raise LLMProviderError(provider, "OpenAI returned an empty response.")

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
        raise LLMProviderError(provider, "OpenAI returned an empty response.")

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