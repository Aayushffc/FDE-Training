from collections.abc import AsyncIterator
from time import perf_counter
from openai import APIError, AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.core.logging import log_llm_call
from app.infrastructure.llm_providers.base import BaseLLMProvider
from app.modules.chat.schemas import ChatRequest, ChatResponse
from app.modules.prompts.service import SYSTEM_PROMPT


class OpenAIProvider(BaseLLMProvider):
    """Adapter for OpenAI Chat Completions API."""

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def default_model(self) -> str:
        return settings.default_openai_model

    @property
    def api_key_env_var(self) -> str:
        return "OPENAI_API_KEY"

    @property
    def raw_api_key(self) -> str | None:
        return settings.openai_api_key

    async def generate_chat_completion(
        self,
        payload_or_message: ChatRequest | str,
        system_prompt: str | None = None,
    ) -> ChatResponse:
        message, sys_prompt = self.normalize_inputs(payload_or_message, system_prompt)
        api_key = self.require_api_key()

        model = self.default_model
        started_at = perf_counter()

        try:
            async with AsyncOpenAI(
                api_key=api_key,
                timeout=settings.request_timeout_seconds,
            ) as client:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": message},
                    ],
                )
        except (APIError, TimeoutError) as exc:
            latency_ms = (perf_counter() - started_at) * 1000
            log_llm_call(
                provider=self.provider_name,
                model=model,
                latency_ms=latency_ms,
                status="error",
                error=type(exc).__name__,
            )
            raise LLMProviderError.from_upstream(self.provider_name, exc) from exc

        latency_ms = (perf_counter() - started_at) * 1000
        if not response.choices:
            log_llm_call(
                provider=self.provider_name,
                model=model,
                latency_ms=latency_ms,
                status="error",
                error="empty_response",
            )
            raise LLMProviderError(self.provider_name, "OpenAI returned an empty response.", status_code=502)

        choice = response.choices[0]
        reply = choice.message.content or ""
        if not reply:
            log_llm_call(
                provider=self.provider_name,
                model=model,
                latency_ms=latency_ms,
                status="error",
                error="empty_response",
            )
            raise LLMProviderError(self.provider_name, "OpenAI returned an empty response.", status_code=502)

        input_tokens = response.usage.prompt_tokens if response.usage else 0
        output_tokens = response.usage.completion_tokens if response.usage else 0
        cost = log_llm_call(
            provider=self.provider_name,
            model=model,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

        return ChatResponse(
            provider=self.provider_name,
            model=response.model,
            reply=reply,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            latency_ms=latency_ms,
            estimated_cost_usd=cost,
        )

    async def stream_chat_completion(
        self,
        payload_or_message: ChatRequest | str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[str]:
        message, sys_prompt = self.normalize_inputs(payload_or_message, system_prompt)
        api_key = self.require_api_key()

        model = self.default_model
        started_at = perf_counter()
        accumulated_text: list[str] = []
        input_tokens = max(1, (len(message) + len(sys_prompt)) // 4)
        output_tokens = 0

        try:
            async with AsyncOpenAI(
                api_key=api_key,
                timeout=settings.request_timeout_seconds,
            ) as client:
                stream = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": message},
                    ],
                    stream=True,
                    stream_options={"include_usage": True},
                )
                async for chunk in stream:
                    if hasattr(chunk, "usage") and chunk.usage:
                        input_tokens = getattr(chunk.usage, "prompt_tokens", input_tokens)
                        output_tokens = getattr(chunk.usage, "completion_tokens", output_tokens)
                    if chunk.choices and chunk.choices[0].delta.content:
                        text = chunk.choices[0].delta.content
                        accumulated_text.append(text)
                        yield text

            latency_ms = (perf_counter() - started_at) * 1000
            if output_tokens == 0:
                output_tokens = max(1, len("".join(accumulated_text)) // 4)

            log_llm_call(
                provider=self.provider_name,
                model=model,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                status="success",
            )
        except (APIError, TimeoutError) as exc:
            latency_ms = (perf_counter() - started_at) * 1000
            log_llm_call(
                provider=self.provider_name,
                model=model,
                latency_ms=latency_ms,
                status="error",
                error=type(exc).__name__,
            )
            raise LLMProviderError.from_upstream(self.provider_name, exc) from exc



openai_provider = OpenAIProvider()


async def generate_chat_completion(
    payload_or_message: ChatRequest | str,
    system_prompt: str | None = None,
) -> ChatResponse:
    return await openai_provider.generate_chat_completion(payload_or_message, system_prompt=system_prompt)
