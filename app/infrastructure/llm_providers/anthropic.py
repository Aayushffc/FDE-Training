from collections.abc import AsyncIterator
from time import perf_counter
from anthropic import APIError, AsyncAnthropic

from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.core.logging import log_llm_call
from app.infrastructure.llm_providers.base import BaseLLMProvider
from app.modules.chat.schemas import ChatRequest, ChatResponse
from app.modules.prompts.service import SYSTEM_PROMPT


class AnthropicProvider(BaseLLMProvider):
    """Adapter for Anthropic Messages API."""

    @property
    def provider_name(self) -> str:
        return "anthropic"

    @property
    def default_model(self) -> str:
        return settings.default_anthropic_model

    @property
    def api_key_env_var(self) -> str:
        return "ANTHROPIC_API_KEY"

    @property
    def raw_api_key(self) -> str | None:
        return settings.anthropic_api_key

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
            async with AsyncAnthropic(
                api_key=api_key,
                timeout=settings.request_timeout_seconds,
            ) as client:
                response = await client.messages.create(
                    model=model,
                    max_tokens=1024,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": message}],
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
        reply = "\n".join(block.text for block in response.content if block.type == "text")
        if not reply:
            log_llm_call(
                provider=self.provider_name,
                model=model,
                latency_ms=latency_ms,
                status="error",
                error="empty_response",
            )
            raise LLMProviderError(self.provider_name, "Anthropic returned an empty response.", status_code=502)

        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
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

        try:
            async with AsyncAnthropic(
                api_key=api_key,
                timeout=settings.request_timeout_seconds,
            ) as client:
                async with client.messages.stream(
                    model=model,
                    max_tokens=1024,
                    system=sys_prompt,
                    messages=[{"role": "user", "content": message}],
                ) as stream:
                    async for text in stream.text_stream:
                        yield text
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



anthropic_provider = AnthropicProvider()


async def generate_chat_completion(
    payload_or_message: ChatRequest | str,
    system_prompt: str | None = None,
) -> ChatResponse:
    return await anthropic_provider.generate_chat_completion(payload_or_message, system_prompt=system_prompt)
