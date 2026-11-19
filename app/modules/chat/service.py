from collections.abc import AsyncIterator
import json
import re
from time import perf_counter
from typing import Any

from app.infrastructure.llm_providers.registry import LLMProviderRegistry, default_registry
from app.modules.chat.exceptions import PromptNotFoundError, UnsupportedProviderError
from app.modules.chat.schemas import (
    ChatPromptVersionRequest,
    ChatPromptVersionResponse,
    ChatRequest,
    ChatResponse,
    ChatStreamRequest,
    SentimentAnalysisRequest,
    SentimentAnalysisResponse,
)
from app.modules.prompts.service import PromptService, default_prompt_service

# Mock handler mapping for test overrides
_PROVIDER_HANDLERS: dict[str, Any] = {}


class ChatService:
    """Application domain service for orchestrating chat, streaming, prompt experiments, and sentiment analysis."""

    def __init__(
        self,
        registry: LLMProviderRegistry | None = None,
        prompt_service: PromptService | None = None,
    ) -> None:
        self.registry = registry or default_registry
        self.prompt_service = prompt_service or default_prompt_service

    async def generate_chat_completion(self, payload: ChatRequest | ChatPromptVersionRequest) -> ChatResponse:
        """Handle standard conversational chat or prompt-version chat."""
        # Check if an explicit handler mock has been registered for tests
        if payload.provider in _PROVIDER_HANDLERS:
            res = await _PROVIDER_HANDLERS[payload.provider](payload)
            if getattr(payload, "prompt_version", None) and getattr(res, "prompt_version", None) is None:
                res.prompt_version = payload.prompt_version
            return res

        provider = self.registry.get(payload.provider)
        if provider is None:
            raise UnsupportedProviderError(payload.provider)

        resolved_prompt_version: str | None = None
        prompt_ver = getattr(payload, "prompt_version", None)
        if prompt_ver:
            raw_version = prompt_ver.strip()
            if "/" in raw_version:
                category, version = raw_version.split("/", 1)
            else:
                category = "customer_support"
                version = raw_version

            try:
                system_prompt = self.prompt_service.load_prompt(category, version)
                resolved_prompt_version = f"{category}/{version}"
            except FileNotFoundError:
                available = self.prompt_service.list_versions(category)
                raise PromptNotFoundError(
                    version=version,
                    category=category,
                    available=available,
                )
        else:
            system_prompt = self.prompt_service.get_system_prompt()
            resolved_prompt_version = "default"

        response = await provider.generate_chat_completion(payload.message, system_prompt=system_prompt)
        response.prompt_version = resolved_prompt_version
        return response

    async def generate_prompt_version_completion(
        self, payload: ChatPromptVersionRequest
    ) -> ChatPromptVersionResponse:
        """Execute a completion using a specific versioned prompt template."""
        res = await self.generate_chat_completion(payload)
        return ChatPromptVersionResponse(
            provider=res.provider,
            model=res.model,
            reply=res.reply,
            prompt_version=res.prompt_version or payload.prompt_version,
            input_tokens=res.input_tokens,
            output_tokens=res.output_tokens,
            latency_ms=res.latency_ms,
            estimated_cost_usd=res.estimated_cost_usd,
        )

    async def analyze_sentiment(
        self, payload: SentimentAnalysisRequest
    ) -> SentimentAnalysisResponse:
        """
        Analyze customer support ticket text for category, sentiment, urgency, churn risk,
        key entities, and recommended actions using structured prompt templates.
        """
        provider = self.registry.get(payload.provider)
        if provider is None:
            raise UnsupportedProviderError(payload.provider)

        # Load customer support v2 structured extraction prompt
        system_prompt = self.prompt_service.load_prompt("customer_support", "v2")
        raw_res = await provider.generate_chat_completion(payload.text, system_prompt=system_prompt)

        # Clean JSON markdown fences if the model outputted them
        cleaned_text = raw_res.reply.strip()
        fence_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", cleaned_text, re.DOTALL)
        if fence_match:
            cleaned_text = fence_match.group(1).strip()
        elif cleaned_text.startswith("{") and cleaned_text.endswith("}"):
            pass
        else:
            # Fallback search for first { to last }
            start = cleaned_text.find("{")
            end = cleaned_text.rfind("}")
            if start != -1 and end != -1:
                cleaned_text = cleaned_text[start : end + 1]

        try:
            parsed = json.loads(cleaned_text)
        except Exception:
            parsed = {
                "category": "Customer Support",
                "sentiment": "Neutral",
                "urgency": "Medium",
                "churn_risk": False,
                "key_entities": [],
                "summary": raw_res.reply[:200],
                "recommended_action": "Review inquiry and follow standard support protocol.",
            }

        return SentimentAnalysisResponse(
            category=str(parsed.get("category", "General Inquiry")),
            sentiment=str(parsed.get("sentiment", "Neutral")),
            urgency=str(parsed.get("urgency", "Medium")),
            churn_risk=bool(parsed.get("churn_risk", False)),
            key_entities=list(parsed.get("key_entities", [])),
            summary=str(parsed.get("summary", "Inquiry processed.")),
            recommended_action=str(parsed.get("recommended_action", "Follow up with customer.")),
            provider=raw_res.provider,
            model=raw_res.model,
            latency_ms=raw_res.latency_ms,
            estimated_cost_usd=raw_res.estimated_cost_usd,
        )

    async def stream_chat_completion(
        self, payload: ChatRequest | ChatStreamRequest
    ) -> AsyncIterator[str]:
        """Yield text token chunks incrementally from the resolved provider."""
        provider = self.registry.get(payload.provider)
        if provider is None:
            raise UnsupportedProviderError(payload.provider)

        prompt_ver = getattr(payload, "prompt_version", None)
        if prompt_ver:
            raw_version = prompt_ver.strip()
            if "/" in raw_version:
                category, version = raw_version.split("/", 1)
            else:
                category = "customer_support"
                version = raw_version

            try:
                system_prompt = self.prompt_service.load_prompt(category, version)
            except FileNotFoundError:
                available = self.prompt_service.list_versions(category)
                raise PromptNotFoundError(
                    version=version,
                    category=category,
                    available=available,
                )
        else:
            system_prompt = self.prompt_service.get_system_prompt()

        async for chunk in provider.stream_chat_completion(payload.message, system_prompt=system_prompt):
            yield chunk

    async def stream_sse(
        self, payload: ChatRequest | ChatStreamRequest
    ) -> AsyncIterator[str]:
        """Yield Server-Sent Events (SSE) formatted text chunks for streaming HTTP responses."""
        started_at = perf_counter()
        try:
            async for chunk in self.stream_chat_completion(payload):
                data = json.dumps({"content": chunk, "is_complete": False})
                yield f"data: {data}\n\n"

            latency_ms = round((perf_counter() - started_at) * 1000, 2)
            done_data = json.dumps({
                "content": "",
                "is_complete": True,
                "provider": payload.provider,
                "latency_ms": latency_ms,
            })
            yield f"data: {done_data}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as exc:
            err_data = json.dumps({
                "error": "llm_provider_error",
                "detail": str(exc),
                "is_complete": True,
            })
            yield f"data: {err_data}\n\n"
            yield "data: [DONE]\n\n"


default_chat_service = ChatService()


async def generate_chat_completion(payload: ChatRequest) -> ChatResponse:
    """Package-level convenience function for generating completions."""
    return await default_chat_service.generate_chat_completion(payload)


def get_chat_service() -> ChatService:
    """Dependency injection provider for FastAPI endpoints."""
    return default_chat_service
