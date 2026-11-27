import logging
from time import perf_counter
from typing import Any
from groq import APIError as GroqAPIError, AsyncGroq
from openai import APIError as OpenAIAPIError, AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.core.logging import log_llm_call
from app.infrastructure.llm_providers.registry import LLMProviderRegistry, default_registry
from app.modules.agents.adapters.retrieval_adapter import RetrievalAdapter
from app.modules.agents.config import AgentConfig, default_agent_config
from app.modules.agents.graph import run_agent_graph
from app.modules.agents.nodes import LLMCaller, normalize_openai_chat_completion
from app.modules.agents.schemas import AgentChatRequest, AgentChatResponse, ToolCallRecord
from app.modules.agents.state import AgentState
from app.modules.prompts.service import PromptService, default_prompt_service

logger = logging.getLogger(__name__)


class AgentToolService:
    """Orchestrates LangGraph tool-calling workflows using configured LLM providers and backend tools."""

    def __init__(
        self,
        registry: LLMProviderRegistry | None = None,
        prompt_service: PromptService | None = None,
        config: AgentConfig | None = None,
        retrieval_adapter: RetrievalAdapter | None = None,
    ) -> None:
        self.registry = registry or default_registry
        self.prompt_service = prompt_service or default_prompt_service
        self.config = config or default_agent_config
        self.retrieval_adapter = retrieval_adapter

    def get_agent_system_prompt(self) -> str:
        """Dynamically load the agent order support prompt template from the database."""
        return self.prompt_service.get_system_prompt("agent_order_support")

    @staticmethod
    def _create_single_turn_caller(
        client: Any,
        provider: str,
        model: str,
    ) -> LLMCaller:
        """Extract a single LLM turn call without owning the multi-step tool loop."""

        async def call_llm(
            messages: list[dict[str, Any]],
            tools: list[dict[str, Any]],
        ) -> dict[str, Any]:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
            except (GroqAPIError, OpenAIAPIError, TimeoutError) as exc:
                raise LLMProviderError.from_upstream(provider, exc) from exc

            try:
                return normalize_openai_chat_completion(response, fallback_model=model)
            except ValueError as exc:
                raise LLMProviderError(
                    provider,
                    f"{provider.capitalize()} returned a malformed response: {exc}",
                    status_code=502,
                ) from exc

        return call_llm

    async def execute_agent_graph(self, payload: AgentChatRequest) -> AgentState:
        """Run the full LangGraph workflow and return the final AgentState for inspection."""
        provider_adapter = self.registry.get(payload.provider)
        if provider_adapter is None:
            raise LLMProviderError(
                payload.provider,
                f"Unsupported LLM provider: {payload.provider}",
                status_code=400,
            )

        if payload.provider not in self.config.supported_providers:
            raise LLMProviderError(
                payload.provider,
                f"Tool calling is only configured for {', '.join(self.config.supported_providers)} "
                f"(received '{payload.provider}').",
                status_code=400,
            )

        api_key = provider_adapter.require_api_key()
        model = provider_adapter.default_model
        system_prompt = self.get_agent_system_prompt()

        client_cls_map = {
            "groq": AsyncGroq,
            "openai": AsyncOpenAI,
        }
        client_cls = client_cls_map[payload.provider]

        async with client_cls(api_key=api_key, timeout=settings.request_timeout_seconds) as client:
            caller = self._create_single_turn_caller(
                client=client,
                provider=payload.provider,
                model=model,
            )
            return await run_agent_graph(
                llm_caller=caller,
                user_request=payload.message,
                system_prompt=system_prompt,
                provider=payload.provider,
                model=model,
                retrieval_adapter=self.retrieval_adapter,
                config=self.config,
            )

    async def execute_agent_chat(self, payload: AgentChatRequest) -> AgentChatResponse:
        """Execute the LangGraph agent workflow and verify execution_status before returning."""
        started_at = perf_counter()
        final_state = await self.execute_agent_graph(payload)

        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        used_model = final_state.get("model") or ""
        in_tok = final_state.get("prompt_tokens", 0)
        out_tok = final_state.get("completion_tokens", 0)
        execution_status = final_state.get("execution_status", "completed")
        errors = final_state.get("errors") or []

        if execution_status != "completed":
            logger.warning(
                "Agent graph finished with non-completed status=%s task_id=%s errors=%s",
                execution_status,
                final_state.get("task_id"),
                errors,
            )

        log_status = "success" if execution_status == "completed" else execution_status
        cost = log_llm_call(
            provider=payload.provider,
            model=used_model,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
            status=log_status,
            error=errors[-1] if execution_status != "completed" and errors else None,
        )

        tool_records = [
            ToolCallRecord(**record) for record in final_state.get("tool_results", [])
        ]

        return AgentChatResponse(
            reply=final_state.get("final_result") or "",
            provider=payload.provider,
            model=used_model,
            execution_status=execution_status,
            tool_calls=tool_records,
            total_latency_ms=latency_ms,
            estimated_cost_usd=cost,
        )


default_agent_tool_service = AgentToolService()


def get_agent_tool_service() -> AgentToolService:
    return default_agent_tool_service
