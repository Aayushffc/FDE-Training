import json
from time import perf_counter
from typing import Any
from groq import AsyncGroq
from openai import AsyncOpenAI

from app.core.config import settings
from app.core.exceptions import LLMProviderError
from app.core.logging import log_llm_call
from app.infrastructure.llm_providers.registry import LLMProviderRegistry, default_registry
from app.modules.agents.schemas import AgentChatRequest, AgentChatResponse, ToolCallRecord
from app.modules.agents.tools import AVAILABLE_TOOLS, dispatch_tool_call
from app.modules.prompts.service import PromptService, default_prompt_service


class AgentToolService:
    """Orchestrates LLM function/tool calling interactions with backend validation and execution."""

    def __init__(
        self,
        registry: LLMProviderRegistry | None = None,
        prompt_service: PromptService | None = None,
    ) -> None:
        self.registry = registry or default_registry
        self.prompt_service = prompt_service or default_prompt_service

    def get_agent_system_prompt(self) -> str:
        """Dynamically load the agent order support prompt template from the database."""
        return self.prompt_service.get_system_prompt("agent_order_support")

    async def _execute_tool_loop(
        self,
        client: Any,
        model: str,
        user_message: str,
        system_prompt: str,
    ) -> tuple[str, str, list[ToolCallRecord], int, int]:
        """
        Generic, provider-agnostic multi-step tool calling loop:
        1. Send prompt + tool declarations to LLM.
        2. Detect if LLM requests tool execution.
        3. Validate parameters & execute backend function.
        4. Append tool execution output back to conversation.
        5. Request final response synthesis from LLM.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ]

        tool_records: list[ToolCallRecord] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0

        # Step 1: Initial call with tool declarations
        response = await client.chat.completions.create(
            model=model,
            messages=messages,
            tools=AVAILABLE_TOOLS,
            tool_choice="auto",
        )
        if response.usage:
            total_prompt_tokens += response.usage.prompt_tokens
            total_completion_tokens += response.usage.completion_tokens

        first_choice = response.choices[0]
        first_msg = first_choice.message

        # Step 2: Check if LLM requested a tool call
        if first_msg.tool_calls:
            messages.append(first_msg)

            for call in first_msg.tool_calls:
                fn_name = call.function.name
                raw_args = call.function.arguments

                # Step 3: Validate arguments & execute backend function
                result_dict, exec_latency = dispatch_tool_call(fn_name, raw_args)
                parsed_args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args

                tool_records.append(
                    ToolCallRecord(
                        tool_name=fn_name,
                        arguments=parsed_args,
                        result=result_dict,
                        execution_latency_ms=exec_latency,
                    )
                )

                # Step 4: Append tool execution result back to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps(result_dict),
                })

            # Step 5: Final LLM synthesis
            final_response = await client.chat.completions.create(
                model=model,
                messages=messages,
            )
            if final_response.usage:
                total_prompt_tokens += final_response.usage.prompt_tokens
                total_completion_tokens += final_response.usage.completion_tokens

            final_reply = final_response.choices[0].message.content or ""
            return final_reply, final_response.model, tool_records, total_prompt_tokens, total_completion_tokens

        # If no tool was needed, return direct reply
        return first_msg.content or "", response.model, [], total_prompt_tokens, total_completion_tokens

    async def execute_agent_chat(self, payload: AgentChatRequest) -> AgentChatResponse:
        started_at = perf_counter()

        provider_adapter = self.registry.get(payload.provider)
        if provider_adapter is None:
            raise LLMProviderError(
                payload.provider,
                f"Unsupported LLM provider: {payload.provider}",
                status_code=400,
            )

        # DRY: Let provider adapter handle API key requirement validation
        api_key = provider_adapter.require_api_key()
        model = provider_adapter.default_model
        system_prompt = self.get_agent_system_prompt()

        if payload.provider == "groq":
            async with AsyncGroq(api_key=api_key, timeout=settings.request_timeout_seconds) as client:
                reply, model, tool_calls, in_tok, out_tok = await self._execute_tool_loop(
                    client=client,
                    model=model,
                    user_message=payload.message,
                    system_prompt=system_prompt,
                )
        elif payload.provider == "openai":
            async with AsyncOpenAI(api_key=api_key, timeout=settings.request_timeout_seconds) as client:
                reply, model, tool_calls, in_tok, out_tok = await self._execute_tool_loop(
                    client=client,
                    model=model,
                    user_message=payload.message,
                    system_prompt=system_prompt,
                )
        else:
            raise LLMProviderError(
                payload.provider,
                f"Tool calling is not currently configured for provider '{payload.provider}'.",
                status_code=400,
            )

        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        cost = log_llm_call(
            provider=payload.provider,
            model=model,
            latency_ms=latency_ms,
            input_tokens=in_tok,
            output_tokens=out_tok,
        )

        return AgentChatResponse(
            reply=reply,
            provider=payload.provider,
            model=model,
            tool_calls=tool_calls,
            total_latency_ms=latency_ms,
            estimated_cost_usd=cost,
        )


default_agent_tool_service = AgentToolService()


def get_agent_tool_service() -> AgentToolService:
    return default_agent_tool_service

