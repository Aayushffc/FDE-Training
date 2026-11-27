import asyncio
import json
import logging
from collections.abc import Awaitable, Callable
from datetime import datetime, timezone
from time import perf_counter
from typing import Any, Literal

from app.core.exceptions import StructuredOutputValidationError
from app.modules.agents.adapters.retrieval_adapter import RetrievalAdapter
from app.modules.agents.config import AgentConfig, default_agent_config
from app.modules.agents.state import AgentState
from app.modules.agents.tools import (
    AVAILABLE_TOOLS,
    dispatch_tool_call_async,
    parse_raw_arguments,
)

logger = logging.getLogger(__name__)

LLMCaller = Callable[
    [list[dict[str, Any]], list[dict[str, Any]]],
    Awaitable[dict[str, Any]],
]


def _build_trace_event(
    state: AgentState,
    *,
    node: str,
    status: str,
    duration_ms: float = 0.0,
    tool_call_id: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    """Create a consistent execution trace event across all graph nodes."""
    event: dict[str, Any] = {
        "task_id": state.get("task_id", ""),
        "node": node,
        "status": status,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "duration_ms": round(duration_ms, 2),
    }
    if tool_call_id is not None:
        event["tool_call_id"] = tool_call_id
    event.update(extra)
    return event


def _trim_task_messages(
    messages: list[dict[str, Any]],
    max_messages: int,
) -> list[dict[str, Any]]:
    """
    Bound task-level message history while preserving the system prompt, initial user
    request, and assistant-tool_call / tool-response group integrity.
    """
    if max_messages <= 2 or len(messages) <= max_messages:
        return list(messages)

    head = messages[:2]
    tail = messages[2:]

    # Group tail into atomic turns so an assistant tool_calls message is never split
    # from its corresponding role='tool' responses.
    groups: list[list[dict[str, Any]]] = []
    for msg in tail:
        if msg.get("role") == "tool" and groups:
            groups[-1].append(msg)
        else:
            groups.append([msg])

    budget = max_messages - len(head)
    selected_groups: list[list[dict[str, Any]]] = []
    used = 0
    for group in reversed(groups):
        if used + len(group) > budget and selected_groups:
            break
        selected_groups.append(group)
        used += len(group)

    flattened: list[dict[str, Any]] = []
    for group in reversed(selected_groups):
        flattened.extend(group)
    return head + flattened


def normalize_openai_chat_completion(
    response: Any,
    fallback_model: str = "",
) -> dict[str, Any]:
    """
    Normalize an OpenAI/Groq chat completion response (or test mock) into plain dicts.
    Validates choices, message, and tool-call structure defensively.
    """
    choices = getattr(response, "choices", None)
    if not choices:
        raise ValueError("Provider response is missing 'choices'.")

    msg = getattr(choices[0], "message", None)
    if msg is None:
        raise ValueError("Provider response choice is missing 'message'.")

    raw_tool_calls = getattr(msg, "tool_calls", None) or []
    normalized_calls: list[dict[str, Any]] = []

    for idx, call in enumerate(raw_tool_calls):
        if isinstance(call, dict):
            fn = call.get("function")
            if not isinstance(fn, dict):
                raise ValueError(f"Tool call at index {idx} has missing or invalid 'function' dict.")
            call_id = call.get("id")
            fn_name = fn.get("name")
            raw_args = fn.get("arguments")
        else:
            fn = getattr(call, "function", None)
            if fn is None:
                raise ValueError(f"Tool call at index {idx} is missing 'function'.")
            call_id = getattr(call, "id", None)
            fn_name = getattr(fn, "name", None)
            raw_args = getattr(fn, "arguments", None)

        if not call_id or not isinstance(call_id, str):
            raise ValueError(f"Tool call at index {idx} is missing a valid 'id'.")
        if not fn_name or not isinstance(fn_name, str):
            raise ValueError(f"Tool call '{call_id}' is missing a valid function 'name'.")
        if raw_args is None or not isinstance(raw_args, (str, dict)):
            raise ValueError(
                f"Tool call '{call_id}' ({fn_name}) has missing or invalid 'arguments'."
            )

        normalized_calls.append(
            {
                "id": call_id,
                "name": fn_name,
                "arguments": raw_args,
            }
        )

    usage = getattr(response, "usage", None)
    prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0 if usage else 0
    completion_tokens = getattr(usage, "completion_tokens", 0) or 0 if usage else 0

    return {
        "model": getattr(response, "model", None) or fallback_model,
        "content": getattr(msg, "content", None),
        "tool_calls": normalized_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
    }


def make_llm_node(
    llm_caller: LLMCaller,
    tools: list[dict[str, Any]] | None = None,
    config: AgentConfig | None = None,
):
    """Create the LangGraph LLM reasoning node for a single turn without an inner loop."""
    declared_tools = tools if tools is not None else AVAILABLE_TOOLS
    cfg = config or default_agent_config

    async def llm_node(state: AgentState) -> dict[str, Any]:
        started_at = perf_counter()
        bounded_messages = _trim_task_messages(
            state["messages"],
            max_messages=cfg.max_history_messages,
        )

        try:
            result = await llm_caller(bounded_messages, declared_tools)
        except ValueError as exc:
            duration_ms = (perf_counter() - started_at) * 1000
            logger.warning("Malformed LLM response during task %s: %s", state.get("task_id"), exc)
            safe_msg = "I received an invalid response from the language model and could not complete your request."
            return {
                "messages": [*state["messages"], {"role": "assistant", "content": safe_msg}],
                "tool_calls": [],
                "execution_status": "failed",
                "errors": [*state.get("errors", []), f"LLM normalization failed: {exc}"],
                "final_result": safe_msg,
                "trace": [
                    *state.get("trace", []),
                    _build_trace_event(
                        state,
                        node="llm_node",
                        status="error",
                        duration_ms=duration_ms,
                        error_code="INVALID_LLM_RESPONSE",
                    ),
                ],
            }

        duration_ms = (perf_counter() - started_at) * 1000
        content = result.get("content")
        requested_calls = result.get("tool_calls") or []
        model_name = result.get("model") or state.get("model", "")
        in_tok = int(result.get("prompt_tokens") or 0)
        out_tok = int(result.get("completion_tokens") or 0)

        if requested_calls:
            assistant_msg: dict[str, Any] = {
                "role": "assistant",
                "content": content or "",
                "tool_calls": [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": (
                                tc["arguments"]
                                if isinstance(tc["arguments"], str)
                                else json.dumps(tc["arguments"])
                            ),
                        },
                    }
                    for tc in requested_calls
                ],
            }
            next_status = "running"
            final_result = state.get("final_result")
            trace_status = "tool_calls_requested"
        else:
            assistant_msg = {
                "role": "assistant",
                "content": content or "",
            }
            next_status = "completed"
            final_result = content or ""
            trace_status = "completed"

        trace_entry = _build_trace_event(
            state,
            node="llm_node",
            status=trace_status,
            duration_ms=duration_ms,
            model=model_name,
            requested_tools=[tc["name"] for tc in requested_calls],
            prompt_tokens=in_tok,
            completion_tokens=out_tok,
        )

        return {
            "model": model_name,
            "messages": [*state["messages"], assistant_msg],
            "tool_calls": requested_calls,
            "execution_status": next_status,
            "final_result": final_result,
            "prompt_tokens": state.get("prompt_tokens", 0) + in_tok,
            "completion_tokens": state.get("completion_tokens", 0) + out_tok,
            "trace": [*state.get("trace", []), trace_entry],
        }

    return llm_node


def _safe_tool_error(code: str, message: str) -> dict[str, Any]:
    return {
        "error": True,
        "code": code,
        "message": message,
    }


def make_tool_node(
    retrieval_adapter: RetrievalAdapter | None = None,
    config: AgentConfig | None = None,
):
    """Create the async LangGraph tool execution node backed by dispatch_tool_call_async()."""
    cfg = config or default_agent_config

    async def tool_node(state: AgentState) -> dict[str, Any]:
        pending_calls = state.get("tool_calls") or []
        messages = list(state["messages"])
        tool_results = list(state.get("tool_results", []))
        errors = list(state.get("errors", []))
        trace = list(state.get("trace", []))
        tool_call_count = state.get("tool_call_count", 0)
        max_tool_calls = state.get("max_tool_calls", cfg.max_tool_calls)

        for call in pending_calls:
            call_id = call["id"]
            fn_name = call["name"]
            raw_args = call.get("arguments", "{}")

            if tool_call_count >= max_tool_calls:
                limit_msg = (
                    f"Maximum tool-call limit ({max_tool_calls}) reached before executing '{fn_name}'."
                )
                errors.append(limit_msg)
                safe_payload = _safe_tool_error(
                    "TOOL_CALL_LIMIT_REACHED",
                    "Maximum tool-call limit reached for this request.",
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call_id,
                        "content": json.dumps(safe_payload),
                    }
                )
                trace.append(
                    _build_trace_event(
                        state,
                        node="tool_node",
                        status="skipped_limit_reached",
                        tool_call_id=call_id,
                        tool_name=fn_name,
                    )
                )
                continue

            step_start = perf_counter()
            parsed_args: dict[str, Any] = {}
            status = "error"

            try:
                parsed_args = parse_raw_arguments(fn_name, raw_args)
                result_dict, exec_latency = await asyncio.wait_for(
                    dispatch_tool_call_async(
                        fn_name,
                        parsed_args,
                        retrieval_adapter=retrieval_adapter,
                        config=cfg,
                    ),
                    timeout=cfg.tool_timeout_seconds,
                )
                status = "success"
            except StructuredOutputValidationError as exc:
                logger.warning("Tool argument validation failed (task=%s, tool=%s): %s", state.get("task_id"), fn_name, exc)
                errors.append(f"Tool '{fn_name}' validation failed: {exc.message}")
                result_dict = _safe_tool_error(
                    "INVALID_TOOL_ARGUMENTS",
                    f"Invalid arguments for '{fn_name}'. Please verify required parameters and try again.",
                )
                exec_latency = round((perf_counter() - step_start) * 1000, 2)
            except TimeoutError:
                logger.warning("Tool timed out after %.1fs (task=%s, tool=%s)", cfg.tool_timeout_seconds, state.get("task_id"), fn_name)
                errors.append(f"Tool '{fn_name}' timed out after {cfg.tool_timeout_seconds}s.")
                result_dict = _safe_tool_error(
                    "TOOL_TIMEOUT",
                    "The requested tool timed out before completing.",
                )
                exec_latency = round((perf_counter() - step_start) * 1000, 2)
            except ValueError as exc:
                logger.warning("Unknown or unsupported tool requested (task=%s, tool=%s): %s", state.get("task_id"), fn_name, exc)
                errors.append(f"Tool '{fn_name}' rejected: {exc}")
                result_dict = _safe_tool_error(
                    "UNKNOWN_TOOL",
                    "The requested tool is not available.",
                )
                exec_latency = round((perf_counter() - step_start) * 1000, 2)
            except Exception:
                logger.exception("Unexpected failure executing tool '%s' (task=%s)", fn_name, state.get("task_id"))
                errors.append(f"Tool '{fn_name}' execution failed unexpectedly.")
                result_dict = _safe_tool_error(
                    "TOOL_EXECUTION_FAILED",
                    "The requested operation could not be completed.",
                )
                exec_latency = round((perf_counter() - step_start) * 1000, 2)

            tool_call_count += 1
            tool_results.append(
                {
                    "tool_call_id": call_id,
                    "tool_name": fn_name,
                    "arguments": parsed_args,
                    "result": result_dict,
                    "execution_latency_ms": exec_latency,
                }
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": call_id,
                    "content": json.dumps(result_dict),
                }
            )
            trace.append(
                _build_trace_event(
                    state,
                    node="tool_node",
                    status=status,
                    duration_ms=exec_latency,
                    tool_call_id=call_id,
                    tool_name=fn_name,
                )
            )

        return {
            "messages": messages,
            "tool_calls": [],
            "tool_results": tool_results,
            "tool_call_count": tool_call_count,
            "errors": errors,
            "trace": trace,
        }

    return tool_node


def limit_reached_node(state: AgentState) -> dict[str, Any]:
    """Terminate safely when the graph hits the configured maximum tool-call limit."""
    max_calls = state.get("max_tool_calls", 5)
    err_msg = (
        f"Execution stopped safely after reaching the maximum limit of {max_calls} tool calls."
    )
    fallback_text = (
        state.get("final_result")
        or "I reached the maximum tool execution limit before completing all requested steps. Please narrow your request."
    )

    messages = list(state["messages"])
    last_msg = messages[-1] if messages else None
    if not (
        last_msg
        and last_msg.get("role") == "assistant"
        and last_msg.get("content") == fallback_text
        and not last_msg.get("tool_calls")
    ):
        messages.append({"role": "assistant", "content": fallback_text})

    return {
        "messages": messages,
        "tool_calls": [],
        "execution_status": "limit_reached",
        "errors": [*state.get("errors", []), err_msg],
        "final_result": fallback_text,
        "trace": [
            *state.get("trace", []),
            _build_trace_event(
                state,
                node="limit_reached_node",
                status="limit_reached",
                tool_call_count=state.get("tool_call_count", 0),
            ),
        ],
    }


def should_continue(state: AgentState) -> Literal["tool_node", "limit_reached", "__end__"]:
    """Route after llm_node: stop on failure/final text, enforce tool-call cap, or run tools."""
    if state.get("execution_status") == "failed":
        return "__end__"
    if not state.get("tool_calls"):
        return "__end__"
    if state.get("tool_call_count", 0) >= state.get("max_tool_calls", 5):
        return "limit_reached"
    return "tool_node"
