from typing import Any, Literal, TypedDict
from uuid import uuid4

ExecutionStatus = Literal[
    "pending",
    "running",
    "completed",
    "limit_reached",
    "failed",
]


class AgentState(TypedDict):
    """Shared LangGraph workflow state passed across LLM and tool execution nodes."""

    task_id: str
    user_request: str
    provider: str
    model: str
    system_prompt: str
    messages: list[dict[str, Any]]
    tool_calls: list[dict[str, Any]]
    tool_results: list[dict[str, Any]]
    tool_call_count: int
    max_tool_calls: int
    execution_status: ExecutionStatus
    errors: list[str]
    final_result: str | None
    prompt_tokens: int
    completion_tokens: int
    trace: list[dict[str, Any]]


def create_initial_state(
    *,
    user_request: str,
    system_prompt: str,
    provider: str = "groq",
    model: str = "",
    max_tool_calls: int = 5,
    task_id: str | None = None,
) -> AgentState:
    """Build a clean, initialized AgentState for a new LangGraph execution."""
    return AgentState(
        task_id=task_id or str(uuid4()),
        user_request=user_request,
        provider=provider,
        model=model,
        system_prompt=system_prompt,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_request},
        ],
        tool_calls=[],
        tool_results=[],
        tool_call_count=0,
        max_tool_calls=max_tool_calls,
        execution_status="running",
        errors=[],
        final_result=None,
        prompt_tokens=0,
        completion_tokens=0,
        trace=[],
    )
