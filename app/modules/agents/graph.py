from typing import Any

from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, StateGraph

from app.modules.agents.adapters.retrieval_adapter import RetrievalAdapter
from app.modules.agents.config import AgentConfig, default_agent_config
from app.modules.agents.nodes import (
    LLMCaller,
    _build_trace_event,
    limit_reached_node,
    make_llm_node,
    make_tool_node,
    should_continue,
)
from app.modules.agents.state import AgentState, create_initial_state


def build_agent_graph(
    llm_caller: LLMCaller,
    *,
    tools: list[dict[str, Any]] | None = None,
    retrieval_adapter: RetrievalAdapter | None = None,
    config: AgentConfig | None = None,
):
    """
    Build and compile the LangGraph tool-calling workflow:
    START -> llm_node -> should_continue -> (tool_node -> llm_node | limit_reached_node -> END | END)
    """
    cfg = config or default_agent_config
    workflow = StateGraph(AgentState)
    workflow.add_node("llm_node", make_llm_node(llm_caller, tools=tools, config=cfg))
    workflow.add_node(
        "tool_node",
        make_tool_node(retrieval_adapter=retrieval_adapter, config=cfg),
    )
    workflow.add_node("limit_reached_node", limit_reached_node)

    workflow.add_edge(START, "llm_node")
    workflow.add_conditional_edges(
        "llm_node",
        should_continue,
        {
            "tool_node": "tool_node",
            "limit_reached": "limit_reached_node",
            END: END,
        },
    )
    workflow.add_edge("tool_node", "llm_node")
    workflow.add_edge("limit_reached_node", END)

    return workflow.compile()


async def run_agent_graph(
    *,
    llm_caller: LLMCaller,
    user_request: str,
    system_prompt: str,
    provider: str = "groq",
    model: str = "",
    tools: list[dict[str, Any]] | None = None,
    retrieval_adapter: RetrievalAdapter | None = None,
    config: AgentConfig | None = None,
    task_id: str | None = None,
) -> AgentState:
    """Initialize AgentState and invoke the compiled LangGraph workflow safely."""
    cfg = config or default_agent_config
    compiled = build_agent_graph(
        llm_caller,
        tools=tools,
        retrieval_adapter=retrieval_adapter,
        config=cfg,
    )
    initial_state = create_initial_state(
        user_request=user_request,
        system_prompt=system_prompt,
        provider=provider,
        model=model,
        max_tool_calls=cfg.max_tool_calls,
        task_id=task_id,
    )

    try:
        return await compiled.ainvoke(
            initial_state,
            config={"recursion_limit": cfg.max_graph_steps},
        )
    except GraphRecursionError:
        err_msg = (
            f"Execution stopped safely after exceeding max_graph_steps ({cfg.max_graph_steps})."
        )
        fallback = "I could not finish processing your request within the maximum allowed workflow steps."
        initial_state["execution_status"] = "failed"
        initial_state["errors"] = [*initial_state["errors"], err_msg]
        initial_state["final_result"] = fallback
        initial_state["messages"] = [
            *initial_state["messages"],
            {"role": "assistant", "content": fallback},
        ]
        initial_state["trace"] = [
            *initial_state["trace"],
            _build_trace_event(
                initial_state,
                node="graph",
                status="recursion_limit_exceeded",
            ),
        ]
        return initial_state
