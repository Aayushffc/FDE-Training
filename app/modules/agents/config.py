from dataclasses import dataclass


@dataclass(frozen=True)
class AgentConfig:
    """Centralized configuration for LangGraph agent execution limits and defaults."""

    max_tool_calls: int = 5
    max_graph_steps: int = 12
    max_history_messages: int = 20
    default_provider: str = "groq"
    supported_providers: tuple[str, ...] = ("groq", "openai")
    tool_timeout_seconds: float = 15.0
    default_rag_top_k: int = 5
    default_rag_strategy: str = "hybrid"
    default_rag_rerank: bool = True


default_agent_config = AgentConfig()
