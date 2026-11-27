import json
from time import perf_counter
from typing import Any, TypeVar
from pydantic import BaseModel

from app.core.exceptions import StructuredOutputValidationError
from app.modules.agents.adapters.retrieval_adapter import (
    RetrievalAdapter,
    default_retrieval_adapter,
)
from app.modules.agents.config import AgentConfig, default_agent_config
from app.modules.agents.schemas import (
    OrderDetails,
    OrderStatusArgs,
    SearchKnowledgeArgs,
)

SchemaT = TypeVar("SchemaT", bound=BaseModel)

# Mock order database for demonstration only.
# NOTE: Production order lookup must derive the authenticated customer/account
# context from the request and enforce ownership rather than trusting LLM input.
MOCK_ORDERS: dict[str, dict[str, Any]] = {
    "ORD123": {
        "order_id": "ORD123",
        "status": "Shipped",
        "carrier": "FedEx",
        "tracking_number": "FX-992817203",
        "estimated_delivery": "2026-11-20",
        "items": ["Ergonomic Mechanical Keyboard", "Braided USB-C Cable"],
        "total_usd": 159.99,
    },
    "ORD-456": {
        "order_id": "ORD-456",
        "status": "Delivered",
        "carrier": "UPS",
        "tracking_number": "1Z9999999999999999",
        "estimated_delivery": "2026-11-18",
        "items": ["4K Ultra-Wide 34-inch Monitor"],
        "total_usd": 699.00,
    },
    "ORD-789": {
        "order_id": "ORD-789",
        "status": "Processing",
        "carrier": "DHL Express",
        "tracking_number": None,
        "estimated_delivery": "2026-11-24",
        "items": ["Noise Cancelling Headphones (Midnight Black)"],
        "total_usd": 249.50,
    },
}

ORDER_STATUS_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "get_order_status",
        "description": (
            "Look up the current shipping status, delivery ETA, carrier, and items "
            "for a given customer order ID."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "string",
                    "description": "The unique customer order identifier, e.g. ORD123, ORD-456",
                }
            },
            "required": ["order_id"],
            "additionalProperties": False,
        },
    },
}

SEARCH_KNOWLEDGE_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "search_knowledge",
        "description": (
            "Search the customer-support knowledge base and policy documents using RAG "
            "for answers grounded in stored documentation with source citations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language question to look up in the knowledge base.",
                },
                "top_k": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "default": default_agent_config.default_rag_top_k,
                    "description": (
                        f"Number of top chunks to evaluate (1 to 10; default: {default_agent_config.default_rag_top_k})."
                    ),
                },
                "strategy": {
                    "type": "string",
                    "enum": ["hybrid", "vector", "keyword"],
                    "default": default_agent_config.default_rag_strategy,
                    "description": f"Retrieval strategy (default: {default_agent_config.default_rag_strategy}).",
                },
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
}

AVAILABLE_TOOLS = [ORDER_STATUS_TOOL_SCHEMA, SEARCH_KNOWLEDGE_TOOL_SCHEMA]


def parse_raw_arguments(
    tool_name: str,
    raw_arguments: dict[str, Any] | str,
) -> dict[str, Any]:
    """Parse raw tool JSON arguments once into a dictionary."""
    try:
        parsed = json.loads(raw_arguments) if isinstance(raw_arguments, str) else raw_arguments
    except Exception as exc:
        raise StructuredOutputValidationError(
            provider="agent_tools",
            message=f"Tool argument JSON parsing failed for '{tool_name}': {exc}",
        ) from exc

    if not isinstance(parsed, dict):
        raise StructuredOutputValidationError(
            provider="agent_tools",
            message=f"Tool arguments for '{tool_name}' must be a JSON object.",
        )
    return parsed


def _validate_tool_args(
    tool_name: str,
    parsed_args: dict[str, Any],
    schema_cls: type[SchemaT],
) -> SchemaT:
    """Validate pre-parsed tool arguments against a Pydantic schema."""
    try:
        return schema_cls.model_validate(parsed_args)
    except Exception as exc:
        raise StructuredOutputValidationError(
            provider="agent_tools",
            message=f"Tool argument validation failed for '{tool_name}': {exc}",
        ) from exc


def execute_get_order_status(args: OrderStatusArgs) -> OrderDetails:
    """Backend implementation of order lookup."""
    clean_id = args.order_id.strip().upper()
    if clean_id in MOCK_ORDERS:
        return OrderDetails(**MOCK_ORDERS[clean_id])

    for key, order_data in MOCK_ORDERS.items():
        if key.upper() == clean_id:
            return OrderDetails(**order_data)

    return OrderDetails(
        order_id=args.order_id,
        status="NotFound",
        carrier=None,
        tracking_number=None,
        estimated_delivery=None,
        items=[],
        total_usd=None,
    )


def _resolve_rag_params(
    args: SearchKnowledgeArgs,
    config: AgentConfig | None = None,
) -> tuple[int, str, bool]:
    cfg = config or default_agent_config
    top_k = args.top_k if args.top_k is not None else cfg.default_rag_top_k
    strategy = args.strategy if args.strategy is not None else cfg.default_rag_strategy
    return top_k, strategy, cfg.default_rag_rerank


def execute_search_knowledge(
    args: SearchKnowledgeArgs,
    retrieval_adapter: RetrievalAdapter | None = None,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """Delegate knowledge search synchronously to the existing RAGService via RetrievalAdapter."""
    adapter = retrieval_adapter or default_retrieval_adapter
    top_k, strategy, rerank = _resolve_rag_params(args, config=config)
    return adapter.search_knowledge(
        query=args.query,
        top_k=top_k,
        strategy=strategy,
        rerank=rerank,
    )


async def execute_search_knowledge_async(
    args: SearchKnowledgeArgs,
    retrieval_adapter: RetrievalAdapter | None = None,
    config: AgentConfig | None = None,
) -> dict[str, Any]:
    """Delegate knowledge search asynchronously without blocking the event loop."""
    adapter = retrieval_adapter or default_retrieval_adapter
    top_k, strategy, rerank = _resolve_rag_params(args, config=config)
    return await adapter.asearch_knowledge(
        query=args.query,
        top_k=top_k,
        strategy=strategy,
        rerank=rerank,
    )


def _execute_order_status_parsed(
    tool_name: str,
    parsed: dict[str, Any],
) -> dict[str, Any]:
    validated = _validate_tool_args(tool_name, parsed, OrderStatusArgs)
    return execute_get_order_status(validated).model_dump()


def dispatch_tool_call(
    tool_name: str,
    raw_arguments: dict[str, Any] | str,
    *,
    retrieval_adapter: RetrievalAdapter | None = None,
    config: AgentConfig | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Synchronously parse, validate, and execute a backend tool.
    Returns (result_dict, execution_latency_ms).
    """
    started_at = perf_counter()
    parsed = parse_raw_arguments(tool_name, raw_arguments)

    if tool_name == "get_order_status":
        result = _execute_order_status_parsed(tool_name, parsed)
    elif tool_name == "search_knowledge":
        validated = _validate_tool_args(tool_name, parsed, SearchKnowledgeArgs)
        result = execute_search_knowledge(
            validated,
            retrieval_adapter=retrieval_adapter,
            config=config,
        )
    else:
        raise ValueError(f"Unknown tool '{tool_name}' requested.")

    latency_ms = round((perf_counter() - started_at) * 1000, 2)
    return result, latency_ms


async def dispatch_tool_call_async(
    tool_name: str,
    raw_arguments: dict[str, Any] | str,
    *,
    retrieval_adapter: RetrievalAdapter | None = None,
    config: AgentConfig | None = None,
) -> tuple[dict[str, Any], float]:
    """
    Parse, validate, and execute a backend tool cleanly inside an async LangGraph node.
    Keeps fast in-memory lookups synchronous and awaits non-blocking I/O tools.
    """
    started_at = perf_counter()
    parsed = parse_raw_arguments(tool_name, raw_arguments)

    if tool_name == "get_order_status":
        result = _execute_order_status_parsed(tool_name, parsed)
    elif tool_name == "search_knowledge":
        validated = _validate_tool_args(tool_name, parsed, SearchKnowledgeArgs)
        result = await execute_search_knowledge_async(
            validated,
            retrieval_adapter=retrieval_adapter,
            config=config,
        )
    else:
        raise ValueError(f"Unknown tool '{tool_name}' requested.")

    latency_ms = round((perf_counter() - started_at) * 1000, 2)
    return result, latency_ms
