from time import perf_counter
from typing import Any
from pydantic import ValidationError

from app.core.exceptions import StructuredOutputValidationError
from app.modules.agents.schemas import OrderDetails, OrderStatusArgs, ToolCallRecord

# Mock order database for demonstration
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

AVAILABLE_TOOLS = [ORDER_STATUS_TOOL_SCHEMA]


def execute_get_order_status(args: OrderStatusArgs) -> OrderDetails:
    """Backend implementation of order lookup."""
    clean_id = args.order_id.strip().upper()
    if clean_id in MOCK_ORDERS:
        return OrderDetails(**MOCK_ORDERS[clean_id])

    # Check case-insensitive match
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


def dispatch_tool_call(tool_name: str, raw_arguments: dict[str, Any] | str) -> tuple[dict[str, Any], float]:
    """
    Validates arguments with Pydantic and executes the backend function.
    Returns (result_dict, execution_latency_ms).
    """
    started_at = perf_counter()

    if tool_name == "get_order_status":
        try:
            if isinstance(raw_arguments, str):
                import json
                parsed_args = json.loads(raw_arguments)
            else:
                parsed_args = raw_arguments
            validated = OrderStatusArgs.model_validate(parsed_args)
        except (ValidationError, Exception) as exc:
            raise StructuredOutputValidationError(
                provider="agent_tools",
                message=f"Tool argument validation failed for '{tool_name}': {exc}",
            ) from exc

        result = execute_get_order_status(validated)
        latency_ms = round((perf_counter() - started_at) * 1000, 2)
        return result.model_dump(), latency_ms

    raise ValueError(f"Unknown tool '{tool_name}' requested.")
