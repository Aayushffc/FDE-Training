from typing import Any
from pydantic import BaseModel, Field

from app.modules.chat.schemas import Provider, MAX_MESSAGE_LENGTH


class OrderStatusArgs(BaseModel):
    """Pydantic schema to validate arguments passed by the LLM for get_order_status."""
    order_id: str = Field(
        ...,
        min_length=3,
        max_length=50,
        description="The unique order identifier string, e.g. ORD123",
    )


class OrderDetails(BaseModel):
    """Domain model representing order lookup output."""
    order_id: str
    status: str
    carrier: str | None = None
    tracking_number: str | None = None
    estimated_delivery: str | None = None
    items: list[str] = Field(default_factory=list)
    total_usd: float | None = None


class ToolCallRecord(BaseModel):
    """Trace record of a tool executed by the backend during an agent turn."""
    tool_name: str
    arguments: dict[str, Any]
    result: dict[str, Any]
    execution_latency_ms: float = Field(..., ge=0)


class AgentChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="User question or customer request sent to the tool-enabled agent.",
        examples=["What is the status of my order ORD123?"],
    )
    provider: Provider = Field(
        default="groq",
        description="LLM provider to route to (default: groq).",
        examples=["groq"],
    )

    model_config = {
        "json_schema_extra": {
            "example": {
                "message": "What is the status of my order ORD123?",
                "provider": "groq",
            }
        }
    }


class AgentChatResponse(BaseModel):
    reply: str = Field(
        ...,
        description="Final synthesized answer returned to the user",
        examples=["Order ORD123 has been Shipped via FedEx (Tracking: FX-992817203). Estimated delivery is 2026-11-20."],
    )
    provider: Provider = Field(..., examples=["groq"])
    model: str = Field(..., examples=["openai/gpt-oss-120b"])
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    total_latency_ms: float = Field(..., ge=0, examples=[1474.48])
    estimated_cost_usd: float = Field(..., ge=0, examples=[0.000192])

    model_config = {
        "json_schema_extra": {
            "example": {
                "reply": "Order ORD123 has been Shipped via FedEx (Tracking: FX-992817203). Estimated delivery is 2026-11-20.",
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "tool_calls": [
                    {
                        "tool_name": "get_order_status",
                        "arguments": {"order_id": "ORD123"},
                        "result": {
                            "order_id": "ORD123",
                            "status": "Shipped",
                            "carrier": "FedEx",
                            "tracking_number": "FX-992817203",
                            "estimated_delivery": "2026-11-20",
                            "items": ["Ergonomic Mechanical Keyboard", "Braided USB-C Cable"],
                            "total_usd": 159.99,
                        },
                        "execution_latency_ms": 0.07,
                    }
                ],
                "total_latency_ms": 1474.48,
                "estimated_cost_usd": 0.000192,
            }
        }
    }
