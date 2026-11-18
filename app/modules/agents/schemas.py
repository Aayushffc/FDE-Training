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
        description="User question or request sent to the tool-enabled agent",
    )
    provider: Provider = Field(
        default="groq",
        description="LLM provider to route to (default: groq)",
    )


class AgentChatResponse(BaseModel):
    reply: str
    provider: Provider
    model: str
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    total_latency_ms: float = Field(..., ge=0)
    estimated_cost_usd: float = Field(..., ge=0)
