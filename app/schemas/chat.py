from typing import Literal
from pydantic import BaseModel, Field

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer clearly and accurately. "
    "If you are unsure, say so."
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt sent to the LLM")
    provider: Literal["openai", "anthropic"] = Field(
        default="openai", description="Which LLM provider to route to"
    )
    model: str | None = Field(
        default=None, description="Optional model override; uses provider default if omitted"
    )
    system_prompt: str | None = Field(
        default=None, description="Optional system instruction; uses a helpful default if omitted"
    )


class ChatResponse(BaseModel):
    provider: Literal["openai", "anthropic"]
    model: str
    reply: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost_usd: float