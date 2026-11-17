from typing import Literal
from pydantic import BaseModel, Field

Provider = Literal["openai", "anthropic", "groq"]

MAX_MESSAGE_LENGTH = 10_000


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="User prompt sent to the LLM",
    )
    provider: Provider = Field(
        default="openai",
        description="Which LLM provider to route to",
    )


class ChatResponse(BaseModel):
    provider: Provider
    model: str
    reply: str
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0)
    estimated_cost_usd: float = Field(..., ge=0)
