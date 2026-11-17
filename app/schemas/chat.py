from typing import Literal
from pydantic import BaseModel, Field

Provider = Literal["openai", "anthropic", "groq"]


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User prompt sent to the LLM")
    provider: Provider = Field(
        default="openai", description="Which LLM provider to route to"
    )


class ChatResponse(BaseModel):
    provider: Provider
    model: str
    reply: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    estimated_cost_usd: float
