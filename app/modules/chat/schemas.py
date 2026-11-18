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
    prompt_version: str | None = Field(
        default=None,
        description="Optional prompt version to test (e.g., 'v1', 'v2', 'v3', or 'customer_support/v2'). If omitted, the default system prompt is used.",
    )
    stream: bool = Field(
        default=False,
        description="Whether to stream response incrementally as Server-Sent Events (SSE).",
    )


class ChatResponse(BaseModel):
    provider: Provider
    model: str
    reply: str
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    latency_ms: float = Field(..., ge=0)
    estimated_cost_usd: float = Field(..., ge=0)
    prompt_version: str | None = Field(
        default=None,
        description="Prompt version applied to this completion",
    )


class StreamChunk(BaseModel):
    content: str = Field(default="", description="Incremental text token delta")
    is_complete: bool = Field(default=False, description="Whether this is the final stream termination chunk")
    provider: Provider | None = None
    model: str | None = None
    latency_ms: float | None = None
    estimated_cost_usd: float | None = None
    error: str | None = None
