from typing import Any, Literal
from pydantic import BaseModel, ConfigDict, Field

Provider = Literal["openai", "anthropic", "groq"]

MAX_MESSAGE_LENGTH = 10_000


# ---------------------------------------------------------------------------
# Simple Chat Schemas (pure conversational AI with default system prompt)
# ---------------------------------------------------------------------------


class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="User message sent to the AI assistant.",
        examples=["Explain what an API Gateway is in three simple sentences."],
    )
    provider: Provider = Field(
        default="openai",
        description="Which LLM provider to route to ('openai', 'groq', or 'anthropic').",
        examples=["groq"],
    )
    prompt_version: str | None = Field(
        default=None,
        description="Optional prompt template version. Left None for standard conversational chat.",
        examples=[None],
    )
    stream: bool = Field(
        default=False,
        description="Stream toggle (defaults to false for buffered JSON response).",
        examples=[False],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Explain what an API Gateway is in three simple sentences.",
                "provider": "groq",
            }
        }
    )


class ChatResponse(BaseModel):
    provider: Provider = Field(
        ...,
        description="Provider that processed the completion",
        examples=["groq"],
    )
    model: str = Field(
        ...,
        description="Model identifier used for inference",
        examples=["openai/gpt-oss-120b"],
    )
    reply: str = Field(
        ...,
        description="Conversational assistant response",
        examples=[
            "An API Gateway is a central entry point that sits between clients and backend services. "
            "It manages cross-cutting concerns such as routing, rate limiting, authentication, and observability. "
            "This simplifies client architecture and shields internal services from direct exposure."
        ],
    )
    input_tokens: int = Field(..., ge=0, description="Prompt token count", examples=[28])
    output_tokens: int = Field(..., ge=0, description="Completion token count", examples=[46])
    latency_ms: float = Field(..., ge=0, description="Generation latency in milliseconds", examples=[380.2])
    estimated_cost_usd: float = Field(..., ge=0, description="Estimated USD cost based on token pricing", examples=[0.000035])
    prompt_version: str | None = Field(
        default="default",
        description="Prompt version applied to this completion",
        examples=["default"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "reply": (
                    "An API Gateway is a central entry point that sits between clients and backend services. "
                    "It manages cross-cutting concerns such as routing, rate limiting, authentication, and observability. "
                    "This simplifies client architecture and shields internal services from direct exposure."
                ),
                "input_tokens": 28,
                "output_tokens": 46,
                "latency_ms": 380.2,
                "estimated_cost_usd": 0.000035,
                "prompt_version": "default",
            }
        }
    )


# ---------------------------------------------------------------------------
# Dedicated Streaming Schema (no stream boolean needed)
# ---------------------------------------------------------------------------


class ChatStreamRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="Prompt to generate an incremental token stream from.",
        examples=["Write a short 3-line poem about coffee and software engineering."],
    )
    provider: Provider = Field(
        default="groq",
        description="Which LLM provider to stream from ('groq', 'openai', or 'anthropic').",
        examples=["groq"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "Write a short 3-line poem about coffee and software engineering.",
                "provider": "groq",
            }
        }
    )


class StreamChunk(BaseModel):
    content: str = Field(default="", description="Incremental text token delta", examples=["Coffee "])
    is_complete: bool = Field(default=False, description="Whether this is the final stream termination chunk", examples=[False])
    provider: Provider | None = Field(default=None, examples=["groq"])
    model: str | None = Field(default=None, examples=["openai/gpt-oss-120b"])
    latency_ms: float | None = Field(default=None, examples=[284.1])
    estimated_cost_usd: float | None = Field(default=None, examples=[0.000045])
    error: str | None = None


# ---------------------------------------------------------------------------
# Prompt Version Experiment Schema
# ---------------------------------------------------------------------------


class ChatPromptVersionRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="Input text sent to the model using the selected prompt version template.",
        examples=["My shipment #12345 arrived damaged, please refund me."],
    )
    provider: Provider = Field(
        default="groq",
        description="Which LLM provider to route to ('groq', 'openai', or 'anthropic').",
        examples=["groq"],
    )
    prompt_version: str = Field(
        default="v2",
        description="Versioned prompt template to test (e.g., 'v1', 'v2', 'v3', or 'customer_support/v2').",
        examples=["v2"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "message": "My shipment #12345 arrived damaged, please refund me.",
                "provider": "groq",
                "prompt_version": "v2",
            }
        }
    )


class ChatPromptVersionResponse(BaseModel):
    provider: Provider = Field(..., examples=["groq"])
    model: str = Field(..., examples=["openai/gpt-oss-120b"])
    reply: str = Field(
        ...,
        description="Model output generated using the versioned prompt",
        examples=[
            "{\n  \"category\": \"Billing & Payments\",\n  \"urgency\": \"High\",\n  \"sentiment\": \"Frustrated\",\n  \"churn_risk\": false,\n  \"key_entities\": [\"#12345\"],\n  \"summary\": \"Customer requests a refund for damaged shipment #12345.\",\n  \"recommended_action\": \"Review order #12345 and initiate replacement or refund procedure.\"\n}"
        ],
    )
    prompt_version: str = Field(..., description="The prompt version executed", examples=["customer_support/v2"])
    input_tokens: int = Field(..., ge=0, examples=[583])
    output_tokens: int = Field(..., ge=0, examples=[78])
    latency_ms: float = Field(..., ge=0, examples=[1535.14])
    estimated_cost_usd: float = Field(..., ge=0, examples=[0.00029595])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "reply": "{\n  \"category\": \"Billing & Payments\",\n  \"urgency\": \"High\",\n  \"sentiment\": \"Frustrated\",\n  \"churn_risk\": false,\n  \"key_entities\": [\"#12345\"],\n  \"summary\": \"Customer requests a refund for damaged shipment #12345.\",\n  \"recommended_action\": \"Review order #12345 and initiate replacement or refund procedure.\"\n}",
                "prompt_version": "customer_support/v2",
                "input_tokens": 583,
                "output_tokens": 78,
                "latency_ms": 1535.14,
                "estimated_cost_usd": 0.00029595,
            }
        }
    )


# ---------------------------------------------------------------------------
# Dedicated Sentiment & Ticket Analysis Schemas
# ---------------------------------------------------------------------------


class SentimentAnalysisRequest(BaseModel):
    text: str = Field(
        ...,
        min_length=1,
        max_length=MAX_MESSAGE_LENGTH,
        description="Customer support ticket, message, or review to analyze.",
        examples=["My package arrived two weeks late and was completely broken! I demand an immediate refund."],
    )
    provider: Provider = Field(
        default="groq",
        description="Which LLM provider to use for analysis ('groq', 'openai', or 'anthropic').",
        examples=["groq"],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "text": "My package arrived two weeks late and was completely broken! I demand an immediate refund.",
                "provider": "groq",
            }
        }
    )


class SentimentAnalysisResponse(BaseModel):
    category: str = Field(
        ...,
        description="Assigned ticket category (e.g., Billing & Payments, Shipping & Delivery, Technical Support)",
        examples=["Shipping & Delivery"],
    )
    sentiment: str = Field(
        ...,
        description="Detected customer emotion (e.g., Frustrated, Angry, Neutral, Positive)",
        examples=["Frustrated"],
    )
    urgency: str = Field(
        ...,
        description="Assessed issue urgency (Low, Medium, High, Critical)",
        examples=["High"],
    )
    churn_risk: bool = Field(
        ...,
        description="Whether the customer expresses an intent or risk to cancel or churn",
        examples=[True],
    )
    key_entities: list[str] = Field(
        default_factory=list,
        description="Extracted entities such as order IDs, dates, product names, or financial terms",
        examples=["package", "two weeks late", "refund"],
    )
    summary: str = Field(
        ...,
        description="Concise one-sentence summary of the customer's issue",
        examples=["Customer received a broken package after a two-week delay and requests an immediate refund."],
    )
    recommended_action: str = Field(
        ...,
        description="Actionable next step for support agents",
        examples=["Acknowledge transit delay, process an expedited replacement or refund, and issue a \$15 service voucher."],
    )
    provider: Provider = Field(..., examples=["groq"])
    model: str = Field(..., examples=["openai/gpt-oss-120b"])
    latency_ms: float = Field(..., ge=0, examples=[512.4])
    estimated_cost_usd: float = Field(..., ge=0, examples=[0.000185])

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "Shipping & Delivery",
                "sentiment": "Frustrated",
                "urgency": "High",
                "churn_risk": True,
                "key_entities": ["package", "two weeks late", "refund"],
                "summary": "Customer received a broken package after a two-week delay and requests an immediate refund.",
                "recommended_action": "Acknowledge transit delay, process an expedited replacement or refund, and issue a $15 service voucher.",
                "provider": "groq",
                "model": "openai/gpt-oss-120b",
                "latency_ms": 512.4,
                "estimated_cost_usd": 0.000185,
            }
        }
    )


# ---------------------------------------------------------------------------
# Prompt Inspection Schema
# ---------------------------------------------------------------------------


class PromptTemplateDetailResponse(BaseModel):
    category: str = Field(..., description="Prompt category namespace", examples=["customer_support"])
    version: str = Field(..., description="Template version identifier", examples=["v2"])
    file_name: str = Field(..., description="File name on disk", examples=["v2.txt"])
    content: str = Field(
        ...,
        description="The raw written prompt template text",
        examples=[
            "You are a Senior Tier-3 Customer Support Operations Analyst at SwiftShip Logistics...\nOutput only the JSON object."
        ],
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "category": "customer_support",
                "version": "v2",
                "file_name": "v2.txt",
                "content": "You are a Senior Tier-3 Customer Support Operations Analyst at SwiftShip Logistics.\nAnalyze the customer's support inquiry and output a strictly valid JSON object...",
            }
        }
    )
