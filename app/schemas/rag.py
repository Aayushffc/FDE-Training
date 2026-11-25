from typing import Any, Dict, List, Optional
from pydantic import BaseModel, ConfigDict, Field


class RAGSourceItem(BaseModel):
    """Source chunk cited in RAG answer generation."""

    source_label: str = Field(..., description="Label used in the prompt context (e.g. 'Source 1')")
    document_id: str = Field(..., description="UUID of the parent document")
    chunk_id: str = Field(..., description="UUID of the document chunk")
    chunk_index: Optional[int] = Field(default=None, description="Position of chunk within the parent document")
    content: str = Field(..., description="Excerpt content of the cited chunk")
    relevance_score: Optional[float] = Field(default=None, description="Normalized relevance or similarity score")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class RAGRequest(BaseModel):
    """Request payload for RAG answer generation."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="User question to be answered using retrieved context",
        examples=["How can I recover my account after too many failed attempts?"],
    )
    retrieval_strategy: str = Field(
        default="vector",
        description="Retrieval strategy: 'vector' (semantic, default), 'keyword', or 'hybrid'",
        examples=["vector"],
    )
    rerank: bool = Field(
        default=True,
        description="Whether to evaluate and reorder candidate chunks with the cross-encoder reranker",
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Number of candidate chunks to retrieve and consider",
    )
    provider: Optional[str] = Field(
        default="groq",
        description="LLM provider for answer synthesis: 'groq', 'openai', or 'anthropic'",
        examples=["groq"],
    )
    prompt_category: Optional[str] = Field(
        default="rag",
        description="Database prompt category to retrieve system prompt template from",
        examples=["rag"],
    )
    prompt_version: Optional[str] = Field(
        default="v1",
        description="Version identifier of prompt template to retrieve from database",
        examples=["v1"],
    )

    model_config = ConfigDict(extra="ignore")


class RAGResponse(BaseModel):
    """Response payload containing generated answer, cited sources, grounding state, and execution metrics."""

    question: str = Field(..., description="The original question answered")
    answer: str = Field(..., description="Synthesized answer strictly grounded in the context")
    sources: List[RAGSourceItem] = Field(default_factory=list, description="Ordered source chunks cited in the answer")
    retrieval_strategy: str = Field(..., description="Retrieval method utilized")
    reranked: bool = Field(..., description="Whether reranking was executed")
    is_grounded: bool = Field(..., description="True if answer was synthesized; False if model returned 'I don't know'")
    citations: List[str] = Field(default_factory=list, description="Explicit source tags found in the generated answer text")
    total_sources_evaluated: int = Field(..., description="Total candidate sources evaluated in the context")

    # Performance, Token Consumption, and Pricing Metrics
    latency_ms: float = Field(default=0.0, ge=0, description="Total pipeline latency (retrieval + rerank + synthesis) in milliseconds")
    llm_latency_ms: Optional[float] = Field(default=None, ge=0, description="Inference latency of the LLM generation step in milliseconds")
    input_tokens: int = Field(default=0, ge=0, description="Input / prompt token count consumed")
    output_tokens: int = Field(default=0, ge=0, description="Completion / answer token count generated")
    total_tokens: int = Field(default=0, ge=0, description="Total tokens consumed (input + output)")
    estimated_cost_usd: float = Field(default=0.0, ge=0, description="Estimated total LLM inference cost in USD")
    provider: Optional[str] = Field(default=None, description="LLM provider that synthesized the answer (e.g. 'groq', 'openai')")
    model: Optional[str] = Field(default=None, description="Model identifier used for inference")
