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
        default="hybrid",
        description="Retrieval strategy: 'hybrid' (RRF, recommended), 'vector', or 'keyword'",
        examples=["hybrid"],
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
        default=None,
        description="LLM provider for answer synthesis (defaults to GROQ or OPENAI)",
    )
    model: Optional[str] = Field(
        default=None,
        description="Model name override",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata filters (e.g. {'category': 'account'})",
    )


class RAGResponse(BaseModel):
    """Response payload containing generated answer, cited sources, and grounding state."""

    question: str = Field(..., description="The original question answered")
    answer: str = Field(..., description="Synthesized answer strictly grounded in the context")
    sources: List[RAGSourceItem] = Field(default_factory=list, description="Ordered source chunks cited in the answer")
    retrieval_strategy: str = Field(..., description="Retrieval method utilized")
    reranked: bool = Field(..., description="Whether reranking was executed")
    is_grounded: bool = Field(..., description="True if answer was synthesized; False if model returned 'I don't know'")
    citations: List[str] = Field(default_factory=list, description="Explicit source tags found in the generated answer text")
    total_sources_evaluated: int = Field(..., description="Total candidate sources evaluated in the context")
