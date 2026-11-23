import uuid
from typing import Any, Dict, List, Optional

from pydantic import AliasChoices, BaseModel, ConfigDict, Field


class BaseSearchRequest(BaseModel):
    """
    Common base schema for search requests.
    """
    query: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="Search query string",
        examples=["I can't access my account"],
    )
    top_k: int = Field(
        default=5,
        ge=1,
        le=50,
        description="Maximum number of documents to return",
        examples=[5],
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional metadata key-value filters to restrict candidate documents (e.g. {'category': 'account'})",
        examples=[{"category": "account"}],
    )


class SemanticSearchRequest(BaseSearchRequest):
    """
    Payload for semantic vector similarity search over stored vector documents.
    """
    pass


class KeywordSearchRequest(BaseSearchRequest):
    """
    Payload for PostgreSQL full-text keyword search using tsvector and websearch_to_tsquery.
    """
    pass


class SearchResultItem(BaseModel):
    """
    Individual document or chunk search result for semantic vector search.
    """
    id: uuid.UUID = Field(..., description="Unique document or chunk ID")
    document_id: Optional[uuid.UUID] = Field(default=None, description="Parent document UUID for source citation (if chunk)")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index within the parent document (if chunk)")
    content: str = Field(..., description="Original document or chunk text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("doc_metadata", "chunk_metadata", "metadata"),
        serialization_alias="metadata",
        description="Metadata attributes (category, source, tags, chunk offsets, etc.)",
    )
    similarity: float = Field(..., description="Cosine similarity score (0.0 to 1.0, higher is better)")
    distance: float = Field(..., description="Cosine distance metric (0.0 to 2.0, lower is closer)")
    embedding_model: str = Field(..., description="Model used to produce the vector embedding")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class KeywordSearchResultItem(BaseModel):
    """
    Individual document or chunk search result for full-text keyword search.
    """
    id: uuid.UUID = Field(..., description="Unique document or chunk ID")
    document_id: Optional[uuid.UUID] = Field(default=None, description="Parent document UUID for source citation (if chunk)")
    chunk_index: Optional[int] = Field(default=None, description="Chunk index within the parent document (if chunk)")
    content: str = Field(..., description="Original document or chunk text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("doc_metadata", "chunk_metadata", "metadata"),
        serialization_alias="metadata",
        description="Metadata attributes (category, source, tags, chunk offsets, etc.)",
    )
    rank: float = Field(..., description="PostgreSQL full-text relevance rank (ts_rank)")
    headline: Optional[str] = Field(default=None, description="Highlighted text fragment or matched terms")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class SemanticSearchResponse(BaseModel):
    """
    Response schema returning ranked nearest neighbor documents from vector search.
    """
    query: str = Field(..., description="Original query searched")
    top_k: int = Field(..., description="Requested top-K limit")
    total_results: int = Field(..., description="Number of matching results returned")
    search_type: str = Field(default="semantic", description="Type of search executed")
    results: List[SearchResultItem] = Field(default_factory=list, description="Ranked list of matching documents")


class KeywordSearchResponse(BaseModel):
    """
    Response schema returning full-text matched documents ranked by ts_rank.
    """
    query: str = Field(..., description="Original query searched")
    top_k: int = Field(..., description="Requested top-K limit")
    total_results: int = Field(..., description="Number of matching results returned")
    search_type: str = Field(default="keyword", description="Type of search executed")
    results: List[KeywordSearchResultItem] = Field(default_factory=list, description="Ranked list of matching documents")


class HybridSearchRequest(BaseSearchRequest):
    """
    Payload for hybrid search combining vector semantic retrieval and keyword FTS.
    """
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in linear fusion; (1 - alpha) is used for keyword search.",
        examples=[0.5],
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Smoothing constant for Reciprocal Rank Fusion (standard default 60).",
        examples=[60],
    )
    fusion_method: str = Field(
        default="rrf",
        description="Rank fusion strategy: 'rrf' (Reciprocal Rank Fusion) or 'weighted' (score normalization).",
        examples=["rrf"],
    )
    rerank: bool = Field(
        default=False,
        description="Whether to apply cross-encoder reranking to reorder the fused candidates.",
        examples=[False],
    )


class ChunkedSearchRequest(BaseSearchRequest):
    """
    Payload for executing search strictly over document chunks.
    Supports strategy: 'hybrid' (vector + keyword RRF), 'vector' (semantic only), or 'keyword' (FTS only).
    """
    strategy: str = Field(
        default="hybrid",
        description="Retrieval strategy: 'hybrid' (vector + keyword RRF), 'vector' (semantic only), or 'keyword' (FTS only).",
        examples=["hybrid"],
    )
    alpha: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Weight for vector search in linear fusion; (1 - alpha) is used for keyword search.",
        examples=[0.5],
    )
    rrf_k: int = Field(
        default=60,
        ge=1,
        le=1000,
        description="Smoothing constant for Reciprocal Rank Fusion (standard default 60).",
        examples=[60],
    )
    rerank: bool = Field(
        default=False,
        description="Whether to apply cross-encoder reranking to reorder the candidate chunks.",
        examples=[False],
    )


class ChunkSearchResultItem(BaseModel):
    """
    Search result item representing a retrieved document chunk.
    """
    id: uuid.UUID = Field(..., description="Unique chunk ID")
    document_id: uuid.UUID = Field(..., description="Parent document UUID for source citation")
    chunk_index: int = Field(..., description="Chunk index within the parent document")
    content: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("chunk_metadata", "metadata"),
        serialization_alias="metadata",
        description="Chunk metadata (including parent attributes and offset markers)",
    )
    similarity: Optional[float] = Field(default=None, description="Cosine similarity score (0.0 to 1.0)")
    distance: Optional[float] = Field(default=None, description="Cosine distance metric")
    rank: Optional[float] = Field(default=None, description="Full-text keyword relevance rank (ts_rank)")
    embedding_model: Optional[str] = Field(default=None, description="Embedding model identifier")
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder relevance score after reranking")
    rerank_rank: Optional[int] = Field(default=None, description="Ordinal rank assigned after reranking")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class HybridChunkSearchResultItem(BaseModel):
    """
    Search result item for hybrid retrieval combining vector similarity and keyword search ranks.
    """
    id: uuid.UUID = Field(..., description="Unique chunk ID")
    document_id: uuid.UUID = Field(..., description="Parent document UUID for source citation")
    chunk_index: int = Field(..., description="Chunk index within the parent document")
    content: str = Field(..., description="Chunk text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("chunk_metadata", "metadata"),
        serialization_alias="metadata",
        description="Merged chunk and parent metadata",
    )
    combined_score: float = Field(..., description="Combined fusion score (RRF score or weighted score)")
    vector_rank: Optional[int] = Field(default=None, description="1-based rank in vector search (if matched)")
    keyword_rank: Optional[int] = Field(default=None, description="1-based rank in keyword search (if matched)")
    vector_similarity: Optional[float] = Field(default=None, description="Raw cosine similarity score")
    keyword_score: Optional[float] = Field(default=None, description="Raw full-text ts_rank score")
    matched_by: str = Field(
        default="both",
        description="Which retrieval branch matched this chunk ('both', 'vector_only', 'keyword_only')",
    )
    rerank_score: Optional[float] = Field(default=None, description="Cross-encoder relevance score after reranking")
    rerank_rank: Optional[int] = Field(default=None, description="Ordinal rank assigned after reranking")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class ChunkSearchResponse(BaseModel):
    """
    Response schema for chunk-level search (vector or keyword).
    """
    query: str = Field(..., description="Original search query")
    top_k: int = Field(..., description="Requested top-K limit")
    total_results: int = Field(..., description="Number of results found")
    search_type: str = Field(..., description="Search mode executed ('chunk_vector' or 'chunk_keyword')")
    results: List[ChunkSearchResultItem] = Field(default_factory=list, description="Ranked matching chunks")


class HybridSearchResponse(BaseModel):
    """
    Response schema for hybrid chunk search combining vector and keyword retrieval.
    """
    query: str = Field(..., description="Original search query")
    top_k: int = Field(..., description="Requested top-K limit")
    total_results: int = Field(..., description="Number of unique results returned")
    search_type: str = Field(default="hybrid", description="Type of search executed")
    fusion_method: str = Field(default="rrf", description="Fusion algorithm used ('rrf' or 'weighted')")
    alpha: float = Field(default=0.5, description="Vector balance weight used")
    results: List[HybridChunkSearchResultItem] = Field(default_factory=list, description="Ranked fused results")

