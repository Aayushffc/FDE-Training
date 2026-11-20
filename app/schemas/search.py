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
    Individual document search result for semantic vector search.
    """
    id: uuid.UUID = Field(..., description="Unique document ID")
    content: str = Field(..., description="Original document text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("doc_metadata", "metadata"),
        serialization_alias="metadata",
        description="Document metadata attributes (category, source, tags, etc.)",
    )
    similarity: float = Field(..., description="Cosine similarity score (0.0 to 1.0, higher is better)")
    distance: float = Field(..., description="Cosine distance metric (0.0 to 2.0, lower is closer)")
    embedding_model: str = Field(..., description="Model used to produce the document's vector embedding")

    model_config = ConfigDict(from_attributes=True, populate_by_name=True)


class KeywordSearchResultItem(BaseModel):
    """
    Individual document search result for full-text keyword search.
    """
    id: uuid.UUID = Field(..., description="Unique document ID")
    content: str = Field(..., description="Original document text content")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        validation_alias=AliasChoices("doc_metadata", "metadata"),
        serialization_alias="metadata",
        description="Document metadata attributes (category, source, tags, etc.)",
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
