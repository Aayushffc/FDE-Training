from typing import Any, Dict, List, Optional

from sqlalchemy.orm import Session

from app.infrastructure.embedding_providers import BaseEmbeddingProvider, get_embedding_provider
from app.infrastructure.persistence import ChunkRepository, DocumentRepository
from app.infrastructure.ranking import fuse_rankings, fuse_weighted_scores
from app.schemas.search import (
    ChunkSearchResponse,
    ChunkSearchResultItem,
    DocumentHybridSearchResponse,
    DocumentHybridSearchResultItem,
    HybridChunkSearchResultItem,
    HybridSearchResponse,
    KeywordSearchResultItem,
    KeywordSearchResponse,
    SearchResultItem,
    SemanticSearchResponse,
)


class SearchService:
    """
    Application Service orchestrating vector search, PostgreSQL full-text search,
    and hybrid rank fusion across documents and chunks.
    Delegates database access to DocumentRepository and ChunkRepository.
    Delegates fusion algorithms to ranking infrastructure.
    """

    def __init__(
        self,
        db: Session,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        embedding_service: Optional[BaseEmbeddingProvider] = None,
        document_repo: Optional[DocumentRepository] = None,
        chunk_repo: Optional[ChunkRepository] = None,
    ) -> None:
        self.db = db
        self.embedding_provider = embedding_provider or embedding_service or get_embedding_provider()
        self.document_repo = document_repo or DocumentRepository(db=db)
        self.chunk_repo = chunk_repo or ChunkRepository(db=db)

    @property
    def dialect_name(self) -> str:
        return self.document_repo.dialect_name

    def has_chunks(self) -> bool:
        """Check whether document chunks are populated in the database."""
        try:
            return self.chunk_repo.count() > 0
        except Exception:
            return False

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SemanticSearchResponse:
        """Execute semantic cosine similarity search strictly over stored UNCHUNKED documents."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")

        query_vector = self.embedding_provider.embed_text(cleaned_query)
        rows = self.document_repo.search_vectors(query_vector=query_vector, top_k=top_k, filters=filters)

        result_items = [
            SearchResultItem(
                id=doc.id,
                content=doc.content,
                metadata=doc.doc_metadata,
                similarity=round(sim_val, 4),
                distance=round(dist_val, 4),
                embedding_model=doc.embedding_model,
            )
            for doc, sim_val, dist_val in rows
        ]

        return SemanticSearchResponse(
            query=cleaned_query,
            top_k=top_k,
            total_results=len(result_items),
            results=result_items,
        )

    def keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> KeywordSearchResponse:
        """Execute full-text keyword search strictly over stored UNCHUNKED documents."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")

        rows = self.document_repo.search_keywords(cleaned_query=cleaned_query, top_k=top_k, filters=filters)
        result_items = [
            KeywordSearchResultItem(
                id=doc.id,
                content=doc.content,
                metadata=doc.doc_metadata,
                rank=round(rank_val, 4),
            )
            for doc, rank_val in rows
        ]

        return KeywordSearchResponse(
            query=cleaned_query,
            top_k=top_k,
            total_results=len(result_items),
            results=result_items,
        )

    def document_hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5,
        rrf_k: int = 60,
    ) -> DocumentHybridSearchResponse:
        """
        Execute hybrid search over UNCHUNKED parent documents combining vector similarity and keyword search ranks.
        Uses Reciprocal Rank Fusion (RRF): RRF_Score = sum(w / (k + rank)).
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")

        fetch_k = max(top_k * 3, 20)
        vec_res = self.semantic_search(query=cleaned_query, top_k=fetch_k, filters=filters)
        kw_res = self.keyword_search(query=cleaned_query, top_k=fetch_k, filters=filters)

        fused = fuse_rankings(
            vector_results=vec_res.results,
            keyword_results=kw_res.results,
            id_getter=lambda x: x.id,
            alpha=alpha,
            rrf_k=rrf_k,
        )

        fused_items: List[DocumentHybridSearchResultItem] = []
        for f in fused:
            item = f["item"]
            v_rank = f["vector_rank"]
            k_rank = f["keyword_rank"]
            fused_items.append(
                DocumentHybridSearchResultItem(
                    id=item.id,
                    content=item.content,
                    metadata=item.metadata,
                    combined_score=f["combined_score"],
                    vector_rank=v_rank,
                    keyword_rank=k_rank,
                    vector_similarity=getattr(item, "similarity", None) if v_rank is not None else None,
                    keyword_score=getattr(item, "rank", None) if k_rank is not None else None,
                    matched_by=f["matched_by"],
                    embedding_model=getattr(item, "embedding_model", None),
                )
            )

        return DocumentHybridSearchResponse(
            query=cleaned_query,
            top_k=top_k,
            total_results=len(fused),
            search_type="document_hybrid",
            fusion_method="rrf",
            alpha=alpha,
            results=fused_items[:top_k],
        )

    def chunk_vector_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> ChunkSearchResponse:
        """
        Execute semantic cosine similarity search over DocumentChunk records.
        Returns top_k most relevant chunks with citation back to parent document.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")

        query_vector = self.embedding_provider.embed_text(cleaned_query)
        rows = self.chunk_repo.search_vectors(query_vector=query_vector, top_k=top_k, filters=filters)

        result_items = [
            ChunkSearchResultItem(
                id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                metadata=chunk.chunk_metadata,
                similarity=round(sim_val, 4),
                distance=round(dist_val, 4),
                embedding_model=chunk.embedding_model,
            )
            for chunk, sim_val, dist_val in rows
        ]

        return ChunkSearchResponse(
            query=cleaned_query,
            top_k=top_k,
            total_results=len(result_items),
            search_type="chunk_vector",
            results=result_items,
        )

    def chunk_keyword_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> ChunkSearchResponse:
        """Execute full-text keyword search over DocumentChunk records using PostgreSQL FTS."""
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")

        rows = self.chunk_repo.search_keywords(cleaned_query=cleaned_query, top_k=top_k, filters=filters)
        result_items = [
            ChunkSearchResultItem(
                id=chunk.id,
                document_id=chunk.document_id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                metadata=chunk.chunk_metadata,
                rank=round(rank_val, 4),
                embedding_model=chunk.embedding_model,
            )
            for chunk, rank_val in rows
        ]

        return ChunkSearchResponse(
            query=cleaned_query,
            top_k=top_k,
            total_results=len(result_items),
            search_type="chunk_keyword",
            results=result_items,
        )

    def chunk_hybrid_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        alpha: float = 0.5,
        rrf_k: int = 60,
        fusion_method: str = "rrf",
    ) -> HybridSearchResponse:
        """
        Execute hybrid search combining vector similarity and keyword search ranks.
        Supports Reciprocal Rank Fusion (RRF, default) and weighted score normalization.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")

        candidate_k = max(top_k * 3, 20)
        vector_resp = self.chunk_vector_search(query=cleaned_query, top_k=candidate_k, filters=filters)
        keyword_resp = self.chunk_keyword_search(query=cleaned_query, top_k=candidate_k, filters=filters)

        if fusion_method == "weighted":
            fused = fuse_weighted_scores(
                vector_results=vector_resp.results,
                keyword_results=keyword_resp.results,
                id_getter=lambda x: x.id,
                alpha=alpha,
            )
        else:
            fused = fuse_rankings(
                vector_results=vector_resp.results,
                keyword_results=keyword_resp.results,
                id_getter=lambda x: x.id,
                alpha=alpha,
                rrf_k=rrf_k,
            )

        fused_items: List[HybridChunkSearchResultItem] = []
        for f in fused:
            item = f["item"]
            v_rank = f["vector_rank"]
            k_rank = f["keyword_rank"]
            v_sim = f.get("vector_similarity")
            if v_sim is None and v_rank is not None:
                v_sim = getattr(item, "similarity", None)
            k_score = f.get("keyword_score")
            if k_score is None and k_rank is not None:
                k_score = getattr(item, "rank", None)

            fused_items.append(
                HybridChunkSearchResultItem(
                    id=item.id,
                    document_id=item.document_id,
                    chunk_index=item.chunk_index,
                    content=item.content,
                    metadata=item.metadata,
                    combined_score=f["combined_score"],
                    vector_rank=v_rank,
                    keyword_rank=k_rank,
                    vector_similarity=v_sim,
                    keyword_score=k_score,
                    matched_by=f["matched_by"],
                )
            )

        return HybridSearchResponse(
            query=cleaned_query,
            top_k=top_k,
            total_results=len(fused),
            search_type="hybrid",
            fusion_method=fusion_method,
            alpha=alpha,
            results=fused_items[:top_k],
        )

    def chunk_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        strategy: str = "hybrid",
        alpha: float = 0.5,
        rrf_k: int = 60,
        rerank: bool = False,
    ) -> HybridSearchResponse:
        """
        Unified search method operating strictly on 'document_chunks'.
        Supports 'hybrid' (vector + keyword RRF), 'vector', or 'keyword' strategies,
        with optional cross-encoder reranking.
        """
        strat = strategy.lower().strip()
        if strat == "vector":
            v_resp = self.chunk_vector_search(query=query, top_k=top_k, filters=filters)
            items = [
                HybridChunkSearchResultItem(
                    id=c.id,
                    document_id=c.document_id,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    metadata=c.metadata,
                    combined_score=c.similarity or 0.0,
                    vector_similarity=c.similarity,
                    matched_by="vector_only",
                )
                for c in v_resp.results
            ]
            resp = HybridSearchResponse(
                query=query,
                top_k=top_k,
                total_results=len(items),
                search_type="chunk_vector",
                results=items,
            )
        elif strat == "keyword":
            k_resp = self.chunk_keyword_search(query=query, top_k=top_k, filters=filters)
            items = [
                HybridChunkSearchResultItem(
                    id=c.id,
                    document_id=c.document_id,
                    chunk_index=c.chunk_index,
                    content=c.content,
                    metadata=c.metadata,
                    combined_score=c.rank or 0.0,
                    keyword_score=c.rank,
                    matched_by="keyword_only",
                )
                for c in k_resp.results
            ]
            resp = HybridSearchResponse(
                query=query,
                top_k=top_k,
                total_results=len(items),
                search_type="chunk_keyword",
                results=items,
            )
        else:
            resp = self.chunk_hybrid_search(
                query=query,
                top_k=top_k,
                filters=filters,
                alpha=alpha,
                rrf_k=rrf_k,
            )

        if rerank and resp.results:
            from app.services.reranking_service import RerankingService
            reranker = RerankingService()
            reranked = reranker.rerank(query=query, items=resp.results, top_k=top_k)
            resp.results = reranked

        return resp
