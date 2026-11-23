import math
import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.infrastructure.embedding_providers import BaseEmbeddingProvider, get_embedding_provider
from app.infrastructure.ranking.rrf import compute_rrf_score
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
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


def _compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Fallback cosine similarity computation for non-PostgreSQL (SQLite) test sessions."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


class SearchService:
    """
    Coordinates semantic vector search and PostgreSQL full-text keyword search:
    - Semantic Search: Converts natural language queries into 768-dim embeddings via Gemini
      and performs cosine-distance retrieval via pgvector `<=>`.
    - Keyword Search: Leverages PostgreSQL full-text search (to_tsvector, websearch_to_tsquery, ts_rank).
    - Metadata Filtering: Evaluates JSONB metadata constraints across both search paradigms.
    """

    def __init__(
        self,
        db: Session,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        embedding_service: Optional[BaseEmbeddingProvider] = None,
    ) -> None:
        self.db = db
        self.embedding_provider = embedding_provider or embedding_service or get_embedding_provider()

    @property
    def dialect_name(self) -> str:
        return self.db.bind.dialect.name if self.db.bind else "postgresql"

    def _apply_metadata_filters(self, stmt: Any, filters: Optional[Dict[str, Any]]) -> Any:
        """Helper to apply JSON metadata equality filters (DRY)."""
        if filters:
            for key, val in filters.items():
                if not key or key.startswith("additionalProp") or (isinstance(val, (dict, list)) and not val):
                    continue
                stmt = stmt.where(Document.doc_metadata[key].as_string() == str(val))
        return stmt

    def _apply_chunk_metadata_filters(self, stmt: Any, filters: Optional[Dict[str, Any]]) -> Any:
        """Helper to apply JSON metadata equality filters on chunk metadata."""
        if filters:
            for key, val in filters.items():
                if not key or key.startswith("additionalProp") or (isinstance(val, (dict, list)) and not val):
                    continue
                stmt = stmt.where(DocumentChunk.chunk_metadata[key].as_string() == str(val))
        return stmt

    def has_chunks(self) -> bool:
        """Check whether document chunks are populated in the database."""
        try:
            count = self.db.scalar(select(func.count(DocumentChunk.id))) or 0
            return count > 0
        except Exception:
            return False

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SemanticSearchResponse:
        """
        Execute semantic cosine similarity search strictly over stored UNCHUNKED documents.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")

        # 1. Generate query embedding using Gemini
        query_vector = self.embedding_provider.embed_text(cleaned_query)

        result_items: List[SearchResultItem] = []

        if self.dialect_name == "postgresql":
            # Native PostgreSQL pgvector cosine distance operator (<=>)
            distance_col = Document.embedding.cosine_distance(query_vector).label("distance")
            stmt = select(Document, distance_col)
            stmt = self._apply_metadata_filters(stmt, filters)
            stmt = stmt.order_by(distance_col.asc()).limit(top_k)

            rows = self.db.execute(stmt).all()
            for doc, raw_dist in rows:
                dist_val = float(raw_dist)
                sim_val = max(0.0, min(1.0, 1.0 - dist_val))
                result_items.append(
                    SearchResultItem(
                        id=doc.id,
                        content=doc.content,
                        metadata=doc.doc_metadata,
                        similarity=round(sim_val, 4),
                        distance=round(dist_val, 4),
                        embedding_model=doc.embedding_model,
                    )
                )
        else:
            # SQLite fallback: fetch candidate records and rank via Python cosine similarity
            stmt = select(Document)
            stmt = self._apply_metadata_filters(stmt, filters)
            candidates = list(self.db.scalars(stmt).all())

            scored: List[Tuple[Document, float, float]] = []
            for doc in candidates:
                sim = _compute_cosine_similarity(doc.embedding, query_vector)
                dist = 1.0 - sim
                scored.append((doc, dist, sim))

            scored.sort(key=lambda x: x[1])
            for doc, dist_val, sim_val in scored[:top_k]:
                result_items.append(
                    SearchResultItem(
                        id=doc.id,
                        content=doc.content,
                        metadata=doc.doc_metadata,
                        similarity=round(sim_val, 4),
                        distance=round(dist_val, 4),
                        embedding_model=doc.embedding_model,
                    )
                )

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
        """
        Execute full-text keyword search strictly over stored UNCHUNKED documents.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")

        result_items: List[KeywordSearchResultItem] = []

        if self.dialect_name == "postgresql":
            # PostgreSQL full-text search: to_tsvector, websearch_to_tsquery, ts_rank
            ts_vec = func.to_tsvector("english", Document.content)
            ts_query = func.websearch_to_tsquery("english", cleaned_query)
            rank_col = func.ts_rank(ts_vec, ts_query).label("rank")

            stmt = select(Document, rank_col).where(ts_vec.bool_op("@@")(ts_query))
            stmt = self._apply_metadata_filters(stmt, filters)
            stmt = stmt.order_by(rank_col.desc()).limit(top_k)

            rows = self.db.execute(stmt).all()
            for doc, raw_rank in rows:
                rank_val = float(raw_rank)
                result_items.append(
                    KeywordSearchResultItem(
                        id=doc.id,
                        content=doc.content,
                        metadata=doc.doc_metadata,
                        rank=round(rank_val, 4),
                    )
                )
        else:
            # SQLite fallback: keyword matching using tokenized words
            stmt = select(Document)
            stmt = self._apply_metadata_filters(stmt, filters)
            candidates = list(self.db.scalars(stmt).all())

            # Extract distinct alphanumeric keywords
            keywords = [w.lower() for w in re.findall(r"\w+", cleaned_query) if len(w) > 1]
            scored_candidates: List[Tuple[Document, float]] = []

            for doc in candidates:
                content_lower = doc.content.lower()
                matches = sum(1 for kw in keywords if kw in content_lower)
                if matches > 0:
                    # Normalized score based on matched tokens
                    score = matches / max(1, len(keywords))
                    scored_candidates.append((doc, score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            for doc, score in scored_candidates[:top_k]:
                result_items.append(
                    KeywordSearchResultItem(
                        id=doc.id,
                        content=doc.content,
                        metadata=doc.doc_metadata,
                        rank=round(score, 4),
                    )
                )

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

        vector_map: Dict[Any, Tuple[int, Any]] = {
            item.id: (rank, item) for rank, item in enumerate(vec_res.results, start=1)
        }
        keyword_map: Dict[Any, Tuple[int, Any]] = {
            item.id: (rank, item) for rank, item in enumerate(kw_res.results, start=1)
        }

        all_doc_ids = list(dict.fromkeys(list(vector_map.keys()) + list(keyword_map.keys())))
        fused_items: List[DocumentHybridSearchResultItem] = []

        for did in all_doc_ids:
            in_vec = did in vector_map
            in_kw = did in keyword_map
            v_rank, v_item = vector_map[did] if in_vec else (None, None)
            k_rank, k_item = keyword_map[did] if in_kw else (None, None)

            rrf_score = 0.0
            if in_vec:
                rrf_score += compute_rrf_score(v_rank, weight=alpha, k=rrf_k)
            if in_kw:
                rrf_score += compute_rrf_score(k_rank, weight=(1.0 - alpha), k=rrf_k)

            base = v_item or k_item
            matched = "both" if (in_vec and in_kw) else ("vector_only" if in_vec else "keyword_only")

            fused_items.append(
                DocumentHybridSearchResultItem(
                    id=base.id,
                    content=base.content,
                    metadata=base.metadata,
                    combined_score=round(rrf_score, 6),
                    vector_rank=v_rank,
                    keyword_rank=k_rank,
                    vector_similarity=v_item.similarity if in_vec else None,
                    keyword_score=k_item.rank if in_kw else None,
                    matched_by=matched,
                    embedding_model=getattr(v_item, "embedding_model", None),
                )
            )

        fused_items.sort(key=lambda x: x.combined_score, reverse=True)

        return DocumentHybridSearchResponse(
            query=cleaned_query,
            top_k=top_k,
            total_results=len(all_doc_ids),
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
        Returns the top_k most semantically relevant chunks with citation back to parent document.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")

        query_vector = self.embedding_provider.embed_text(cleaned_query)
        result_items: List[ChunkSearchResultItem] = []

        if self.dialect_name == "postgresql":
            distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
            stmt = select(DocumentChunk, distance_col)
            stmt = self._apply_chunk_metadata_filters(stmt, filters)
            stmt = stmt.order_by(distance_col.asc()).limit(top_k)

            rows = self.db.execute(stmt).all()
            for chunk, raw_dist in rows:
                dist_val = float(raw_dist)
                sim_val = max(0.0, min(1.0, 1.0 - dist_val))
                result_items.append(
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
                )
        else:
            stmt = select(DocumentChunk)
            stmt = self._apply_chunk_metadata_filters(stmt, filters)
            candidates = list(self.db.scalars(stmt).all())

            scored: List[Tuple[DocumentChunk, float, float]] = []
            for chunk in candidates:
                sim = _compute_cosine_similarity(chunk.embedding, query_vector)
                dist = 1.0 - sim
                scored.append((chunk, dist, sim))

            scored.sort(key=lambda x: x[1])
            for chunk, dist_val, sim_val in scored[:top_k]:
                result_items.append(
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
                )

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
        """
        Execute full-text keyword search over DocumentChunk records using PostgreSQL FTS.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")

        result_items: List[ChunkSearchResultItem] = []

        if self.dialect_name == "postgresql":
            ts_vec = func.to_tsvector("english", DocumentChunk.content)
            ts_query = func.websearch_to_tsquery("english", cleaned_query)
            rank_col = func.ts_rank(ts_vec, ts_query).label("rank")

            stmt = select(DocumentChunk, rank_col).where(ts_vec.bool_op("@@")(ts_query))
            stmt = self._apply_chunk_metadata_filters(stmt, filters)
            stmt = stmt.order_by(rank_col.desc()).limit(top_k)

            rows = self.db.execute(stmt).all()
            for chunk, raw_rank in rows:
                rank_val = float(raw_rank)
                result_items.append(
                    ChunkSearchResultItem(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        metadata=chunk.chunk_metadata,
                        rank=round(rank_val, 4),
                        embedding_model=chunk.embedding_model,
                    )
                )
        else:
            stmt = select(DocumentChunk)
            stmt = self._apply_chunk_metadata_filters(stmt, filters)
            candidates = list(self.db.scalars(stmt).all())

            keywords = [w.lower() for w in re.findall(r"\w+", cleaned_query) if len(w) > 1]
            scored_candidates: List[Tuple[DocumentChunk, float]] = []

            for chunk in candidates:
                content_lower = chunk.content.lower()
                matches = sum(1 for kw in keywords if kw in content_lower)
                if matches > 0:
                    score = matches / max(1, len(keywords))
                    scored_candidates.append((chunk, score))

            scored_candidates.sort(key=lambda x: x[1], reverse=True)
            for chunk, score in scored_candidates[:top_k]:
                result_items.append(
                    ChunkSearchResultItem(
                        id=chunk.id,
                        document_id=chunk.document_id,
                        chunk_index=chunk.chunk_index,
                        content=chunk.content,
                        metadata=chunk.chunk_metadata,
                        rank=round(score, 4),
                        embedding_model=chunk.embedding_model,
                    )
                )

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

        # 1. Fetch vector candidates
        vector_resp = self.chunk_vector_search(query=cleaned_query, top_k=candidate_k, filters=filters)
        vector_map = {item.id: (rank + 1, item) for rank, item in enumerate(vector_resp.results)}

        # 2. Fetch keyword candidates
        keyword_resp = self.chunk_keyword_search(query=cleaned_query, top_k=candidate_k, filters=filters)
        keyword_map = {item.id: (rank + 1, item) for rank, item in enumerate(keyword_resp.results)}

        all_chunk_ids = set(vector_map.keys()) | set(keyword_map.keys())
        if not all_chunk_ids:
            return HybridSearchResponse(
                query=cleaned_query,
                top_k=top_k,
                total_results=0,
                search_type="hybrid",
                fusion_method=fusion_method,
                alpha=alpha,
                results=[],
            )

        fused_items: List[HybridChunkSearchResultItem] = []

        if fusion_method == "weighted":
            max_kw = max((item[1].rank or 0.0 for item in keyword_map.values()), default=1.0)
            if max_kw <= 0.0:
                max_kw = 1.0

            for cid in all_chunk_ids:
                in_vec = cid in vector_map
                in_kw = cid in keyword_map

                v_rank, v_item = vector_map[cid] if in_vec else (None, None)
                k_rank, k_item = keyword_map[cid] if in_kw else (None, None)

                v_sim = v_item.similarity if v_item and v_item.similarity is not None else 0.0
                k_score = (k_item.rank or 0.0) if k_item else 0.0
                normalized_k = k_score / max_kw

                combined = (alpha * v_sim) + ((1.0 - alpha) * normalized_k)
                base = v_item or k_item

                matched = "both" if (in_vec and in_kw) else ("vector_only" if in_vec else "keyword_only")
                fused_items.append(
                    HybridChunkSearchResultItem(
                        id=base.id,
                        document_id=base.document_id,
                        chunk_index=base.chunk_index,
                        content=base.content,
                        metadata=base.metadata,
                        combined_score=round(combined, 6),
                        vector_rank=v_rank,
                        keyword_rank=k_rank,
                        vector_similarity=v_sim if in_vec else None,
                        keyword_score=k_score if in_kw else None,
                        matched_by=matched,
                    )
                )
        else:
            # Reciprocal Rank Fusion (RRF)
            for cid in all_chunk_ids:
                in_vec = cid in vector_map
                in_kw = cid in keyword_map

                v_rank, v_item = vector_map[cid] if in_vec else (None, None)
                k_rank, k_item = keyword_map[cid] if in_kw else (None, None)

                rrf_score = 0.0
                if in_vec:
                    rrf_score += compute_rrf_score(v_rank, weight=alpha, k=rrf_k)
                if in_kw:
                    rrf_score += compute_rrf_score(k_rank, weight=(1.0 - alpha), k=rrf_k)

                base = v_item or k_item
                matched = "both" if (in_vec and in_kw) else ("vector_only" if in_vec else "keyword_only")

                fused_items.append(
                    HybridChunkSearchResultItem(
                        id=base.id,
                        document_id=base.document_id,
                        chunk_index=base.chunk_index,
                        content=base.content,
                        metadata=base.metadata,
                        combined_score=round(rrf_score, 6),
                        vector_rank=v_rank,
                        keyword_rank=k_rank,
                        vector_similarity=v_item.similarity if in_vec else None,
                        keyword_score=k_item.rank if in_kw else None,
                        matched_by=matched,
                    )
                )

        # Sort descending by combined score
        fused_items.sort(key=lambda x: x.combined_score, reverse=True)

        return HybridSearchResponse(
            query=cleaned_query,
            top_k=top_k,
            total_results=len(all_chunk_ids),
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

