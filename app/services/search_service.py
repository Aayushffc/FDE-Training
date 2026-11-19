import math
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.infrastructure.embedding_providers import BaseEmbeddingProvider, get_embedding_provider
from app.models.document import Document
from app.schemas.search import SearchResultItem, SemanticSearchResponse


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
    Coordinates semantic vector search:
    1. Embeds user natural-language queries into 768-dim vectors via Gemini.
    2. Performs cosine distance similarity queries against Neon PostgreSQL (using pgvector `<=>`).
    3. Evaluates optional JSON metadata filters (e.g. {'category': 'account'}).
    4. Ranks and returns the top-K most semantically similar documents.
    """

    def __init__(
        self,
        db: Session,
        embedding_provider: Optional[BaseEmbeddingProvider] = None,
        embedding_service: Optional[BaseEmbeddingProvider] = None,
    ) -> None:
        self.db = db
        self.embedding_provider = embedding_provider or embedding_service or get_embedding_provider()

    def semantic_search(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> SemanticSearchResponse:
        """
        Execute semantic cosine similarity search with optional metadata filtering.
        """
        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Search query cannot be empty.")

        # 1. Generate query embedding using Gemini
        query_vector = self.embedding_provider.embed_text(cleaned_query)

        # 2. Determine dialect (PostgreSQL vs SQLite for tests)
        dialect_name = self.db.bind.dialect.name if self.db.bind else "postgresql"

        result_items: List[SearchResultItem] = []

        if dialect_name == "postgresql":
            # Native PostgreSQL pgvector cosine distance operator (<=>)
            distance_col = Document.embedding.cosine_distance(query_vector).label("distance")
            stmt = select(Document, distance_col)

            # Apply metadata filters
            if filters:
                for key, val in filters.items():
                    stmt = stmt.where(Document.doc_metadata[key].as_string() == str(val))

            # Order by distance ascending (nearest vectors first)
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
            if filters:
                for key, val in filters.items():
                    stmt = stmt.where(Document.doc_metadata[key].as_string() == str(val))

            candidates = list(self.db.scalars(stmt).all())
            scored: List[Tuple[Document, float, float]] = []

            for doc in candidates:
                sim = _compute_cosine_similarity(doc.embedding, query_vector)
                dist = 1.0 - sim
                scored.append((doc, dist, sim))

            # Sort by distance ascending (similarity descending)
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
