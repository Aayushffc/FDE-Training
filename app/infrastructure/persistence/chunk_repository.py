import math
import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.document_chunk import DocumentChunk


def _compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Fallback cosine similarity computation for non-PostgreSQL (e.g. SQLite test) databases."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


class ChunkRepository:
    """
    Persistence adapter for DocumentChunk records.
    Encapsulates raw SQL, pgvector operators (<=>), and PostgreSQL full-text search for chunks.
    """

    def __init__(self, db: Session) -> None:
        self.db = db

    @property
    def dialect_name(self) -> str:
        return self.db.bind.dialect.name if self.db.bind else "postgresql"

    def apply_filters(self, stmt: Any, filters: Optional[Dict[str, Any]]) -> Any:
        if filters:
            for key, val in filters.items():
                if not key or key.startswith("additionalProp") or (isinstance(val, (dict, list)) and not val):
                    continue
                stmt = stmt.where(DocumentChunk.chunk_metadata[key].as_string() == str(val))
        return stmt

    def count(self) -> int:
        return self.db.scalar(select(func.count(DocumentChunk.id))) or 0

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float, float]]:
        """
        Executes dense vector search across stored DocumentChunk records.
        Returns list of (DocumentChunk, similarity, distance).
        """
        if self.dialect_name == "postgresql":
            distance_col = DocumentChunk.embedding.cosine_distance(query_vector).label("distance")
            stmt = select(DocumentChunk, distance_col)
            stmt = self.apply_filters(stmt, filters)
            stmt = stmt.order_by(distance_col.asc()).limit(top_k)

            rows = self.db.execute(stmt).all()
            results: List[Tuple[DocumentChunk, float, float]] = []
            for chunk, raw_dist in rows:
                dist_val = float(raw_dist)
                sim_val = max(0.0, min(1.0, 1.0 - dist_val))
                results.append((chunk, sim_val, dist_val))
            return results
        else:
            stmt = select(DocumentChunk)
            stmt = self.apply_filters(stmt, filters)
            candidates = list(self.db.scalars(stmt).all())

            scored: List[Tuple[DocumentChunk, float, float]] = []
            for chunk in candidates:
                sim = _compute_cosine_similarity(chunk.embedding, query_vector)
                dist = 1.0 - sim
                scored.append((chunk, sim, dist))

            scored.sort(key=lambda x: x[2])  # sort ascending by distance
            return scored[:top_k]

    def search_keywords(
        self,
        cleaned_query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[DocumentChunk, float]]:
        """
        Executes sparse keyword search across stored DocumentChunk records.
        Returns list of (DocumentChunk, rank_score).
        """
        if self.dialect_name == "postgresql":
            ts_vec = func.to_tsvector("english", DocumentChunk.content)
            ts_query = func.websearch_to_tsquery("english", cleaned_query)
            rank_col = func.ts_rank(ts_vec, ts_query).label("rank")

            stmt = select(DocumentChunk, rank_col).where(ts_vec.bool_op("@@")(ts_query))
            stmt = self.apply_filters(stmt, filters)
            stmt = stmt.order_by(rank_col.desc()).limit(top_k)

            rows = self.db.execute(stmt).all()
            return [(chunk, float(raw_rank)) for chunk, raw_rank in rows]
        else:
            stmt = select(DocumentChunk)
            stmt = self.apply_filters(stmt, filters)
            candidates = list(self.db.scalars(stmt).all())

            keywords = [w.lower() for w in re.findall(r"\w+", cleaned_query) if len(w) > 1]
            scored: List[Tuple[DocumentChunk, float]] = []

            for chunk in candidates:
                content_lower = chunk.content.lower()
                matches = sum(1 for kw in keywords if kw in content_lower)
                if matches > 0:
                    score = matches / max(1, len(keywords))
                    scored.append((chunk, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
