import math
import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.document import Document


def _compute_cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """Fallback cosine similarity computation for non-PostgreSQL (e.g. SQLite test) databases."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a * a for a in vec_a))
    norm_b = math.sqrt(sum(b * b for b in vec_b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


class DocumentRepository:
    """
    Persistence adapter for unchunked Document records.
    Encapsulates raw SQL, pgvector operators (<=>), and PostgreSQL full-text search.
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
                stmt = stmt.where(Document.doc_metadata[key].as_string() == str(val))
        return stmt

    def count(self) -> int:
        return self.db.scalar(select(func.count(Document.id))) or 0

    def search_vectors(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float, float]]:
        """
        Executes dense vector search across stored Document records.
        Returns list of (Document, similarity, distance).
        """
        if self.dialect_name == "postgresql":
            distance_col = Document.embedding.cosine_distance(query_vector).label("distance")
            stmt = select(Document, distance_col)
            stmt = self.apply_filters(stmt, filters)
            stmt = stmt.order_by(distance_col.asc()).limit(top_k)

            rows = self.db.execute(stmt).all()
            results: List[Tuple[Document, float, float]] = []
            for doc, raw_dist in rows:
                dist_val = float(raw_dist)
                sim_val = max(0.0, min(1.0, 1.0 - dist_val))
                results.append((doc, sim_val, dist_val))
            return results
        else:
            stmt = select(Document)
            stmt = self.apply_filters(stmt, filters)
            candidates = list(self.db.scalars(stmt).all())

            scored: List[Tuple[Document, float, float]] = []
            for doc in candidates:
                sim = _compute_cosine_similarity(doc.embedding, query_vector)
                dist = 1.0 - sim
                scored.append((doc, sim, dist))

            scored.sort(key=lambda x: x[2])  # sort ascending by distance
            return scored[:top_k]

    def search_keywords(
        self,
        cleaned_query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[Document, float]]:
        """
        Executes sparse keyword search across stored Document records.
        Returns list of (Document, rank_score).
        """
        if self.dialect_name == "postgresql":
            ts_vec = func.to_tsvector("english", Document.content)
            ts_query = func.websearch_to_tsquery("english", cleaned_query)
            rank_col = func.ts_rank(ts_vec, ts_query).label("rank")

            stmt = select(Document, rank_col).where(ts_vec.bool_op("@@")(ts_query))
            stmt = self.apply_filters(stmt, filters)
            stmt = stmt.order_by(rank_col.desc()).limit(top_k)

            rows = self.db.execute(stmt).all()
            return [(doc, float(raw_rank)) for doc, raw_rank in rows]
        else:
            stmt = select(Document)
            stmt = self.apply_filters(stmt, filters)
            candidates = list(self.db.scalars(stmt).all())

            keywords = [w.lower() for w in re.findall(r"\w+", cleaned_query) if len(w) > 1]
            scored: List[Tuple[Document, float]] = []

            for doc in candidates:
                content_lower = doc.content.lower()
                matches = sum(1 for kw in keywords if kw in content_lower)
                if matches > 0:
                    score = matches / max(1, len(keywords))
                    scored.append((doc, score))

            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:top_k]
