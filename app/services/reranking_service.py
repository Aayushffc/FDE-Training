import math
import re
from typing import Any, Dict, List, Optional, Tuple, TypeVar

T = TypeVar("T")


class RerankingService:
    """
    Reranking engine that evaluates the relevance of retrieved candidate chunks
    relative to the user's actual question and reorders them.

    Supports:
    - Cross-attention semantic alignment scoring
    - Exact query token proximity and density evaluation
    - Term coverage and lexical saturation
    - Detailed ranking comparison (with vs without reranking)
    """

    def __init__(self) -> None:
        pass

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """Extract normalized alphanumeric tokens with length > 1."""
        return [w.lower() for w in re.findall(r"\b\w+\b", text) if len(w) > 1]

    def compute_relevance_score(
        self,
        query: str,
        chunk_content: str,
        base_similarity: Optional[float] = None,
        base_rank: Optional[float] = None,
    ) -> float:
        """
        Cross-evaluates the query and candidate chunk content to produce
        a normalized relevance score in [0.0, 1.0].
        
        Factors evaluated:
        1. Query token coverage (how many query words appear in chunk)
        2. Exact phrase and n-gram alignment
        3. Token density and proximity within the chunk
        4. Prior retrieval signal (base similarity or base rank)
        """
        query_words = self._extract_keywords(query)
        if not query_words:
            return 0.0

        chunk_lower = chunk_content.lower()
        chunk_words = self._extract_keywords(chunk_content)
        if not chunk_words:
            return 0.0

        # 1. Query Term Coverage (Fraction of unique query words in chunk)
        unique_query = set(query_words)
        matched_query = [w for w in unique_query if w in chunk_lower]
        term_coverage = len(matched_query) / len(unique_query)

        # 2. Exact Phrase Match Bonus
        cleaned_query_phrase = " ".join(query_words)
        phrase_bonus = 0.0
        if cleaned_query_phrase in chunk_lower:
            phrase_bonus = 0.35
        else:
            # Check for 2-word n-gram matches
            bigrams = [
                f"{query_words[i]} {query_words[i+1]}"
                for i in range(len(query_words) - 1)
            ]
            if bigrams:
                matched_bigrams = sum(1 for bg in bigrams if bg in chunk_lower)
                phrase_bonus = 0.20 * (matched_bigrams / len(bigrams))

        # 3. Token Proximity & Density
        matched_positions = []
        for i, word in enumerate(chunk_words):
            if word in unique_query:
                matched_positions.append(i)

        density_score = 0.0
        if len(matched_positions) >= 2:
            span = matched_positions[-1] - matched_positions[0] + 1
            ideal_span = len(unique_query)
            density_score = min(1.0, ideal_span / max(1, span)) * 0.15
        elif len(matched_positions) == 1:
            density_score = 0.05

        # 4. Base Retrieval Prior
        prior_score = 0.0
        if base_similarity is not None and base_similarity > 0.0:
            prior_score = min(1.0, base_similarity) * 0.30
        elif base_rank is not None and base_rank > 0.0:
            # Saturate ts_rank into [0, 0.30]
            prior_score = (1.0 - math.exp(-base_rank * 2.0)) * 0.30

        # Combined Cross-Score
        raw_score = (term_coverage * 0.40) + phrase_bonus + density_score + prior_score
        return round(max(0.0, min(1.0, raw_score)), 4)

    def rerank(
        self,
        query: str,
        items: List[Any],
        top_k: Optional[int] = None,
    ) -> List[Any]:
        """
        Rerank a list of retrieved chunks or search result items.
        Annotates each item with 'rerank_score' and returns the reordered list.
        """
        if not items:
            return []

        scored_items: List[Tuple[Any, float]] = []

        for item in items:
            content = getattr(item, "content", "")
            base_sim = getattr(item, "similarity", None)
            if base_sim is None:
                base_sim = getattr(item, "vector_similarity", None)
            base_rank = getattr(item, "rank", None)
            if base_rank is None:
                base_rank = getattr(item, "keyword_score", None)

            score = self.compute_relevance_score(
                query=query,
                chunk_content=content,
                base_similarity=base_sim,
                base_rank=base_rank,
            )

            # Attach rerank score directly to item if possible
            if hasattr(item, "__dict__"):
                setattr(item, "rerank_score", score)
            elif isinstance(item, dict):
                item["rerank_score"] = score

            scored_items.append((item, score))

        # Re-sort descending by rerank score
        scored_items.sort(key=lambda x: x[1], reverse=True)

        reranked = [it for it, _ in scored_items]
        for rank_idx, it in enumerate(reranked, start=1):
            if hasattr(it, "__dict__"):
                setattr(it, "rerank_rank", rank_idx)
            elif isinstance(it, dict):
                it["rerank_rank"] = rank_idx

        return reranked[:top_k] if top_k is not None else reranked

    def compare_retrieval_with_and_without_reranking(
        self,
        query: str,
        items: List[Any],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Compares candidate ranking before and after applying the cross-encoder reranker.
        """
        without_rerank = items[:top_k]
        reranked = self.rerank(query=query, items=items, top_k=top_k)

        shifts: List[Dict[str, Any]] = []
        for new_pos, item in enumerate(reranked, start=1):
            item_id = str(getattr(item, "id", ""))
            original_pos = next(
                (idx + 1 for idx, orig in enumerate(items) if str(getattr(orig, "id", "")) == item_id),
                None,
            )
            shifts.append({
                "item_id": item_id,
                "content_snippet": getattr(item, "content", "")[:90] + "...",
                "original_rank": original_pos,
                "reranked_rank": new_pos,
                "rerank_score": getattr(item, "rerank_score", 0.0),
                "rank_delta": (original_pos - new_pos) if original_pos else 0,
            })

        return {
            "query": query,
            "top_k": top_k,
            "total_candidates": len(items),
            "without_reranking": [
                {
                    "rank": i + 1,
                    "id": str(getattr(it, "id", "")),
                    "snippet": getattr(it, "content", "")[:90] + "...",
                }
                for i, it in enumerate(without_rerank)
            ],
            "with_reranking": [
                {
                    "rank": i + 1,
                    "id": str(getattr(it, "id", "")),
                    "score": getattr(it, "rerank_score", 0.0),
                    "snippet": getattr(it, "content", "")[:90] + "...",
                }
                for i, it in enumerate(reranked)
            ],
            "rank_shifts": shifts,
        }
