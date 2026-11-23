from typing import Any, Dict, List, Optional, Tuple, TypeVar

from app.infrastructure.rerankers import BaseReranker, HeuristicReranker, get_reranker

T = TypeVar("T")


class RerankingService:
    """
    Service layer orchestrator for reranking retrieved candidate chunks.
    Delegates candidate scoring to the configured infrastructure Reranker provider
    (e.g., HeuristicReranker).
    """

    def __init__(self, reranker: Optional[BaseReranker] = None) -> None:
        self.reranker: BaseReranker = reranker or get_reranker()

    def compute_relevance_score(
        self,
        query: str,
        chunk_content: str,
        base_similarity: Optional[float] = None,
        base_rank: Optional[float] = None,
    ) -> float:
        """Calculate alignment score between query and chunk content."""
        return self.reranker.compute_relevance_score(
            query=query,
            chunk_content=chunk_content,
            base_similarity=base_similarity,
            base_rank=base_rank,
        )

    def rerank(
        self,
        query: str,
        items: List[Any],
        top_k: Optional[int] = None,
    ) -> List[Any]:
        """Rerank candidate items and annotate with 'rerank_score'."""
        return self.reranker.rerank(query=query, items=items, top_k=top_k)

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
            item_id = str(getattr(item, "id", "") if hasattr(item, "id") else (item.get("id", "") if isinstance(item, dict) else ""))
            original_pos = next(
                (
                    idx + 1
                    for idx, orig in enumerate(items)
                    if str(getattr(orig, "id", "") if hasattr(orig, "id") else (orig.get("id", "") if isinstance(orig, dict) else "")) == item_id
                ),
                None,
            )
            content = getattr(item, "content", "") if hasattr(item, "content") else (item.get("content", "") if isinstance(item, dict) else "")
            rerank_score = getattr(item, "rerank_score", 0.0) if hasattr(item, "rerank_score") else (item.get("rerank_score", 0.0) if isinstance(item, dict) else 0.0)

            shifts.append({
                "item_id": item_id,
                "content_snippet": content[:90] + "...",
                "original_rank": original_pos,
                "reranked_rank": new_pos,
                "rerank_score": rerank_score,
                "rank_delta": (original_pos - new_pos) if original_pos else 0,
            })

        return {
            "query": query,
            "top_k": top_k,
            "total_candidates": len(items),
            "without_reranking": [
                {
                    "rank": i + 1,
                    "id": str(getattr(it, "id", "") if hasattr(it, "id") else (it.get("id", "") if isinstance(it, dict) else "")),
                    "snippet": (getattr(it, "content", "") if hasattr(it, "content") else (it.get("content", "") if isinstance(it, dict) else ""))[:90] + "...",
                }
                for i, it in enumerate(without_rerank)
            ],
            "with_reranking": [
                {
                    "rank": i + 1,
                    "id": str(getattr(it, "id", "") if hasattr(it, "id") else (it.get("id", "") if isinstance(it, dict) else "")),
                    "score": getattr(it, "rerank_score", 0.0) if hasattr(it, "rerank_score") else (it.get("rerank_score", 0.0) if isinstance(it, dict) else 0.0),
                    "snippet": (getattr(it, "content", "") if hasattr(it, "content") else (it.get("content", "") if isinstance(it, dict) else ""))[:90] + "...",
                }
                for i, it in enumerate(reranked)
            ],
            "rank_shifts": shifts,
        }


__all__ = ["RerankingService"]
