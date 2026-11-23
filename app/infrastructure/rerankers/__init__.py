from app.infrastructure.rerankers.base import BaseReranker
from app.infrastructure.rerankers.heuristic import HeuristicReranker


def get_reranker() -> BaseReranker:
    """
    Factory to retrieve the candidate chunk reranker.
    Returns the high-performance HeuristicReranker.
    """
    return HeuristicReranker()


__all__ = [
    "BaseReranker",
    "HeuristicReranker",
    "get_reranker",
]
