import abc
from typing import Any, List, Optional


class BaseReranker(abc.ABC):
    """
    Abstract Base Class for candidate chunk rerankers.
    Defines the contract for scoring query-passage pairs and reordering retrieval candidates.
    """

    @abc.abstractmethod
    def compute_relevance_score(
        self,
        query: str,
        chunk_content: str,
        base_similarity: Optional[float] = None,
        base_rank: Optional[float] = None,
    ) -> float:
        """
        Compute an alignment / relevance score for a given query and chunk content.
        Returns a normalized score in [0.0, 1.0].
        """
        pass

    @abc.abstractmethod
    def rerank(
        self,
        query: str,
        items: List[Any],
        top_k: Optional[int] = None,
    ) -> List[Any]:
        """
        Rerank candidate items according to query relevance and attach 'rerank_score'.
        """
        pass
