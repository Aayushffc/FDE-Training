import asyncio
import inspect
from collections.abc import Callable
from typing import Any

from sqlalchemy.orm import Session

from app.db.session import get_session_factory
from app.schemas.rag import RAGRequest, RAGResponse
from app.services.rag_service import RAGService


class RetrievalAdapter:
    """
    Tool adapter wrapping the existing RAGService without duplicating retrieval logic.
    Opens and closes its own database session per call and normalizes outputs into
    strictly JSON-serializable primitives for LangGraph state/tool messages.
    """

    def __init__(
        self,
        rag_service: RAGService | None = None,
        session_factory: Callable[[], Session] | None = None,
    ) -> None:
        self._rag_service = rag_service
        self._session_factory = session_factory

    @staticmethod
    def _normalize_response(query: str, response: RAGResponse) -> dict[str, Any]:
        return {
            "query": str(query),
            "answer": str(response.answer),
            "is_grounded": bool(response.is_grounded),
            "citations": [str(c) for c in (response.citations or [])],
            "retrieval_strategy": str(response.retrieval_strategy),
            "sources": [
                {
                    "source_label": str(src.source_label),
                    "document_id": str(src.document_id),
                    "chunk_id": str(src.chunk_id),
                    "chunk_index": int(src.chunk_index) if src.chunk_index is not None else None,
                    "content": str(src.content),
                    "relevance_score": (
                        float(src.relevance_score) if src.relevance_score is not None else None
                    ),
                }
                for src in (response.sources or [])
            ],
        }

    @staticmethod
    def _build_request(
        *,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        rerank: bool = True,
        provider: str | None = None,
    ) -> RAGRequest:
        return RAGRequest(
            question=query,
            top_k=top_k,
            retrieval_strategy=strategy,
            rerank=rerank,
            provider=provider,
        )

    def search_knowledge(
        self,
        *,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        rerank: bool = True,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """Execute knowledge retrieval synchronously via the existing RAGService."""
        payload = self._build_request(
            query=query,
            top_k=top_k,
            strategy=strategy,
            rerank=rerank,
            provider=provider,
        )

        if self._rag_service is not None:
            response = self._rag_service.ask(payload)
            return self._normalize_response(query, response)

        factory = self._session_factory or get_session_factory()
        db = factory()
        try:
            service = RAGService(db=db)
            response = service.ask(payload)
            return self._normalize_response(query, response)
        finally:
            db.close()

    async def asearch_knowledge(
        self,
        *,
        query: str,
        top_k: int = 5,
        strategy: str = "hybrid",
        rerank: bool = True,
        provider: str | None = None,
    ) -> dict[str, Any]:
        """
        Execute knowledge retrieval without blocking the asyncio event loop.
        Runs synchronous DB/embedding retrieval in a worker thread when needed.
        """
        payload = self._build_request(
            query=query,
            top_k=top_k,
            strategy=strategy,
            rerank=rerank,
            provider=provider,
        )

        if self._rag_service is not None:
            # Support both async generate_rag_answer and sync mocks cleanly.
            gen = getattr(self._rag_service, "generate_rag_answer", None)
            if inspect.iscoroutinefunction(gen):
                response = await gen(payload)
            else:
                response = await asyncio.to_thread(self._rag_service.ask, payload)
            return self._normalize_response(query, response)

        return await asyncio.to_thread(
            self.search_knowledge,
            query=query,
            top_k=top_k,
            strategy=strategy,
            rerank=rerank,
            provider=provider,
        )


default_retrieval_adapter = RetrievalAdapter()
