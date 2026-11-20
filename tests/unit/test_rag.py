import uuid
from unittest.mock import AsyncMock, MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.infrastructure.llm_providers.base import BaseLLMProvider
from app.infrastructure.llm_providers.registry import LLMProviderRegistry
from app.main import app
from app.models.document import Document
from app.models.document_chunk import DocumentChunk
from app.modules.chat.schemas import ChatResponse
from app.schemas.rag import RAGRequest
from app.services.context_service import ContextEngineeringService
from app.services.embedding_service import GeminiEmbeddingService
from app.services.rag_service import RAGService
from app.services.reranking_service import RerankingService
from app.services.search_service import SearchService


@pytest.fixture
def rag_test_db():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()

    doc = Document(
        id=uuid.uuid4(),
        content="Parent FAQ documentation on security and billing.",
        embedding=[0.5] * 768,
        doc_metadata={"category": "security"},
        embedding_model="gemini-embedding-001",
    )
    session.add(doc)
    session.flush()

    c1 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=0,
        chunk_size=400,
        overlap=80,
        content="To reset your password, visit the login screen and click 'Forgot Password'. A reset token link will be dispatched via email.",
        embedding=[1.0] * 768,
        chunk_metadata={"category": "account", "topic": "auth"},
        embedding_model="gemini-embedding-001",
    )
    c2 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=1,
        chunk_size=400,
        overlap=80,
        content="Following five consecutive failed login attempts, accounts enter a 30-minute cooling-off lockout period.",
        embedding=[0.8] * 384 + [0.2] * 384,
        chunk_metadata={"category": "account", "topic": "lockout"},
        embedding_model="gemini-embedding-001",
    )
    c3 = DocumentChunk(
        id=uuid.uuid4(),
        document_id=doc.id,
        chunk_index=2,
        chunk_size=400,
        overlap=80,
        content="European Union businesses with a valid VIES VAT number receive reverse charge tax exemptions.",
        embedding=[0.1] * 768,
        chunk_metadata={"category": "billing", "topic": "tax"},
        embedding_model="gemini-embedding-001",
    )
    session.add_all([c1, c2, c3])
    session.commit()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def mock_embedder():
    svc = MagicMock(spec=GeminiEmbeddingService)
    svc.model_name = "gemini-embedding-001"
    svc.embed_text.return_value = [1.0] * 768
    svc.embed_batch.side_effect = lambda texts: [[1.0] * 768 for _ in texts]
    return svc


@pytest.fixture
def mock_llm_provider():
    provider = MagicMock(spec=BaseLLMProvider)
    provider.provider_name = "mock_provider"
    provider.default_model = "mock-model"

    async def fake_completion(*args, **kwargs):
        sys_prompt = kwargs.get("system_prompt", "")
        msgs = kwargs.get("messages", [])
        payload_msg = kwargs.get("payload_or_message") or (args[0] if args else "")
        prompt_text = payload_msg if isinstance(payload_msg, str) else (msgs[0]["content"] if msgs else "")

        if "weather in tokyo" in prompt_text.lower() or "quantum teleportation" in prompt_text.lower():
            return ChatResponse(
                reply="I don't know based on the provided context.",
                model="mock-model",
                provider="groq",
                input_tokens=50,
                output_tokens=15,
                latency_ms=50.0,
                estimated_cost_usd=0.0001,
            )

        return ChatResponse(
            reply="You can reset your password by selecting 'Forgot Password' on the sign-in screen [Source 1].",
            model="mock-model",
            provider="groq",
            input_tokens=120,
            output_tokens=30,
            latency_ms=100.0,
            estimated_cost_usd=0.0002,
        )

    provider.generate_chat_completion = AsyncMock(side_effect=fake_completion)
    return provider


@pytest.fixture
def rag_client(rag_test_db):
    def override_db():
        yield rag_test_db

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


class TestRerankingService:
    def test_reranker_promotes_relevant_chunk(self):
        reranker = RerankingService()
        query = "cooling-off lockout period after failed attempts"

        class DummyChunk:
            def __init__(self, id_val, content, similarity):
                self.id = id_val
                self.content = content
                self.similarity = similarity

        c1 = DummyChunk("1", "General password settings and login options.", 0.70)
        c2 = DummyChunk("2", "Accounts enter a cooling-off lockout period after failed authentication attempts.", 0.65)

        # Before reranking: c1 is first due to raw similarity
        items = [c1, c2]
        reranked = reranker.rerank(query=query, items=items, top_k=2)

        # After reranking: c2 should be promoted because it contains exact phrase matches
        assert reranked[0].id == "2"
        assert reranked[0].rerank_score > reranked[1].rerank_score

    def test_compare_retrieval_with_and_without_reranking(self):
        reranker = RerankingService()
        query = "VIES VAT registration"

        class DummyChunk:
            def __init__(self, id_val, content, similarity):
                self.id = id_val
                self.content = content
                self.similarity = similarity

        c1 = DummyChunk("a", "Invoices are generated on the first of each month.", 0.80)
        c2 = DummyChunk("b", "Businesses with valid VIES VAT registration number receive reverse charge.", 0.70)

        comp = reranker.compare_retrieval_with_and_without_reranking(query=query, items=[c1, c2], top_k=2)
        assert comp["without_reranking"][0]["id"] == "a"
        assert comp["with_reranking"][0]["id"] == "b"
        assert len(comp["rank_shifts"]) == 2


class TestContextEngineeringService:
    def test_deduplicate_chunks(self):
        ctx_service = ContextEngineeringService()

        class DummyChunk:
            def __init__(self, id_val, content):
                self.id = id_val
                self.content = content

        c1 = DummyChunk("1", "Exact duplicate content for password reset.")
        c2 = DummyChunk("1", "Exact duplicate content for password reset.")
        c3 = DummyChunk("2", "Completely different content about billing invoices.")

        deduped = ctx_service.deduplicate_chunks([c1, c2, c3])
        assert len(deduped) == 2
        assert deduped[0].id == "1"
        assert deduped[1].id == "2"

    def test_trim_context_enforces_budget(self):
        ctx_service = ContextEngineeringService()

        class DummyChunk:
            def __init__(self, id_val, content):
                self.id = id_val
                self.content = content

        items = [DummyChunk(str(i), "X" * 300) for i in range(10)]
        # Budget = 700 chars -> should admit at most 2 chunks (~380 chars each with overhead)
        trimmed = ctx_service.trim_context(items, max_chars=700)
        assert len(trimmed) == 1 or len(trimmed) == 2

    def test_build_rag_prompt_contains_strict_instructions(self):
        ctx_service = ContextEngineeringService()

        class DummyChunk:
            def __init__(self, id_val, doc_id, index, content):
                self.id = id_val
                self.document_id = doc_id
                self.chunk_index = index
                self.content = content

        chunks = [
            DummyChunk("c1", "d1", 0, "First chunk content."),
            DummyChunk("c2", "d1", 1, "Second chunk content."),
        ]

        sys_prompt, user_prompt, manifest = ctx_service.build_rag_prompt("How to login?", chunks)
        assert "I don't know based on the provided context" in sys_prompt
        assert "[Source 1]" in user_prompt
        assert "[Source 2]" in user_prompt
        assert len(manifest) == 2


class TestRAGServiceAndEndpoints:
    @pytest.mark.asyncio
    async def test_generate_rag_answer_grounded_with_citations(
        self, rag_test_db, mock_embedder, mock_llm_provider
    ):
        search_service = SearchService(db=rag_test_db, embedding_provider=mock_embedder)
        registry = LLMProviderRegistry()
        registry.register("groq", mock_llm_provider)

        rag_service = RAGService(
            db=rag_test_db,
            search_service=search_service,
            registry=registry,
        )

        req = RAGRequest(
            question="How can I reset my forgotten password?",
            retrieval_strategy="hybrid",
            rerank=True,
            provider="groq",
        )

        resp = await rag_service.generate_rag_answer(req)
        assert resp.is_grounded is True
        assert "[Source 1]" in resp.answer
        assert len(resp.sources) >= 1
        assert resp.retrieval_strategy == "hybrid"
        assert resp.reranked is True

    @pytest.mark.asyncio
    async def test_generate_rag_answer_unknown_query_responds_i_dont_know(
        self, rag_test_db, mock_embedder, mock_llm_provider
    ):
        search_service = SearchService(db=rag_test_db, embedding_provider=mock_embedder)
        registry = LLMProviderRegistry()
        registry.register("groq", mock_llm_provider)

        rag_service = RAGService(
            db=rag_test_db,
            search_service=search_service,
            registry=registry,
        )

        req = RAGRequest(
            question="What is the weather in Tokyo tomorrow?",
            retrieval_strategy="vector",
            rerank=False,
            provider="groq",
        )

        resp = await rag_service.generate_rag_answer(req)
        assert resp.is_grounded is False
        assert "i don't know" in resp.answer.lower()

    def test_post_rag_ask_endpoint(self, rag_client, mock_embedder, mock_llm_provider):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "app.services.search_service.get_embedding_provider",
                lambda: mock_embedder,
            )
            from app.infrastructure.llm_providers.registry import default_registry
            default_registry.register("groq", mock_llm_provider)

            resp = rag_client.post(
                "/v1/rag/ask",
                json={
                    "question": "How do I reset my password?",
                    "retrieval_strategy": "hybrid",
                    "rerank": True,
                    "provider": "groq",
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert "answer" in data
            assert "sources" in data
            assert data["retrieval_strategy"] == "hybrid"
            assert data["reranked"] is True
