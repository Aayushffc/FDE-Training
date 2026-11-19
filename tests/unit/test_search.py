import uuid
from unittest.mock import MagicMock
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.document import Document
from app.schemas.search import SemanticSearchRequest
from app.services.embedding_service import GeminiEmbeddingService
from app.services.search_service import SearchService


@pytest.fixture
def search_db_session():
    """In-memory SQLite database session pre-populated with test documents."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()

    # Seed 4 distinct test documents
    # Vector 1: Highly aligned with query (all 1.0s)
    # Vector 2: Moderately aligned (all 0.5s)
    # Vector 3: Orthogonal or low alignment
    # Vector 4: Different category
    docs = [
        Document(
            id=uuid.uuid4(),
            content="To reset your password, visit the login page and click Forgot Password.",
            embedding=[1.0] * 768,
            doc_metadata={"category": "account", "source": "faq"},
            embedding_model="gemini-embedding-001",
        ),
        Document(
            id=uuid.uuid4(),
            content="Two-factor authentication can be enabled in security settings.",
            embedding=[0.8] * 768,
            doc_metadata={"category": "account", "source": "faq"},
            embedding_model="gemini-embedding-001",
        ),
        Document(
            id=uuid.uuid4(),
            content="Update payment methods and review billing invoices.",
            embedding=[0.1] * 768,
            doc_metadata={"category": "billing", "source": "faq"},
            embedding_model="gemini-embedding-001",
        ),
        Document(
            id=uuid.uuid4(),
            content="Upgrade your subscription tier from Starter to Pro.",
            embedding=[0.05] * 768,
            doc_metadata={"category": "subscription", "source": "faq"},
            embedding_model="gemini-embedding-001",
        ),
    ]
    session.add_all(docs)
    session.commit()

    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def mock_embedding_provider():
    """Deterministic mock embedding provider."""
    provider = MagicMock(spec=GeminiEmbeddingService)
    provider.model_name = "gemini-embedding-001"
    provider.dimensions = 768
    # Returns vector identical to first document
    provider.embed_text.return_value = [1.0] * 768
    return provider


@pytest.fixture
def search_client(search_db_session, mock_embedding_provider):
    def override_get_db():
        yield search_db_session

    from app.api.v1.endpoints.search import get_search_service

    def override_get_search_service():
        return SearchService(db=search_db_session, embedding_provider=mock_embedding_provider)

    app.dependency_overrides[get_db] = override_get_db
    app.dependency_overrides[get_search_service] = override_get_search_service
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestSearchService:
    def test_semantic_search_ranks_by_similarity(self, search_db_session, mock_embedding_provider):
        service = SearchService(db=search_db_session, embedding_provider=mock_embedding_provider)
        response = service.semantic_search(query="How do I reset password?", top_k=3)

        assert response.query == "How do I reset password?"
        assert response.total_results == 3
        # First document should have highest similarity
        assert "password" in response.results[0].content.lower()
        assert response.results[0].similarity >= response.results[1].similarity

    def test_semantic_search_with_category_filter(self, search_db_session, mock_embedding_provider):
        service = SearchService(db=search_db_session, embedding_provider=mock_embedding_provider)
        # Restrict to billing only
        response = service.semantic_search(
            query="Invoice issues",
            top_k=5,
            filters={"category": "billing"},
        )

        assert response.total_results == 1
        assert response.results[0].metadata["category"] == "billing"
        assert "billing invoices" in response.results[0].content

    def test_empty_query_raises_value_error(self, search_db_session, mock_embedding_provider):
        service = SearchService(db=search_db_session, embedding_provider=mock_embedding_provider)
        with pytest.raises(ValueError, match="cannot be empty"):
            service.semantic_search(query="   ")


class TestSearchEndpoints:
    def test_post_search_semantic_endpoint(self, search_client):
        payload = {
            "query": "I can't access my account",
            "top_k": 2,
            "filters": {"category": "account"},
        }
        resp = search_client.post("/search/semantic", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert data["query"] == "I can't access my account"
        assert len(data["results"]) == 2
        for item in data["results"]:
            assert item["metadata"]["category"] == "account"
            assert "similarity" in item
            assert "distance" in item
            assert "id" in item

    def test_post_v1_search_semantic_endpoint(self, search_client):
        payload = {
            "query": "How do I upgrade subscription?",
            "top_k": 1,
            "filters": {"category": "subscription"},
        }
        resp = search_client.post("/v1/search/semantic", json=payload)
        assert resp.status_code == 200
        data = resp.json()

        assert len(data["results"]) == 1
        assert data["results"][0]["metadata"]["category"] == "subscription"
        assert "Pro" in data["results"][0]["content"]

    def test_empty_query_validation_error(self, search_client):
        resp = search_client.post("/search/semantic", json={"query": ""})
        assert resp.status_code == 422
