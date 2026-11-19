import uuid
from unittest.mock import MagicMock, patch
import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.document import Document
from app.schemas.document import DocumentIngestRequest
from app.services.embedding_service import GeminiEmbeddingService
from app.services.ingestion_service import IngestionService


@pytest.fixture
def test_db_session():
    """In-memory SQLite database session for document tests."""
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSession = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSession()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def mock_embedding_service():
    """Mock Gemini embedding service returning deterministic 768-dim vectors."""
    service = MagicMock(spec=GeminiEmbeddingService)
    service.model_name = "gemini-embedding-001"
    service.provider_name = "gemini"
    service.model = "gemini-embedding-001"
    service.dimensions = 768
    service.embed_text.return_value = [0.01] * 768
    service.embed_batch.side_effect = lambda texts: [[0.01] * 768 for _ in texts]
    return service


@pytest.fixture
def client(test_db_session):
    def override_get_db():
        yield test_db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestGeminiEmbeddingService:
    def test_missing_api_key_raises_value_error(self):
        with pytest.raises(ValueError, match="GEMINI_API_KEY is not configured"):
            GeminiEmbeddingService(api_key="")

    def test_empty_text_raises_value_error(self):
        with patch("google.genai.Client"):
            service = GeminiEmbeddingService(api_key="test-key")
            with pytest.raises(ValueError, match="cannot be empty"):
                service.embed_text("   ")

    def test_embed_text_calls_gemini_with_768_dims(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            fake_embedding = MagicMock()
            fake_embedding.values = [0.1] * 768
            fake_response = MagicMock()
            fake_response.embeddings = [fake_embedding]
            mock_client.models.embed_content.return_value = fake_response

            service = GeminiEmbeddingService(api_key="test-key", model="gemini-embedding-001", dimensions=768)
            vec = service.embed_text("To reset your password...")

            assert len(vec) == 768
            assert vec[0] == 0.1
            mock_client.models.embed_content.assert_called_once()
            call_kwargs = mock_client.models.embed_content.call_args[1]
            assert call_kwargs["model"] == "gemini-embedding-001"
            assert call_kwargs["contents"] == "To reset your password..."
            assert call_kwargs["config"].output_dimensionality == 768

    def test_embed_batch_calls_gemini(self):
        with patch("google.genai.Client") as mock_client_cls:
            mock_client = MagicMock()
            mock_client_cls.return_value = mock_client

            emb1, emb2 = MagicMock(), MagicMock()
            emb1.values = [0.1] * 768
            emb2.values = [0.2] * 768
            fake_response = MagicMock(embeddings=[emb1, emb2])
            mock_client.models.embed_content.return_value = fake_response

            service = GeminiEmbeddingService(api_key="test-key", model="gemini-embedding-001")
            results = service.embed_batch(["text1", "text2"])

            assert len(results) == 2
            assert len(results[0]) == 768
            assert results[1][0] == 0.2


class TestIngestionService:
    def test_ingest_single_document(self, test_db_session, mock_embedding_service):
        service = IngestionService(db=test_db_session, embedding_service=mock_embedding_service)
        doc = service.ingest_document(
            content="To reset your password, visit the login page.",
            metadata={"category": "account", "source": "faq", "tags": ["password"]},
        )
        test_db_session.commit()

        assert doc.id is not None
        assert doc.content == "To reset your password, visit the login page."
        assert len(doc.embedding) == 768
        assert doc.embedding_model == "gemini-embedding-001"
        assert doc.doc_metadata["category"] == "account"
        assert doc.created_at is not None

        # Retrieve back
        fetched = service.get_document(doc.id)
        assert fetched is not None
        assert fetched.id == doc.id
        assert fetched.doc_metadata["source"] == "faq"

    def test_ingest_batch_documents(self, test_db_session, mock_embedding_service):
        service = IngestionService(db=test_db_session, embedding_service=mock_embedding_service)
        items = [
            DocumentIngestRequest(content="Doc 1", metadata={"tag": "a"}),
            DocumentIngestRequest(content="Doc 2", metadata={"tag": "b"}),
        ]
        docs = service.ingest_batch(items)
        test_db_session.commit()

        assert len(docs) == 2
        assert docs[0].content == "Doc 1"
        assert docs[1].content == "Doc 2"

        listed, total = service.list_documents()
        assert total == 2
        assert len(listed) == 2


class TestDocumentAPIEndpoints:
    def test_ingest_document_endpoint(self, client, mock_embedding_service):
        with patch("app.services.ingestion_service.get_embedding_service", return_value=mock_embedding_service):
            payload = {
                "content": "To reset password, click forgot password.",
                "metadata": {"category": "account", "source": "faq"},
            }
            resp = client.post("/v1/documents", json=payload)
            assert resp.status_code == 201
            data = resp.json()
            assert "id" in data
            assert data["content"] == payload["content"]
            assert data["metadata"]["category"] == "account"
            assert data["embedding_model"] == "gemini-embedding-001"
            assert data["embedding_dimensions"] == 768

            # Fetch it via GET /v1/documents/{id}
            doc_id = data["id"]
            get_resp = client.get(f"/v1/documents/{doc_id}")
            assert get_resp.status_code == 200
            assert get_resp.json()["id"] == doc_id

    def test_ingest_batch_documents_endpoint(self, client, mock_embedding_service):
        with patch("app.services.ingestion_service.get_embedding_service", return_value=mock_embedding_service):
            payload = {
                "documents": [
                    {"content": "First doc", "metadata": {"index": 1}},
                    {"content": "Second doc", "metadata": {"index": 2}},
                ]
            }
            resp = client.post("/v1/documents/batch", json=payload)
            assert resp.status_code == 201
            data = resp.json()
            assert len(data) == 2
            assert data[0]["content"] == "First doc"
            assert data[1]["content"] == "Second doc"

    def test_get_document_not_found(self, client):
        random_id = uuid.uuid4()
        resp = client.get(f"/v1/documents/{random_id}")
        assert resp.status_code == 404

    def test_list_documents_endpoint(self, client):
        resp = client.get("/v1/documents?limit=10&offset=0")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "documents" in data
