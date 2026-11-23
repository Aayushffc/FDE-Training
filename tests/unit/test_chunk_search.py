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
from app.models.document_chunk import DocumentChunk
from app.schemas.chunk import ChunkDocumentRequest, DocumentChunkIngestRequest
from app.schemas.search import HybridSearchRequest, KeywordSearchRequest, SemanticSearchRequest
from app.services.embedding_service import GeminiEmbeddingService
from app.services.ingestion_service import IngestionService
from app.services.search_service import SearchService


@pytest.fixture
def chunk_test_db():
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
def mock_embedder():
    svc = MagicMock(spec=GeminiEmbeddingService)
    svc.model_name = "gemini-embedding-001"
    svc.embed_text.return_value = [0.1] * 768
    svc.embed_batch.side_effect = lambda texts: [[0.1] * 768 for _ in texts]
    return svc


@pytest.fixture
def test_client(chunk_test_db):
    def override_db():
        yield chunk_test_db

    app.dependency_overrides[get_db] = override_db
    with TestClient(app) as client:
        yield client
    app.dependency_overrides.clear()


class TestChunkIngestionPipeline:
    def test_ingest_document_with_chunks(self, chunk_test_db, mock_embedder):
        ingestion = IngestionService(db=chunk_test_db, embedding_provider=mock_embedder)
        content = (
            "Section 1: Account setup and password credentials. " * 3
            + "Section 2: Payment billing methods and credit card invoices. " * 3
            + "Section 3: Plan tier subscription cancellation and refunds. " * 3
        )
        doc, chunks = ingestion.ingest_document_with_chunks(
            content=content,
            metadata={"source": "test_guide", "category": "account"},
            chunk_size=150,
            chunk_overlap=30,
        )

        assert doc.id is not None
        assert len(chunks) >= 2
        for i, chunk in enumerate(chunks):
            assert chunk.document_id == doc.id
            assert chunk.chunk_index == i
            assert chunk.chunk_metadata["category"] == "account"
            assert chunk.chunk_metadata["chunk_index"] == i
            assert chunk.chunk_metadata["total_chunks"] == len(chunks)
            assert len(chunk.embedding) == 768

    def test_chunk_existing_document(self, chunk_test_db, mock_embedder):
        ingestion = IngestionService(db=chunk_test_db, embedding_provider=mock_embedder)
        doc = ingestion.ingest_document(
            content="Alpha paragraph on credentials. Beta paragraph on billing. Gamma paragraph on subscriptions.",
            metadata={"category": "billing"},
        )
        chunk_test_db.commit()

        chunks = ingestion.chunk_existing_document(
            document_id=doc.id,
            chunk_size=50,
            chunk_overlap=15,
        )

        assert len(chunks) >= 2
        retrieved_chunks, count = ingestion.list_chunks(document_id=doc.id)
        assert count == len(chunks)
        assert len(retrieved_chunks) == len(chunks)

    def test_chunk_existing_document_not_found(self, chunk_test_db, mock_embedder):
        ingestion = IngestionService(db=chunk_test_db, embedding_provider=mock_embedder)
        with pytest.raises(ValueError, match="not found"):
            ingestion.chunk_existing_document(document_id=uuid.uuid4())


class TestChunkRetrieval:
    @pytest.fixture
    def populated_db(self, chunk_test_db):
        doc = Document(
            id=uuid.uuid4(),
            content="Parent document on customer service.",
            embedding=[0.5] * 768,
            doc_metadata={"category": "support"},
            embedding_model="gemini-embedding-001",
        )
        chunk_test_db.add(doc)
        chunk_test_db.flush()

        c1 = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=0,
            content="Reset forgotten password credentials via secure email token link.",
            embedding=[1.0] * 768,
            chunk_metadata={"category": "account", "topic": "auth"},
            embedding_model="gemini-embedding-001",
        )
        c2 = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=1,
            content="Payment methods include credit cards and billing invoices with VAT.",
            embedding=[1.0] * 384 + [0.0] * 384,
            chunk_metadata={"category": "billing", "topic": "payments"},
            embedding_model="gemini-embedding-001",
        )
        c3 = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=2,
            content="Subscriptions can be upgraded from monthly to annual billing tiers.",
            embedding=[0.0] * 384 + [1.0] * 384,
            chunk_metadata={"category": "subscription", "topic": "billing"},
            embedding_model="gemini-embedding-001",
        )
        chunk_test_db.add_all([c1, c2, c3])
        chunk_test_db.commit()
        return chunk_test_db

    def test_chunk_vector_search(self, populated_db, mock_embedder):
        mock_embedder.embed_text.return_value = [1.0] * 768
        searcher = SearchService(db=populated_db, embedding_provider=mock_embedder)

        res = searcher.chunk_vector_search(query="password credentials", top_k=2)
        assert res.total_results >= 2
        assert res.search_type == "chunk_vector"
        assert "password" in res.results[0].content.lower()
        assert res.results[0].similarity > res.results[1].similarity

    def test_chunk_keyword_search(self, populated_db, mock_embedder):
        searcher = SearchService(db=populated_db, embedding_provider=mock_embedder)

        res = searcher.chunk_keyword_search(query="billing invoices", top_k=2)
        assert res.total_results >= 1
        assert res.search_type == "chunk_keyword"
        assert "billing" in res.results[0].content.lower()

    def test_chunk_hybrid_search_rrf(self, populated_db, mock_embedder):
        mock_embedder.embed_text.return_value = [1.0] * 768
        searcher = SearchService(db=populated_db, embedding_provider=mock_embedder)

        res = searcher.chunk_hybrid_search(
            query="password credentials",
            top_k=3,
            alpha=0.6,
            rrf_k=60,
            fusion_method="rrf",
        )

        assert res.total_results >= 1
        assert res.search_type == "hybrid"
        top = res.results[0]
        assert top.combined_score > 0
        assert top.document_id is not None
        assert "password" in top.content.lower()


class TestChunkEndpoints:
    def test_post_chunk_document_endpoint(self, test_client, chunk_test_db, mock_embedder):
        from app.api.v1.endpoints.documents import get_db

        doc = Document(
            id=uuid.uuid4(),
            content="Paragraph one describes access control. Paragraph two details pricing rules.",
            embedding=[0.2] * 768,
            doc_metadata={"category": "account"},
            embedding_model="gemini-embedding-001",
        )
        chunk_test_db.add(doc)
        chunk_test_db.commit()

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "app.services.ingestion_service.get_embedding_provider",
                lambda: mock_embedder,
            )
            response = test_client.post(
                f"/v1/documents/{doc.id}/chunks",
                json={"chunk_size": 50, "chunk_overlap": 15},
            )
            assert response.status_code == 201
            data = response.json()
            assert isinstance(data, list)
            assert len(data) >= 2
            assert data[0]["document_id"] == str(doc.id)

    def test_get_document_chunks_endpoint(self, test_client, chunk_test_db):
        doc = Document(
            id=uuid.uuid4(),
            content="Sample text",
            embedding=[0.2] * 768,
            doc_metadata={},
            embedding_model="gemini-embedding-001",
        )
        chunk_test_db.add(doc)
        chunk_test_db.flush()

        chunk = DocumentChunk(
            id=uuid.uuid4(),
            document_id=doc.id,
            chunk_index=0,
            content="Sample text chunk",
            embedding=[0.2] * 768,
            chunk_metadata={"chunk_index": 0},
            embedding_model="gemini-embedding-001",
        )
        chunk_test_db.add(chunk)
        chunk_test_db.commit()

        resp = test_client.get(f"/v1/documents/{doc.id}/chunks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 1
        assert data["chunks"][0]["id"] == str(chunk.id)

    def test_post_chunked_search_endpoint(self, test_client, chunk_test_db, mock_embedder):
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                "app.services.search_service.get_embedding_provider",
                lambda: mock_embedder,
            )
            resp = test_client.post(
                "/v1/search/chunked",
                json={"query": "account reset", "top_k": 3, "strategy": "hybrid", "alpha": 0.5},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["search_type"] == "hybrid"
            assert "results" in data
