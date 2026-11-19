import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool
from fastapi.testclient import TestClient

from app.db.base import Base
from app.db.session import get_db
from app.main import app
from app.models.prompt import PromptCategory, PromptVersion
from app.services.prompt_service import PromptService


# Create an in-memory SQLite database for unit tests
@pytest.fixture
def db_session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(bind=engine)


@pytest.fixture
def client(db_session):
    def override_get_db():
        yield db_session

    app.dependency_overrides[get_db] = override_get_db
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestDBPromptService:
    def test_create_and_retrieve_category(self, db_session):
        service = PromptService(db_session)
        category = service.get_or_create_category(
            name="rag_synthesis",
            description="Prompts tailored for RAG grounding",
        )
        assert category.id is not None
        assert category.name == "rag_synthesis"

        # Re-fetching returns existing
        same_cat = service.get_or_create_category(name="rag_synthesis")
        assert same_cat.id == category.id

    def test_create_and_retrieve_prompt_version(self, db_session):
        service = PromptService(db_session)
        prompt_text = "You are a grounding assistant. Only use retrieved context."
        version = service.create_prompt_version(
            category_name="rag_synthesis",
            version="v1",
            name="Grounding Baseline",
            prompt_text=prompt_text,
            description="Initial version",
            variables=["context", "query"],
        )
        assert version.id is not None
        assert version.version == "v1"
        assert version.prompt_text == prompt_text
        assert version.category_name == "rag_synthesis"
        assert version.variables == ["context", "query"]

        fetched = service.get_prompt_version("rag_synthesis", "v1")
        assert fetched is not None
        assert fetched.id == version.id
        assert fetched.prompt_text == prompt_text

    def test_update_existing_prompt_version(self, db_session):
        service = PromptService(db_session)
        service.create_prompt_version(
            category_name="support",
            version="v1",
            prompt_text="Original text",
        )
        updated = service.create_prompt_version(
            category_name="support",
            version="v1",
            prompt_text="Updated text",
            description="Revised guidelines",
        )
        assert updated.prompt_text == "Updated text"
        assert updated.description == "Revised guidelines"

    def test_list_categories_and_versions(self, db_session):
        service = PromptService(db_session)
        service.create_prompt_version(
            category_name="cat_a",
            version="v1",
            prompt_text="Text A1",
        )
        service.create_prompt_version(
            category_name="cat_a",
            version="v2",
            prompt_text="Text A2",
        )
        service.create_prompt_version(
            category_name="cat_b",
            version="v1",
            prompt_text="Text B1",
        )

        categories = service.list_categories()
        assert len(categories) == 2
        assert "cat_a" in categories
        assert "cat_b" in categories

        cat_models = service.get_all_categories_with_versions()
        assert len(cat_models) == 2

        cat_a_versions = service.list_versions("cat_a")
        assert len(cat_a_versions) == 2
        assert cat_a_versions == ["v1", "v2"]


class TestPromptAPIEndpoints:
    def test_create_and_fetch_prompt_without_db_prefix(self, client):
        create_payload = {
            "category_name": "customer_support",
            "version": "v10",
            "name": "Support v10",
            "prompt_text": "Always greet the customer with their order status.",
            "description": "Auto-order test prompt",
            "variables": ["order_status"],
            "is_active": True,
        }
        # POST /v1/prompts
        post_resp = client.post("/v1/prompts", json=create_payload)
        assert post_resp.status_code == 201
        data = post_resp.json()
        assert data["version"] == "v10"
        assert data["prompt_text"] == create_payload["prompt_text"]
        assert data["category_name"] == "customer_support"

        # GET /v1/prompts/{category}/{version}
        get_resp = client.get("/v1/prompts/customer_support/v10")
        assert get_resp.status_code == 200
        get_data = get_resp.json()
        assert get_data["version"] == "v10"
        assert get_data["prompt_text"] == create_payload["prompt_text"]

        # GET /v1/prompts/{version} shortcut
        shortcut_resp = client.get("/v1/prompts/v10")
        assert shortcut_resp.status_code == 200
        assert shortcut_resp.json()["version"] == "v10"

    def test_get_nonexistent_returns_404(self, client):
        resp = client.get("/v1/prompts/nonexistent_cat/v999")
        assert resp.status_code == 404

    def test_list_prompts_endpoint(self, client):
        client.post("/v1/prompts", json={
            "category_name": "analytics",
            "version": "v1",
            "prompt_text": "Analyze query metrics",
        })
        resp = client.get("/v1/prompts")
        assert resp.status_code == 200
        cats = resp.json()
        assert any(c["name"] == "analytics" for c in cats)

    def test_categories_table_endpoints(self, client):
        # 1. Create a category via POST /v1/prompts/categories
        create_cat_resp = client.post("/v1/prompts/categories", json={
            "name": "code_review",
            "description": "Specialized prompts for code inspection",
        })
        assert create_cat_resp.status_code == 201
        cat_data = create_cat_resp.json()
        assert cat_data["name"] == "code_review"
        assert cat_data["id"] is not None

        # Duplicate category returns 409
        dup_resp = client.post("/v1/prompts/categories", json={
            "name": "code_review",
        })
        assert dup_resp.status_code == 409

        # 2. List categories via GET /v1/prompts/categories
        list_cat_resp = client.get("/v1/prompts/categories")
        assert list_cat_resp.status_code == 200
        cat_list = list_cat_resp.json()
        assert any(c["name"] == "code_review" for c in cat_list)

        # 3. Add a prompt version to code_review
        ver_resp = client.post("/v1/prompts", json={
            "category_name": "code_review",
            "version": "v1",
            "prompt_text": "Check for SQL injection vulnerabilities.",
        })
        assert ver_resp.status_code == 201
        version_id = ver_resp.json()["id"]

        # 4. Get category by name via GET /v1/prompts/categories/code_review
        get_cat_resp = client.get("/v1/prompts/categories/code_review")
        assert get_cat_resp.status_code == 200
        cat_with_versions = get_cat_resp.json()
        assert cat_with_versions["name"] == "code_review"
        assert len(cat_with_versions["prompt_versions"]) == 1
        assert cat_with_versions["prompt_versions"][0]["version"] == "v1"

        # 5. List versions for category via GET /v1/prompts/categories/code_review/versions
        list_vers_resp = client.get("/v1/prompts/categories/code_review/versions")
        assert list_vers_resp.status_code == 200
        versions_list = list_vers_resp.json()
        assert len(versions_list) == 1
        assert versions_list[0]["version"] == "v1"

        # 6. Get prompt version by ID via GET /v1/prompts/versions/{version_id}
        get_ver_id_resp = client.get(f"/v1/prompts/versions/{version_id}")
        assert get_ver_id_resp.status_code == 200
        assert get_ver_id_resp.json()["id"] == version_id
        assert get_ver_id_resp.json()["prompt_text"] == "Check for SQL injection vulnerabilities."

        # 7. Delete prompt version via DELETE /v1/prompts/code_review/v1
        del_resp = client.delete("/v1/prompts/code_review/v1")
        assert del_resp.status_code == 204

        # Verify 404 after deletion
        get_after_del = client.get("/v1/prompts/code_review/v1")
        assert get_after_del.status_code == 404
