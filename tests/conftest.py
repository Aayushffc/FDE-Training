from collections.abc import Generator
from dataclasses import replace

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture
def mock_settings(monkeypatch: pytest.MonkeyPatch) -> Generator[Settings, None, None]:
    """Provide isolated test settings across all provider services."""
    test_cfg = Settings(
        openai_api_key="sk-test-openai-secret-key-12345",
        anthropic_api_key="sk-ant-test-anthropic-secret-key-12345",
        groq_api_key="gsk_test_groq_secret_key_12345",
        default_openai_model="gpt-6-luna",
        default_anthropic_model="claude-haiku-5-5",
        default_groq_model="openai/gpt-oss-120b",
        request_timeout_seconds=10.0,
    )
    monkeypatch.setattr("app.config.settings", test_cfg)
    monkeypatch.setattr("app.services.openai_service.settings", test_cfg)
    monkeypatch.setattr("app.services.anthropic_service.settings", test_cfg)
    monkeypatch.setattr("app.services.groq_service.settings", test_cfg)
    yield test_cfg


def override_settings(monkeypatch: pytest.MonkeyPatch, base: Settings, **changes) -> Settings:
    updated = replace(base, **changes)
    monkeypatch.setattr("app.config.settings", updated)
    monkeypatch.setattr("app.services.openai_service.settings", updated)
    monkeypatch.setattr("app.services.anthropic_service.settings", updated)
    monkeypatch.setattr("app.services.groq_service.settings", updated)
    return updated
