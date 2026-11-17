"""
Level 4 — Live API smoke tests.

Skipped by default during normal `uv run pytest`.
Run explicitly with:
    uv run pytest -m live
"""

from dotenv import load_dotenv
from fastapi.testclient import TestClient
import pytest

from app.config import load_settings
from app.main import app
from app.schemas.chat import ChatResponse

load_dotenv(override=False)

_PLACEHOLDER_PREFIXES = ("your_", "changeme", "replace_")


def _has_real_key(key: str | None) -> bool:
    if not key:
        return False
    lowered = key.strip().lower()
    return not lowered.startswith(_PLACEHOLDER_PREFIXES)


def _refresh_live_settings(monkeypatch: pytest.MonkeyPatch):
    live_cfg = load_settings()
    monkeypatch.setattr("app.config.settings", live_cfg)
    monkeypatch.setattr("app.services.openai_service.settings", live_cfg)
    monkeypatch.setattr("app.services.anthropic_service.settings", live_cfg)
    monkeypatch.setattr("app.services.groq_service.settings", live_cfg)
    return live_cfg


@pytest.mark.live
class TestLiveProviderSmoke:
    def test_live_openai_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _refresh_live_settings(monkeypatch)
        if not _has_real_key(cfg.openai_api_key):
            pytest.skip("OPENAI_API_KEY not configured with a live key")

        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"message": "Reply with exactly the word: pong", "provider": "openai"},
        )
        assert response.status_code == 200, response.text
        parsed = ChatResponse.model_validate(response.json())
        assert parsed.provider == "openai"
        assert parsed.reply.strip() != ""
        assert parsed.input_tokens > 0
        assert parsed.output_tokens > 0
        assert parsed.latency_ms > 0

    def test_live_anthropic_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _refresh_live_settings(monkeypatch)
        if not _has_real_key(cfg.anthropic_api_key):
            pytest.skip("ANTHROPIC_API_KEY not configured with a live key")

        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"message": "Reply with exactly the word: pong", "provider": "anthropic"},
        )
        assert response.status_code == 200, response.text
        parsed = ChatResponse.model_validate(response.json())
        assert parsed.provider == "anthropic"
        assert parsed.reply.strip() != ""
        assert parsed.input_tokens > 0
        assert parsed.output_tokens > 0
        assert parsed.latency_ms > 0

    def test_live_groq_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        cfg = _refresh_live_settings(monkeypatch)
        if not _has_real_key(cfg.groq_api_key):
            pytest.skip("GROQ_API_KEY not configured with a live key")

        client = TestClient(app)
        response = client.post(
            "/chat",
            json={"message": "Reply with exactly the word: pong", "provider": "groq"},
        )
        assert response.status_code == 200, response.text
        parsed = ChatResponse.model_validate(response.json())
        assert parsed.provider == "groq"
        assert parsed.reply.strip() != ""
        assert parsed.input_tokens > 0
        assert parsed.output_tokens > 0
        assert parsed.latency_ms > 0
