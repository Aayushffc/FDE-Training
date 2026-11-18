from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.core.exceptions import LLMProviderError
from app.modules.chat.schemas import MAX_MESSAGE_LENGTH, ChatResponse
from tests.conftest import override_settings


class TestHealthAndRootEndpoints:
    def test_v1_health_returns_200(self, client: TestClient) -> None:
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_root_health_returns_404_no_legacy_fallback(self, client: TestClient) -> None:
        response = client.get("/health")
        assert response.status_code == 404

    def test_root_redirects_to_docs(self, client: TestClient) -> None:
        response = client.get("/", follow_redirects=False)
        assert response.status_code == 307
        assert response.headers["location"] == "/docs"


class TestChatEndpoint:
    def test_v1_chat_with_openai_calls_openai_service(self, client: TestClient) -> None:
        expected = ChatResponse(
            provider="openai",
            model="gpt-6-luna",
            reply="Hello from OpenAI",
            input_tokens=12,
            output_tokens=8,
            latency_ms=120.4,
            estimated_cost_usd=0.0000052,
        )
        with patch(
            "app.modules.chat.service._PROVIDER_HANDLERS",
            {"openai": AsyncMock(return_value=expected)},
        ) as mocked_handlers:
            response = client.post("/v1/chat", json={"message": "Hi", "provider": "openai"})

        assert response.status_code == 200
        mocked_handlers["openai"].assert_awaited_once()
        data = response.json()
        assert data["provider"] == "openai"
        assert data["reply"] == "Hello from OpenAI"
        assert data["input_tokens"] == 12
        assert data["output_tokens"] == 8
        assert data["latency_ms"] == 120.4
        assert data["estimated_cost_usd"] == 0.0000052

    def test_legacy_unversioned_chat_returns_404(self, client: TestClient) -> None:
        response = client.post("/chat", json={"message": "Hi", "provider": "openai"})
        assert response.status_code == 404

    def test_chat_with_anthropic_calls_anthropic_service(self, client: TestClient) -> None:
        expected = ChatResponse(
            provider="anthropic",
            model="claude-haiku-5-5",
            reply="Hello from Anthropic",
            input_tokens=20,
            output_tokens=15,
            latency_ms=210.0,
            estimated_cost_usd=0.0000095,
        )
        with patch(
            "app.modules.chat.service._PROVIDER_HANDLERS",
            {"anthropic": AsyncMock(return_value=expected)},
        ) as mocked_handlers:
            response = client.post("/v1/chat", json={"message": "Hi", "provider": "anthropic"})

        assert response.status_code == 200
        mocked_handlers["anthropic"].assert_awaited_once()
        assert response.json()["provider"] == "anthropic"

    def test_chat_with_groq_calls_groq_service(self, client: TestClient) -> None:
        expected = ChatResponse(
            provider="groq",
            model="openai/gpt-oss-120b",
            reply="Hello from Groq",
            input_tokens=95,
            output_tokens=40,
            latency_ms=604.45,
            estimated_cost_usd=0.00003825,
        )
        with patch(
            "app.modules.chat.service._PROVIDER_HANDLERS",
            {"groq": AsyncMock(return_value=expected)},
        ) as mocked_handlers:
            response = client.post("/v1/chat", json={"message": "Hi", "provider": "groq"})

        assert response.status_code == 200
        mocked_handlers["groq"].assert_awaited_once()
        assert response.json()["provider"] == "groq"

    def test_no_model_supplied_uses_provider_default_from_env(
        self,
        client: TestClient,
        mock_settings: Settings,
    ) -> None:
        fake_response = SimpleNamespace(
            model="gpt-6-luna",
            choices=[SimpleNamespace(message=SimpleNamespace(content="Default model reply"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=5),
        )
        with patch("app.infrastructure.llm_providers.openai.AsyncOpenAI") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.return_value = fake_response

            response = client.post("/v1/chat", json={"message": "Hi", "provider": "openai"})

        assert response.status_code == 200
        assert inst.chat.completions.create.call_args.kwargs["model"] == mock_settings.default_openai_model
        data = response.json()
        assert data["model"] == "gpt-6-luna"
        assert data["input_tokens"] == 10
        assert data["output_tokens"] == 5
        assert data["latency_ms"] >= 0
        assert data["estimated_cost_usd"] > 0

    def test_configured_env_model_is_used(
        self,
        client: TestClient,
        mock_settings: Settings,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        override_settings(monkeypatch, mock_settings, default_openai_model="gpt-4o-mini")
        fake_response = SimpleNamespace(
            model="gpt-4o-mini",
            choices=[SimpleNamespace(message=SimpleNamespace(content="Env model reply"))],
            usage=SimpleNamespace(prompt_tokens=10, completion_tokens=10),
        )
        with patch("app.infrastructure.llm_providers.openai.AsyncOpenAI") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.return_value = fake_response

            response = client.post("/v1/chat", json={"message": "Hi", "provider": "openai"})

        assert response.status_code == 200
        assert inst.chat.completions.create.call_args.kwargs["model"] == "gpt-4o-mini"
        assert response.json()["model"] == "gpt-4o-mini"

    def test_empty_message_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat", json={"message": "", "provider": "openai"})
        assert response.status_code == 422

    def test_oversized_message_returns_422(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat",
            json={"message": "x" * (MAX_MESSAGE_LENGTH + 1), "provider": "openai"},
        )
        assert response.status_code == 422

    def test_unsupported_provider_returns_422(self, client: TestClient) -> None:
        response = client.post("/v1/chat", json={"message": "Hello", "provider": "unknown"})
        assert response.status_code == 422

    def test_provider_error_returns_consistent_error_response(self, client: TestClient) -> None:
        failing = AsyncMock(side_effect=LLMProviderError("openai", "OpenAI upstream service error", status_code=502))
        with patch("app.modules.chat.service._PROVIDER_HANDLERS", {"openai": failing}):
            response = client.post("/v1/chat", json={"message": "Hello", "provider": "openai"})

        assert response.status_code == 502
        assert response.json() == {
            "error": "llm_provider_error",
            "provider": "openai",
            "detail": "OpenAI upstream service error",
        }

    def test_provider_timeout_returns_504_response(self, client: TestClient) -> None:
        timed_out = AsyncMock(side_effect=LLMProviderError("anthropic", "Anthropic request timed out.", status_code=504))
        with patch("app.modules.chat.service._PROVIDER_HANDLERS", {"anthropic": timed_out}):
            response = client.post("/v1/chat", json={"message": "Hello", "provider": "anthropic"})

        assert response.status_code == 504
        assert response.json() == {
            "error": "llm_provider_error",
            "provider": "anthropic",
            "detail": "Anthropic request timed out.",
        }

    def test_list_prompts_endpoint(self, client: TestClient) -> None:
        response = client.get("/v1/chat/prompts")
        assert response.status_code == 200
        data = response.json()
        assert "available_customer_support_versions" in data
        assert "v1" in data["available_customer_support_versions"]
        assert "v2" in data["available_customer_support_versions"]
        assert "v3" in data["available_customer_support_versions"]

        alias_response = client.get("/v1/prompts")
        assert alias_response.status_code == 200
        assert alias_response.json() == data

    def test_chat_with_prompt_version_passes_correct_system_prompt(
        self, client: TestClient, mock_settings: Settings
    ) -> None:
        fake_response = SimpleNamespace(
            model="gpt-6-luna",
            choices=[SimpleNamespace(message=SimpleNamespace(content="Customer support response"))],
            usage=SimpleNamespace(prompt_tokens=50, completion_tokens=25),
        )
        with patch("app.infrastructure.llm_providers.openai.AsyncOpenAI") as mock_cls:
            inst = AsyncMock()
            mock_cls.return_value.__aenter__.return_value = inst
            inst.chat.completions.create.return_value = fake_response

            response = client.post(
                "/v1/chat",
                json={
                    "message": "My order was broken",
                    "provider": "openai",
                    "prompt_version": "v2",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["prompt_version"] == "customer_support/v2"

        call_messages = inst.chat.completions.create.call_args.kwargs["messages"]
        assert call_messages[0]["role"] == "system"
        assert "Senior Tier-3 Customer Support Operations Analyst" in call_messages[0]["content"]

    def test_chat_with_invalid_prompt_version_returns_400(self, client: TestClient) -> None:
        response = client.post(
            "/v1/chat",
            json={
                "message": "My order was broken",
                "provider": "openai",
                "prompt_version": "v999_invalid",
            },
        )
        assert response.status_code == 400
        data = response.json()
        assert data["error"] == "prompt_not_found"
        assert "v999_invalid" in data["detail"]
