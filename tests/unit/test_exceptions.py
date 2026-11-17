from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.exceptions import (
    AppBaseException,
    LLMProviderError,
    StructuredOutputValidationError,
    register_exception_handlers,
)


class TestExceptionClasses:
    def test_app_base_exception_defaults(self) -> None:
        exc = AppBaseException("Something broke")
        assert exc.message == "Something broke"
        assert exc.status_code == 500
        assert exc.error_code == "internal_error"
        assert exc.provider is None
        assert str(exc) == "Something broke"

    def test_llm_provider_error_fields(self) -> None:
        exc = LLMProviderError("openai", "Provider failed")
        assert exc.provider == "openai"
        assert exc.status_code == 502
        assert exc.error_code == "llm_provider_error"
        assert exc.message == "Provider failed"

    def test_structured_output_validation_error_fields(self) -> None:
        exc = StructuredOutputValidationError("anthropic", "Invalid JSON schema")
        assert exc.status_code == 422
        assert exc.error_code == "structured_output_validation_error"
        assert exc.provider == "anthropic"
        assert exc.message == "Invalid JSON schema"


class TestExceptionHandlersViaEndpoint:
    def _build_test_app(self) -> TestClient:
        test_app = FastAPI()
        register_exception_handlers(test_app)

        @test_app.get("/custom-error")
        async def raise_custom() -> None:
            raise LLMProviderError("openai", "Provider request timed out", status_code=504)

        @test_app.get("/validation-domain-error")
        async def raise_structured() -> None:
            raise StructuredOutputValidationError("anthropic", "Schema mismatch after retries")

        @test_app.get("/secret-leak-error")
        async def raise_secret_error() -> None:
            raise LLMProviderError("openai", "Upstream failed using key sk-supersecret12345", status_code=502)

        @test_app.get("/unexpected-error")
        async def raise_unexpected() -> None:
            raise TypeError("unexpected internal crash with sk-hidden999")

        return TestClient(test_app, raise_server_exceptions=False)

    def test_custom_exception_returns_consistent_json(self) -> None:
        client = self._build_test_app()
        response = client.get("/custom-error")
        assert response.status_code == 504
        assert response.json() == {
            "error": "llm_provider_error",
            "provider": "openai",
            "detail": "Provider request timed out",
        }

    def test_structured_validation_error_returns_422_json(self) -> None:
        client = self._build_test_app()
        response = client.get("/validation-domain-error")
        assert response.status_code == 422
        assert response.json() == {
            "error": "structured_output_validation_error",
            "provider": "anthropic",
            "detail": "Schema mismatch after retries",
        }

    def test_secret_redacted_in_domain_error_response(self) -> None:
        client = self._build_test_app()
        response = client.get("/secret-leak-error")
        assert response.status_code == 502
        body = response.json()
        assert "sk-supersecret12345" not in body["detail"]
        assert "[REDACTED]" in body["detail"]

    def test_unhandled_exception_returns_consistent_500_without_leaking_internals(self) -> None:
        client = self._build_test_app()
        response = client.get("/unexpected-error")
        assert response.status_code == 500
        body = response.json()
        assert body == {
            "error": "internal_error",
            "provider": None,
            "detail": "An unexpected internal error occurred.",
        }
        assert "sk-hidden999" not in body["detail"]
