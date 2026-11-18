import json
import logging
from pathlib import Path

import pytest

from app.core.logging import ColoredFormatter, log_llm_call, setup_logging
from app.infrastructure.llm_providers.pricing import calculate_cost_usd


class TestCostCalculation:
    def test_known_model_cost_calculation(self) -> None:
        # gpt-4o-mini: input $0.15/1M, output $0.60/1M
        cost = calculate_cost_usd("gpt-4o-mini", input_tokens=1_000, output_tokens=500)
        assert cost == pytest.approx(0.00045)

    def test_known_groq_model_cost_calculation(self) -> None:
        # openai/gpt-oss-120b: input $0.15/1M, output $0.60/1M
        cost = calculate_cost_usd("openai/gpt-oss-120b", input_tokens=95, output_tokens=40)
        assert cost == pytest.approx(0.00003825)

    def test_zero_token_calculation(self) -> None:
        assert calculate_cost_usd("gpt-6-luna", input_tokens=0, output_tokens=0) == 0.0

    def test_unknown_model_logs_warning_and_returns_zero(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.WARNING, logger="llm_observability"):
            cost = calculate_cost_usd("nonexistent-model-x", input_tokens=100, output_tokens=50)

        assert cost == 0.0
        assert any(
            "Pricing not configured for model=nonexistent-model-x" in record.message
            for record in caplog.records
        )


class TestStructuredLLMLogging:
    def test_successful_llm_call_emits_structured_json_log(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="llm_observability"):
            cost = log_llm_call(
                provider="openai",
                model="gpt-6-luna",
                latency_ms=123.456,
                input_tokens=100,
                output_tokens=200,
            )

        assert cost == pytest.approx(0.00011)
        llm_records = [r for r in caplog.records if r.message.startswith("LLM_CALL ")]
        assert len(llm_records) == 1
        payload = json.loads(llm_records[0].message.removeprefix("LLM_CALL "))
        assert payload["provider"] == "openai"
        assert payload["model"] == "gpt-6-luna"
        assert payload["latency_ms"] == 123.46
        assert payload["input_tokens"] == 100
        assert payload["output_tokens"] == 200
        assert payload["estimated_cost_usd"] == pytest.approx(0.00011)
        assert payload["status"] == "success"
        assert payload["error"] is None

    def test_failed_llm_call_logs_error_status(
        self,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        with caplog.at_level(logging.INFO, logger="llm_observability"):
            cost = log_llm_call(
                provider="anthropic",
                model="claude-haiku-5-5",
                latency_ms=55.1,
                status="error",
                error="APITimeoutError",
            )

        assert cost == 0.0
        llm_records = [r for r in caplog.records if r.message.startswith("LLM_CALL ")]
        assert len(llm_records) == 1
        payload = json.loads(llm_records[0].message.removeprefix("LLM_CALL "))
        assert payload["status"] == "error"
        assert payload["error"] == "APITimeoutError"


class TestLoggingSetupAndFormatting:
    def test_colored_formatter_adds_ansi_codes_without_mutating_original_record(self) -> None:
        formatter = ColoredFormatter("%(levelname)s | %(message)s")
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname=__file__,
            lineno=1,
            msg="hello",
            args=(),
            exc_info=None,
        )
        formatted = formatter.format(record)
        assert "\033[32mINFO\033[0m" in formatted
        assert record.levelname == "INFO"

    def test_development_setup_creates_log_file_without_ansi_codes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.chdir(tmp_path)
        monkeypatch.setenv("APP_ENV", "development")

        root_logger = logging.getLogger()
        original_handlers = list(root_logger.handlers)
        original_level = root_logger.level
        original_configured = getattr(root_logger, "_app_logging_configured", False)

        try:
            root_logger.handlers.clear()
            setattr(root_logger, "_app_logging_configured", False)

            log_file = setup_logging()
            assert log_file is not None
            assert log_file == Path("logs") / "app.log"

            logging.getLogger("test.file.check").info("clean file log check")
            for handler in root_logger.handlers:
                handler.flush()

            content = (tmp_path / log_file).read_text(encoding="utf-8")
            assert "clean file log check" in content
            assert "\033[" not in content
        finally:
            for handler in list(root_logger.handlers):
                if handler not in original_handlers:
                    handler.close()
                    root_logger.removeHandler(handler)
            root_logger.handlers = original_handlers
            root_logger.setLevel(original_level)
            setattr(root_logger, "_app_logging_configured", original_configured)
