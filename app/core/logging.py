from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING

from app.config import MODEL_PRICING_PER_1M

if TYPE_CHECKING:
    from fastapi import FastAPI, Request, Response

# ANSI colors for terminal output
COLORS = {
    "DEBUG": "\033[36m",    # Cyan
    "INFO": "\033[32m",     # Green
    "WARNING": "\033[33m",  # Yellow
    "ERROR": "\033[31m",    # Red
    "CRITICAL": "\033[35m", # Magenta
}
RESET = "\033[0m"


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = COLORS.get(record.levelname, "")
        record_copy = logging.makeLogRecord(record.__dict__.copy())
        record_copy.levelname = f"{color}{record.levelname}{RESET}"

        return super().format(record_copy)


def setup_logging() -> Path | None:
    """
    Configure application logging.

    Development:
      - Colored console logs
      - One log file per application process/session

    Production:
      - Console logs only (normally collected by the hosting platform)
    """

    is_development = os.getenv("APP_ENV", "development") == "development"

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if is_development else logging.INFO)

    # Avoid adding duplicate handlers if setup is called more than once.
    if getattr(root_logger, "_app_logging_configured", False):
        return None

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if is_development else logging.INFO)

    log_format = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"

    console_handler.setFormatter(
        ColoredFormatter(log_format, datefmt=date_format)
    )
    root_logger.addHandler(console_handler)

    log_file = None

    if is_development:
        log_directory = Path("logs")
        log_directory.mkdir(parents=True, exist_ok=True)

        session_id = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_file = log_directory / f"dev-{session_id}-{os.getpid()}.log"

        file_handler = logging.FileHandler(
            log_file,
            encoding="utf-8",
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(
            logging.Formatter(log_format, datefmt=date_format)
        )

        root_logger.addHandler(file_handler)

    root_logger._app_logging_configured = True

    logging.captureWarnings(True)

    logging.getLogger(__name__).info(
        "Logging initialized | environment=%s | file=%s",
        "development" if is_development else "production",
        log_file or "disabled",
    )

    return log_file


def calculate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    rates = MODEL_PRICING_PER_1M.get(model)

    if rates is None:
        logging.getLogger("llm_observability").warning(
            "Pricing not configured for model=%s; cost cannot be estimated",
            model,
        )
        return 0.0

    cost = (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
    ) / 1_000_000

    return round(cost, 8)


def log_llm_call(
    *,
    provider: str,
    model: str,
    latency_ms: float,
    input_tokens: int = 0,
    output_tokens: int = 0,
    status: str = "success",
    error: str | None = None,
) -> float:

    cost = calculate_cost_usd(
        model,
        input_tokens,
        output_tokens,
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "provider": provider,
        "model": model,
        "latency_ms": round(latency_ms, 2),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "estimated_cost_usd": cost,
        "status": status,
        "error": error,
    }

    logger = logging.getLogger("llm_observability")
    logger.info("LLM_CALL %s", json.dumps(payload))

    return cost


def register_request_logging_middleware(app: FastAPI) -> None:
    access_logger = logging.getLogger("api.access")

    @app.middleware("http")
    async def log_requests(request: Request, call_next) -> Response:
        started_at = perf_counter()
        try:
            response = await call_next(request)
        except Exception:
            duration_ms = (perf_counter() - started_at) * 1000
            access_logger.error(
                "%s %s -> 500 (%.2f ms)",
                request.method,
                request.url.path,
                duration_ms,
            )
            raise

        duration_ms = (perf_counter() - started_at) * 1000
        if response.status_code >= 500:
            log_fn = access_logger.error
        elif response.status_code >= 400:
            log_fn = access_logger.warning
        else:
            log_fn = access_logger.info

        log_fn(
            "%s %s -> %s (%.2f ms)",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
        )
        return response
