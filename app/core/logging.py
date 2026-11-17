import json
import logging
from datetime import datetime, timezone
from app.config import MODEL_PRICING_PER_1M

logger = logging.getLogger("llm_observability")
logger.setLevel(logging.INFO)

if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)


def calculate_cost_usd(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = MODEL_PRICING_PER_1M.get(model, {"input": 0.0, "output": 0.0})
    cost = (input_tokens * rates["input"] + output_tokens * rates["output"]) / 1_000_000
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
    cost = calculate_cost_usd(model, input_tokens, output_tokens)
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
    logger.info(json.dumps(payload))
    return cost