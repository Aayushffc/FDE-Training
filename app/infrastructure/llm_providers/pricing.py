import logging

logger = logging.getLogger("llm_observability")

# Cost per 1,000,000 tokens in USD across supported providers and models
MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    # OpenAI Models
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    # Anthropic Models
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
    # Groq Models
    "llama-3.3-70b-versatile": {"input": 0.59, "output": 0.79},
    "llama-3.1-8b-instant": {"input": 0.05, "output": 0.08},
    "openai/gpt-oss-120b": {"input": 0.15, "output": 0.60},
    "openai/gpt-oss-20b": {"input": 0.075, "output": 0.30},
}


def calculate_cost_usd(
    model: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Calculate the estimated USD cost for a given model and token counts."""
    rates = MODEL_PRICING_PER_1M.get(model)

    if rates is None:
        logger.warning(
            "Pricing not configured for model=%s; cost cannot be estimated",
            model,
        )
        return 0.0

    cost = (
        input_tokens * rates["input"]
        + output_tokens * rates["output"]
    ) / 1_000_000

    return round(cost, 8)
