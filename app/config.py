import os
from dataclasses import dataclass

# Cost per 1,000,000 tokens in USD (update if you test different models)
MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "gpt-6-luna": {"input": 0.10, "output": 0.50},
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "claude-haiku-5-5": {"input": 0.10, "output": 0.50},
    "claude-haiku-4-5": {"input": 1.00, "output": 5.00},
}


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    anthropic_api_key: str | None
    default_openai_model: str = "gpt-6-luna"
    default_anthropic_model: str = "claude-haiku-5-5"
    request_timeout_seconds: float = 30.0


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip() or None,
        default_openai_model=os.getenv("OPENAI_MODEL", "").strip() or "gpt-6-luna",
        default_anthropic_model=os.getenv("ANTHROPIC_MODEL", "").strip() or "claude-haiku-5-5",
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT", "30")),
    )


settings = load_settings()