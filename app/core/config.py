import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    anthropic_api_key: str | None
    groq_api_key: str | None
    db_url: str | None = None
    default_openai_model: str = "gpt-6-luna"
    default_anthropic_model: str = "claude-haiku-5-5"
    default_groq_model: str = "llama-3.3-70b-versatile"
    request_timeout_seconds: float = 30.0


def load_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY", "").strip() or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", "").strip() or None,
        groq_api_key=os.getenv("GROQ_API_KEY", "").strip() or None,
        db_url=os.getenv("DB_URL", "").strip() or os.getenv("DATABASE_URL", "").strip() or None,
        default_openai_model=os.getenv("OPENAI_MODEL", "").strip() or "gpt-6-luna",
        default_anthropic_model=os.getenv("ANTHROPIC_MODEL", "").strip() or "claude-haiku-5-5",
        default_groq_model=os.getenv("GROQ_MODEL", "").strip() or "llama-3.3-70b-versatile",
        request_timeout_seconds=float(os.getenv("REQUEST_TIMEOUT", "30")),
    )


settings = load_settings()
