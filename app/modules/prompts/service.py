"""
Prompt module service interface.
Directly re-exports the unified PromptService from app.services.prompt_service.
All prompt data is queried directly from PostgreSQL.
"""

from app.services.prompt_service import (
    DBPromptService,
    PromptService,
    SYSTEM_PROMPT,
    default_prompt_service,
)

__all__ = [
    "DBPromptService",
    "PromptService",
    "SYSTEM_PROMPT",
    "default_prompt_service",
]
