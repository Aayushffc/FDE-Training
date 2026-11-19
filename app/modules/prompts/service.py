from collections.abc import Callable
from typing import Any, Dict, List, Optional
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal
from app.models.prompt import PromptCategory, PromptVersion


class PromptService:
    """
    Manages versioned prompt templates stored in the database.
    Queries PostgreSQL (or an active DB session) for categories and versions,
    with in-memory caching to avoid redundant queries during LLM runs.
    """

    def __init__(
        self,
        db: Optional[Session] = None,
        session_factory: Optional[Callable[[], Session]] = None,
        auto_seed_if_empty: bool = True,
    ) -> None:
        self._db = db
        self._session_factory = session_factory or SessionLocal
        self._cache: Dict[str, str] = {}
        self._auto_seed_if_empty = auto_seed_if_empty

    def _get_session(self) -> Session:
        if self._db is not None:
            return self._db
        if self._session_factory is not None:
            return self._session_factory()
        raise RuntimeError("No database session or sessionmaker configured for PromptService.")

    def _ensure_populated(self, session: Session) -> None:
        """If the database has zero prompts (e.g. fresh in-memory test DB), seed standard prompts."""
        if not self._auto_seed_if_empty:
            return
        try:
            count = session.scalar(select(func.count(PromptVersion.id)))
            if count == 0:
                from scripts.seed_prompts import seed_database
                seed_database(session)
        except Exception:
            pass

    def load_prompt(self, category: str, version: str) -> str:
        """
        Dynamically load a specific versioned prompt from the database.
        Example: load_prompt("customer_support", "v1") -> queries DB for category='customer_support' and version='v1'
        """
        cache_key = f"{category}/{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        session = self._get_session()
        owns_session = self._db is None
        try:
            self._ensure_populated(session)
            stmt = (
                select(PromptVersion.prompt_text)
                .join(PromptCategory)
                .where(
                    PromptCategory.name == category,
                    PromptVersion.version == version,
                    PromptVersion.is_active == True,
                )
            )
            prompt_text = session.scalar(stmt)
            if prompt_text is None:
                raise FileNotFoundError(
                    f"Versioned prompt '{version}' for category '{category}' not found in database."
                )

            content = prompt_text.strip()
            self._cache[cache_key] = content
            return content
        finally:
            if owns_session:
                session.close()

    def list_versions(self, category: str) -> List[str]:
        """List all active prompt versions for a given category from the database."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            self._ensure_populated(session)
            stmt = (
                select(PromptVersion.version)
                .join(PromptCategory)
                .where(
                    PromptCategory.name == category,
                    PromptVersion.is_active == True,
                )
                .order_by(PromptVersion.version)
            )
            return list(session.scalars(stmt).all())
        finally:
            if owns_session:
                session.close()

    def list_categories(self) -> List[str]:
        """List all prompt category names from the database."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            self._ensure_populated(session)
            stmt = select(PromptCategory.name).order_by(PromptCategory.name)
            return list(session.scalars(stmt).all())
        finally:
            if owns_session:
                session.close()

    def list_all(self) -> Dict[str, List[str]]:
        """List all categories with their active versions from the database."""
        result: Dict[str, List[str]] = {}
        for cat in self.list_categories():
            versions = self.list_versions(cat)
            if versions:
                result[cat] = versions
        return result

    def get_system_prompt(self, template_name: str = "default") -> str:
        """
        Load a system prompt template by name or path from the database.
        - "category/version" -> load_prompt(category, version)
        - "default" -> load_prompt("general", "default")
        - other name -> searches "general" category first, then any category matching version
        """
        if "/" in template_name:
            cat, ver = template_name.split("/", 1)
            return self.load_prompt(cat, ver)

        if template_name in self._cache:
            return self._cache[template_name]

        session = self._get_session()
        owns_session = self._db is None
        try:
            self._ensure_populated(session)
            # First search in 'general' category
            stmt = (
                select(PromptVersion.prompt_text)
                .join(PromptCategory)
                .where(
                    PromptCategory.name == "general",
                    PromptVersion.version == template_name,
                    PromptVersion.is_active == True,
                )
            )
            prompt_text = session.scalar(stmt)

            # If not in general, check any category where version == template_name
            if prompt_text is None:
                stmt_any = (
                    select(PromptVersion.prompt_text)
                    .where(
                        PromptVersion.version == template_name,
                        PromptVersion.is_active == True,
                    )
                )
                prompt_text = session.scalar(stmt_any)

            if prompt_text is None:
                raise FileNotFoundError(
                    f"Prompt template '{template_name}' not found in database."
                )

            content = prompt_text.strip()
            self._cache[template_name] = content
            return content
        finally:
            if owns_session:
                session.close()

    def format_prompt(self, template_name: str, **kwargs: Any) -> str:
        template = self.get_system_prompt(template_name)
        return template.format(**kwargs)

    def clear_cache(self) -> None:
        """Clear cached prompt templates."""
        self._cache.clear()


default_prompt_service = PromptService()


def _get_system_prompt() -> str:
    try:
        return default_prompt_service.get_system_prompt()
    except Exception:
        return (
            "You are a concise, accurate, and reliable AI assistant.\n"
            "Guidelines:\n"
            "- Answer the user's question directly and clearly without unnecessary filler.\n"
            "- Be strictly truthful: if you are unsure or lack sufficient context, say so explicitly rather than guessing or fabricating details.\n"
            "- Use clean formatting (lists, code blocks, or tables) when it improves readability.\n"
            "- Match the depth of your response to the complexity of the request."
        )


SYSTEM_PROMPT = _get_system_prompt()
