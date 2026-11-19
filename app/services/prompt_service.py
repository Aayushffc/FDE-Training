from collections.abc import Callable
from typing import Any, Dict, List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.db.session import SessionLocal
from app.models.prompt import PromptCategory, PromptVersion


class PromptService:
    """
    Manages versioned prompt templates stored in PostgreSQL.
    Directly queries the database for prompt categories and versions,
    with an in-memory cache to optimize performance during LLM calls.
    """

    def __init__(
        self,
        db: Optional[Session] = None,
        session_factory: Optional[Callable[[], Session]] = None,
    ) -> None:
        self._db = db
        self._session_factory = session_factory or SessionLocal
        self._cache: Dict[str, str] = {}

    def _get_session(self) -> Session:
        if self._db is not None:
            return self._db
        if self._session_factory is not None:
            return self._session_factory()
        raise RuntimeError("No database session configured for PromptService.")

    def load_prompt(self, category: str, version: str) -> str:
        """
        Dynamically load a specific versioned prompt from the database.
        Raises FileNotFoundError if not found in the database.
        """
        cache_key = f"{category}/{version}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        session = self._get_session()
        owns_session = self._db is None
        try:
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

    def get_system_prompt(self, template_name: str = "default") -> str:
        """
        Load a system prompt template by name or path directly from the database.
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

    def list_versions(self, category: str) -> List[str]:
        """List all active prompt versions for a given category from the database."""
        session = self._get_session()
        owns_session = self._db is None
        try:
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

    def get_prompt_version(self, category_name: str, version: str) -> Optional[PromptVersion]:
        """Retrieve a PromptVersion model instance by its category name and version."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            stmt = (
                select(PromptVersion)
                .join(PromptCategory)
                .options(joinedload(PromptVersion.category))
                .where(
                    PromptCategory.name == category_name,
                    PromptVersion.version == version,
                )
            )
            return session.scalar(stmt)
        finally:
            if owns_session:
                session.close()

    def get_prompt_version_by_id(self, version_id: int) -> Optional[PromptVersion]:
        """Retrieve a PromptVersion model instance by its primary key ID."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            stmt = (
                select(PromptVersion)
                .options(joinedload(PromptVersion.category))
                .where(PromptVersion.id == version_id)
            )
            return session.scalar(stmt)
        finally:
            if owns_session:
                session.close()

    def get_prompt_versions_by_category(self, category_name: str) -> List[PromptVersion]:
        """Retrieve all active or inactive PromptVersion instances for a specific category."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            stmt = (
                select(PromptVersion)
                .join(PromptCategory)
                .options(joinedload(PromptVersion.category))
                .where(PromptCategory.name == category_name)
                .order_by(PromptVersion.id)
            )
            return list(session.scalars(stmt).all())
        finally:
            if owns_session:
                session.close()

    def get_categories(self) -> List[PromptCategory]:
        """Retrieve all PromptCategory instances from the prompt_categories table."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            stmt = select(PromptCategory).order_by(PromptCategory.id)
            return list(session.scalars(stmt).all())
        finally:
            if owns_session:
                session.close()

    def get_category_by_name(self, name: str) -> Optional[PromptCategory]:
        """Retrieve a single PromptCategory by name along with its versions."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            stmt = (
                select(PromptCategory)
                .options(joinedload(PromptCategory.prompt_versions))
                .where(PromptCategory.name == name)
            )
            return session.scalar(stmt)
        finally:
            if owns_session:
                session.close()

    def create_category(self, name: str, description: Optional[str] = None) -> PromptCategory:
        """Create a new PromptCategory record in PostgreSQL."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            category = PromptCategory(name=name, description=description)
            session.add(category)
            session.flush()
            if owns_session:
                session.commit()
                session.refresh(category)
            return category
        finally:
            if owns_session:
                session.close()

    def delete_prompt_version(self, category_name: str, version: str) -> bool:
        """Delete a prompt version from the database and invalidate caches."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            stmt = (
                select(PromptVersion)
                .join(PromptCategory)
                .where(
                    PromptCategory.name == category_name,
                    PromptVersion.version == version,
                )
            )
            pv = session.scalar(stmt)
            if not pv:
                return False
            session.delete(pv)
            session.flush()
            self._cache.pop(f"{category_name}/{version}", None)
            if owns_session:
                session.commit()
            return True
        finally:
            if owns_session:
                session.close()

    def get_all_categories_with_versions(self) -> List[PromptCategory]:
        """Retrieve all PromptCategory instances along with their versions."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            stmt = (
                select(PromptCategory)
                .options(joinedload(PromptCategory.prompt_versions))
                .order_by(PromptCategory.id)
            )
            return list(session.scalars(stmt).unique().all())
        finally:
            if owns_session:
                session.close()

    def get_or_create_category(self, name: str, description: Optional[str] = None) -> PromptCategory:
        """Find an existing category or create a new one."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            category = session.scalar(
                select(PromptCategory).where(PromptCategory.name == name)
            )
            if not category:
                category = PromptCategory(name=name, description=description)
                session.add(category)
                session.flush()
                if owns_session:
                    session.commit()
                    session.refresh(category)
            return category
        finally:
            if owns_session:
                session.close()

    def create_prompt_version(
        self,
        category_name: str,
        version: str,
        prompt_text: str,
        name: Optional[str] = None,
        description: Optional[str] = None,
        variables: Optional[Any] = None,
        is_active: bool = True,
    ) -> PromptVersion:
        """Create or update a versioned prompt in the specified category in the database."""
        session = self._get_session()
        owns_session = self._db is None
        try:
            category = self.get_or_create_category(name=category_name)

            existing = session.scalar(
                select(PromptVersion).where(
                    PromptVersion.category_id == category.id,
                    PromptVersion.version == version,
                )
            )
            if existing:
                existing.prompt_text = prompt_text
                existing.name = name or existing.name
                existing.description = description or existing.description
                existing.variables = variables or existing.variables
                existing.is_active = is_active
                session.flush()
                result = existing
            else:
                new_prompt = PromptVersion(
                    category_id=category.id,
                    version=version,
                    name=name,
                    prompt_text=prompt_text,
                    description=description,
                    variables=variables,
                    is_active=is_active,
                )
                session.add(new_prompt)
                session.flush()
                result = new_prompt

            # Invalidate in-memory cache
            self._cache.pop(f"{category_name}/{version}", None)
            if version == "default" and category_name == "general":
                self._cache.pop("default", None)

            if owns_session:
                session.commit()
                session.refresh(result)

            return result
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


def _resolve_default_system_prompt() -> str:
    try:
        return default_prompt_service.get_system_prompt("default")
    except Exception:
        return ""


SYSTEM_PROMPT = _resolve_default_system_prompt()

# Alias for backwards-compatibility
DBPromptService = PromptService
