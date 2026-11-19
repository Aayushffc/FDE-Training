from typing import Any, List, Optional
from sqlalchemy import select
from sqlalchemy.orm import Session, joinedload

from app.models.prompt import PromptCategory, PromptVersion
from app.schemas.prompt import (
    PromptCategoryCreate,
    PromptVersionCreate,
)


class DBPromptService:
    """Service for managing versioned prompt templates stored in the database."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def get_or_create_category(self, name: str, description: Optional[str] = None) -> PromptCategory:
        """Find an existing category or create a new one."""
        category = self.db.scalar(
            select(PromptCategory).where(PromptCategory.name == name)
        )
        if not category:
            category = PromptCategory(name=name, description=description)
            self.db.add(category)
            self.db.flush()
        return category

    def list_categories(self) -> List[PromptCategory]:
        """List all prompt categories."""
        stmt = select(PromptCategory).options(joinedload(PromptCategory.prompt_versions)).order_by(PromptCategory.id)
        return list(self.db.scalars(stmt).unique().all())

    def get_category_by_name(self, name: str) -> Optional[PromptCategory]:
        """Retrieve a single category by name."""
        stmt = (
            select(PromptCategory)
            .options(joinedload(PromptCategory.prompt_versions))
            .where(PromptCategory.name == name)
        )
        return self.db.scalar(stmt)

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
        """Create or update a versioned prompt in the specified category."""
        category = self.get_or_create_category(name=category_name)

        existing = self.db.scalar(
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
            self.db.flush()
            return existing

        new_prompt = PromptVersion(
            category_id=category.id,
            version=version,
            name=name,
            prompt_text=prompt_text,
            description=description,
            variables=variables,
            is_active=is_active,
        )
        self.db.add(new_prompt)
        self.db.flush()
        return new_prompt

    def get_prompt_version(self, category_name: str, version: str) -> Optional[PromptVersion]:
        """Retrieve a prompt version by its category name and version identifier."""
        stmt = (
            select(PromptVersion)
            .join(PromptCategory)
            .where(
                PromptCategory.name == category_name,
                PromptVersion.version == version,
            )
        )
        return self.db.scalar(stmt)

    def list_versions_by_category(self, category_name: str) -> List[PromptVersion]:
        """List all prompt versions belonging to a given category."""
        stmt = (
            select(PromptVersion)
            .join(PromptCategory)
            .where(PromptCategory.name == category_name)
            .order_by(PromptVersion.id)
        )
        return list(self.db.scalars(stmt).all())
