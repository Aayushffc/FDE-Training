from datetime import datetime
from typing import Any, List, Optional

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, JSON, String, Text, UniqueConstraint, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base


class PromptCategory(Base):
    """
    Groups versioned prompt templates into functional domains or namespaces
    (e.g., 'customer_support', 'code_reviewer', 'summarizer').
    """

    __tablename__ = "prompt_categories"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    name: Mapped[str] = mapped_column(String(100), unique=True, index=True, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    prompt_versions: Mapped[List["PromptVersion"]] = relationship(
        "PromptVersion",
        back_populates="category",
        cascade="all, delete-orphan",
        order_by="PromptVersion.id",
    )

    def __repr__(self) -> str:
        return f"<PromptCategory(id={self.id}, name='{self.name}')>"


class PromptVersion(Base):
    """
    Stores an immutable or versioned prompt template with its full prompt text,
    metadata, and format variables.
    """

    __tablename__ = "prompt_versions"
    __table_args__ = (
        UniqueConstraint("category_id", "version", name="uq_prompt_category_version"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    category_id: Mapped[int] = mapped_column(
        Integer,
        ForeignKey("prompt_categories.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    version: Mapped[str] = mapped_column(String(50), nullable=False, index=True)
    name: Mapped[Optional[str]] = mapped_column(String(150), nullable=True)
    prompt_text: Mapped[str] = mapped_column(Text, nullable=False)
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    variables: Mapped[Optional[Any]] = mapped_column(JSON, nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    # Relationships
    category: Mapped["PromptCategory"] = relationship(
        "PromptCategory",
        back_populates="prompt_versions",
    )

    @property
    def category_name(self) -> Optional[str]:
        return self.category.name if self.category else None

    @property
    def content(self) -> str:
        return self.prompt_text

    @property
    def file_name(self) -> str:
        return f"{self.version}.txt"

    def __repr__(self) -> str:
        return f"<PromptVersion(id={self.id}, category_id={self.category_id}, version='{self.version}')>"
