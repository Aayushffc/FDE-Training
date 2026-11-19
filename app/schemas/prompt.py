from datetime import datetime
from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, computed_field


class PromptCategoryBase(BaseModel):
    name: str = Field(..., description="Unique category namespace (e.g. 'customer_support')", min_length=1, max_length=100)
    description: Optional[str] = Field(default=None, description="Description of the category domain")


class PromptCategoryCreate(PromptCategoryBase):
    pass


class PromptCategoryRead(PromptCategoryBase):
    id: int = Field(..., description="Category primary key ID")
    created_at: datetime
    updated_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PromptVersionBase(BaseModel):
    version: str = Field(..., description="Prompt version identifier (e.g. 'v1', 'v2')", min_length=1, max_length=50)
    name: Optional[str] = Field(default=None, description="Human-readable title or label", max_length=150)
    prompt_text: str = Field(..., description="The complete written prompt template text")
    description: Optional[str] = Field(default=None, description="Explanation of prompt intent or changes")
    variables: Optional[Any] = Field(default=None, description="List or map of format variables")
    is_active: bool = Field(default=True, description="Whether this prompt version is active")


class PromptVersionCreate(PromptVersionBase):
    category_name: str = Field(..., description="Name of the category to assign this version to")


class PromptVersionRead(PromptVersionBase):
    id: int = Field(..., description="Prompt version primary key ID")
    category_id: int = Field(..., description="Associated category ID")
    category_name: Optional[str] = Field(default=None, description="Associated category name")
    content: Optional[str] = Field(default=None, description="Prompt template text alias")
    file_name: Optional[str] = Field(default=None, description="Virtual file identifier alias")
    created_at: datetime
    updated_at: datetime

    @computed_field
    def category(self) -> Optional[str]:
        """Convenience alias for category_name."""
        return self.category_name

    model_config = ConfigDict(from_attributes=True)


class PromptCategoryWithVersions(PromptCategoryRead):
    prompt_versions: List[PromptVersionRead] = Field(default_factory=list, description="All versions under this category")

    model_config = ConfigDict(from_attributes=True)
