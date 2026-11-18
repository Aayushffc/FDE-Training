from pydantic import BaseModel, Field


class PromptTemplate(BaseModel):
    name: str = Field(..., description="Unique template identifier")
    template: str = Field(..., description="Raw prompt template text with optional format variables")
    description: str | None = Field(default=None, description="Explanation of prompt intent")
