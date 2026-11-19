from typing import List

from fastapi import APIRouter, Depends, HTTPException, Path
from sqlalchemy.orm import Session

from app.api.v1.dependencies import get_chat_service
from app.db.session import get_db
from app.modules.chat.schemas import PromptTemplateDetailResponse
from app.modules.chat.service import ChatService
from app.schemas.prompt import (
    PromptCategoryWithVersions,
    PromptVersionCreate,
    PromptVersionRead,
)
from app.services.prompt_service import DBPromptService

router = APIRouter(prefix="/prompts", tags=["Prompts"])


@router.get("", summary="List available prompt categories and versions")
async def list_available_prompts(
    chat_service: ChatService = Depends(get_chat_service),
) -> dict:
    """
    List all prompt categories and versions available in the versioned prompt library.
    Use these versions when calling `POST /v1/chat/prompt-version`.
    """
    return {
        "default_category": "customer_support",
        "available_customer_support_versions": chat_service.prompt_service.list_versions("customer_support"),
        "all_categories": chat_service.prompt_service.list_all(),
        "usage_example": {
            "message": "My order #12345 arrived damaged, please refund me.",
            "provider": "groq",
            "prompt_version": "v2",
        },
    }


# Database-backed prompt endpoints
@router.get(
    "/db",
    response_model=List[PromptCategoryWithVersions],
    summary="List all prompt categories and versions stored in PostgreSQL",
)
def list_db_prompts(
    db: Session = Depends(get_db),
) -> List[PromptCategoryWithVersions]:
    """Retrieve all categories along with their prompt versions from the database."""
    service = DBPromptService(db)
    return service.list_categories()


@router.get(
    "/db/{category}/{version}",
    response_model=PromptVersionRead,
    summary="Get prompt version details from PostgreSQL",
)
def get_db_prompt(
    category: str = Path(..., description="Prompt category namespace (e.g. 'customer_support')"),
    version: str = Path(..., description="Prompt version (e.g. 'v1', 'v2')"),
    db: Session = Depends(get_db),
) -> PromptVersionRead:
    """Retrieve the complete prompt text and metadata for a specific version in PostgreSQL."""
    service = DBPromptService(db)
    prompt = service.get_prompt_version(category, version)
    if not prompt:
        raise HTTPException(
            status_code=404,
            detail=f"Prompt version '{version}' for category '{category}' was not found in the database.",
        )
    return prompt


@router.post(
    "/db",
    response_model=PromptVersionRead,
    status_code=201,
    summary="Create or update a versioned prompt in PostgreSQL",
)
def create_db_prompt(
    payload: PromptVersionCreate,
    db: Session = Depends(get_db),
) -> PromptVersionRead:
    """Insert or update a prompt version in the PostgreSQL database."""
    service = DBPromptService(db)
    prompt = service.create_prompt_version(
        category_name=payload.category_name,
        version=payload.version,
        prompt_text=payload.prompt_text,
        name=payload.name,
        description=payload.description,
        variables=payload.variables,
        is_active=payload.is_active,
    )
    db.commit()
    db.refresh(prompt)
    return prompt


@router.get(
    "/{category}/{version}",
    response_model=PromptTemplateDetailResponse,
    summary="Get the raw written content of a versioned prompt template",
)
async def get_prompt_content(
    category: str = Path(..., description="Prompt category namespace, e.g. 'customer_support'", examples=["customer_support"]),
    version: str = Path(..., description="Prompt version identifier, e.g. 'v1', 'v2', 'v3'", examples=["v2"]),
    chat_service: ChatService = Depends(get_chat_service),
) -> PromptTemplateDetailResponse:
    """
    Retrieve the exact written text and instructions stored inside a specific prompt template file.
    """
    try:
        content = chat_service.prompt_service.load_prompt(category, version)
        return PromptTemplateDetailResponse(
            category=category,
            version=version,
            file_name=f"{version}.txt",
            content=content,
        )
    except FileNotFoundError:
        available = chat_service.prompt_service.list_versions(category)
        raise HTTPException(
            status_code=404,
            detail={
                "error": "prompt_not_found",
                "detail": f"Prompt template '{version}' not found in category '{category}'.",
                "available_versions_in_category": available,
            },
        )


@router.get(
    "/{version}",
    response_model=PromptTemplateDetailResponse,
    summary="Get the raw written content of a prompt template (defaulting to customer_support category)",
)
async def get_default_category_prompt_content(
    version: str = Path(..., description="Prompt version, e.g. 'v1', 'v2', 'v3'", examples=["v2"]),
    chat_service: ChatService = Depends(get_chat_service),
) -> PromptTemplateDetailResponse:
    """
    Convenience shortcut to fetch prompt template text from the default 'customer_support' category.
    """
    return await get_prompt_content(
        category="customer_support",
        version=version,
        chat_service=chat_service,
    )
