from typing import List

from fastapi import APIRouter, Depends, HTTPException, Path, Response, status
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.schemas.prompt import (
    PromptCategoryCreate,
    PromptCategoryRead,
    PromptCategoryWithVersions,
    PromptVersionCreate,
    PromptVersionRead,
)
from app.services.prompt_service import PromptService

router = APIRouter(prefix="/prompts", tags=["Prompts"])


# ============================================================================
# Core Prompts (Categories with Versions)
# ============================================================================

@router.get(
    "",
    response_model=List[PromptCategoryWithVersions],
    summary="List all prompt categories and versions stored in PostgreSQL",
)
def list_prompts(
    db: Session = Depends(get_db),
) -> List[PromptCategoryWithVersions]:
    """Retrieve all prompt categories along with their versions directly from the database."""
    service = PromptService(db)
    return service.get_all_categories_with_versions()


@router.post(
    "",
    response_model=PromptVersionRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create or update a versioned prompt in PostgreSQL",
)
def create_or_update_prompt(
    payload: PromptVersionCreate,
    db: Session = Depends(get_db),
) -> PromptVersionRead:
    """Insert or update a versioned prompt template under the specified category in PostgreSQL."""
    service = PromptService(db)
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


# ============================================================================
# Prompt Categories Table Endpoints
# ============================================================================

@router.get(
    "/categories",
    response_model=List[PromptCategoryRead],
    summary="List all prompt categories stored in PostgreSQL",
)
def list_categories(
    db: Session = Depends(get_db),
) -> List[PromptCategoryRead]:
    """Retrieve all category records directly from the prompt_categories table."""
    service = PromptService(db)
    return service.get_categories()


@router.post(
    "/categories",
    response_model=PromptCategoryRead,
    status_code=status.HTTP_201_CREATED,
    summary="Create a new prompt category in PostgreSQL",
)
def create_category(
    payload: PromptCategoryCreate,
    db: Session = Depends(get_db),
) -> PromptCategoryRead:
    """Create a new category in the prompt_categories table."""
    service = PromptService(db)
    existing = service.get_category_by_name(payload.name)
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Category '{payload.name}' already exists in the database.",
        )
    category = service.create_category(name=payload.name, description=payload.description)
    db.commit()
    db.refresh(category)
    return category


@router.get(
    "/categories/{category_name}",
    response_model=PromptCategoryWithVersions,
    summary="Get prompt category details and its versions from PostgreSQL",
)
def get_category_by_name(
    category_name: str = Path(..., description="Unique category namespace, e.g. 'customer_support'"),
    db: Session = Depends(get_db),
) -> PromptCategoryWithVersions:
    """Retrieve a single prompt category and all associated prompt versions by name."""
    service = PromptService(db)
    category = service.get_category_by_name(category_name)
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category '{category_name}' not found in the database.",
        )
    return category


@router.get(
    "/categories/{category_name}/versions",
    response_model=List[PromptVersionRead],
    summary="List all prompt versions for a given category from PostgreSQL",
)
def list_category_versions(
    category_name: str = Path(..., description="Prompt category namespace, e.g. 'customer_support'"),
    db: Session = Depends(get_db),
) -> List[PromptVersionRead]:
    """Retrieve all prompt versions associated with the given category."""
    service = PromptService(db)
    category = service.get_category_by_name(category_name)
    if not category:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Category '{category_name}' not found in the database.",
        )
    return service.get_prompt_versions_by_category(category_name)


# ============================================================================
# Prompt Versions Table Endpoints
# ============================================================================

@router.get(
    "/versions/{version_id}",
    response_model=PromptVersionRead,
    summary="Get prompt version details by version ID from PostgreSQL",
)
def get_version_by_id(
    version_id: int = Path(..., description="Prompt version primary key ID"),
    db: Session = Depends(get_db),
) -> PromptVersionRead:
    """Retrieve a specific prompt version record by its primary key ID from prompt_versions."""
    service = PromptService(db)
    prompt = service.get_prompt_version_by_id(version_id)
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Prompt version with ID '{version_id}' was not found in the database.",
        )
    return prompt


# ============================================================================
# Category & Version Detail Endpoints
# ============================================================================

@router.get(
    "/{category}/{version}",
    response_model=PromptVersionRead,
    summary="Get prompt version details from PostgreSQL",
)
def get_prompt_version(
    category: str = Path(..., description="Prompt category namespace (e.g. 'customer_support')"),
    version: str = Path(..., description="Prompt version identifier (e.g. 'v1', 'v2')"),
    db: Session = Depends(get_db),
) -> PromptVersionRead:
    """Retrieve complete prompt text, metadata, and variables for a specific version in PostgreSQL."""
    service = PromptService(db)
    prompt = service.get_prompt_version(category, version)
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "prompt_not_found",
                "detail": f"Prompt version '{version}' for category '{category}' was not found in the database.",
            },
        )
    return prompt


@router.delete(
    "/{category}/{version}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a prompt version from PostgreSQL",
)
def delete_prompt_version(
    category: str = Path(..., description="Prompt category namespace (e.g. 'customer_support')"),
    version: str = Path(..., description="Prompt version identifier (e.g. 'v1', 'v2')"),
    db: Session = Depends(get_db),
) -> Response:
    """Delete a versioned prompt record from PostgreSQL."""
    service = PromptService(db)
    deleted = service.delete_prompt_version(category, version)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "prompt_not_found",
                "detail": f"Prompt version '{version}' for category '{category}' not found in the database.",
            },
        )
    db.commit()
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/{version}",
    response_model=PromptVersionRead,
    summary="Get prompt version details (shortcut defaulting to customer_support category)",
)
def get_default_category_prompt_version(
    version: str = Path(..., description="Prompt version identifier (e.g. 'v1', 'v2')"),
    db: Session = Depends(get_db),
) -> PromptVersionRead:
    """Convenience shortcut to fetch a prompt version from the default 'customer_support' category."""
    service = PromptService(db)
    prompt = service.get_prompt_version("customer_support", version)
    if not prompt:
        # Fallback search in 'general' category
        prompt = service.get_prompt_version("general", version)
    if not prompt:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "prompt_not_found",
                "detail": f"Prompt version '{version}' was not found in the default categories in PostgreSQL.",
            },
        )
    return prompt
