"""User profile endpoints — GET /users/me, PUT /users/me."""

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...db.base import get_db
from .auth import get_current_user
from ...db.models import User

router = APIRouter(prefix="/users", tags=["users"])


class UserProfileResponse(BaseModel):
    name: str
    email: str
    role: str
    organization: Optional[str] = None
    avatar_url: Optional[str] = None


class UpdateProfileRequest(BaseModel):
    name: Optional[str] = None
    avatar_url: Optional[str] = None


@router.get("/me", response_model=UserProfileResponse)
def get_user_profile(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = current_user.profile
    tenant_profile = current_user.tenant.profile if current_user.tenant else None

    first = (profile.first_name or "") if profile else ""
    last = (profile.last_name or "") if profile else ""
    name = f"{first} {last}".strip() or current_user.email

    return UserProfileResponse(
        name=name,
        email=current_user.email,
        role=current_user.role,
        organization=tenant_profile.legal_name if tenant_profile else None,
        avatar_url=profile.profile_image_path if profile else None,
    )


@router.put("/me", response_model=UserProfileResponse)
def update_user_profile(
    request: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    profile = current_user.profile
    if not profile:
        raise HTTPException(status_code=404, detail="User profile not found")

    if request.name is not None:
        parts = request.name.strip().split(" ", 1)
        profile.first_name = parts[0]
        profile.last_name = parts[1] if len(parts) > 1 else ""
        profile.display_name = request.name.strip()

    if request.avatar_url is not None:
        profile.profile_image_path = request.avatar_url

    db.commit()

    tenant_profile = current_user.tenant.profile if current_user.tenant else None
    first = profile.first_name or ""
    last = profile.last_name or ""
    name = f"{first} {last}".strip() or current_user.email

    return UserProfileResponse(
        name=name,
        email=current_user.email,
        role=current_user.role,
        organization=tenant_profile.legal_name if tenant_profile else None,
        avatar_url=profile.profile_image_path,
    )
