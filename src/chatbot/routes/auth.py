"""Authentication endpoints — register, login, me."""

import uuid
from datetime import datetime, timezone
from typing import Optional

import pyotp
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel, EmailStr
from sqlalchemy.orm import Session

from ...db.base import get_db
from ...db.models import User
from ...db import crud
from ..auth import (
    verify_password,
    create_access_token,
    decode_access_token
)
from ..analytics import build_sign_up_event, emit_billing_analytics_event
from ..billing import serialize_subscription, subscription_is_active

router = APIRouter(prefix="/auth", tags=["auth"])
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str
    company_name: str = ""


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str
    tenant_id: str
    subscription: Optional[dict] = None


class UserResponse(BaseModel):
    user_id: str
    email: str
    role: str
    tenant_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    professional_title: Optional[str] = None
    studio_name: Optional[str] = None
    avatar_url: Optional[str] = None
    tone: int = 2
    standing: int = 2
    response_length: int = 2
    dark_mode: bool = False
    primary_color: Optional[str] = None
    totp_enabled: bool = False
    subscription: Optional[dict] = None


class UpdateProfileRequest(BaseModel):
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    studio_name: Optional[str] = None


class UpdateSettingsRequest(BaseModel):
    tone: int
    standing: int
    response_length: int
    dark_mode: Optional[bool] = None
    primary_color: Optional[str] = None


class UpdatePreferencesRequest(BaseModel):
    preferences: dict


class CreateSubUserRequest(BaseModel):
    email: EmailStr
    password: str
    first_name: str = ""
    last_name: str = ""


# ---------------------------------------------------------------------------
# Dependency — get current user from token
# ---------------------------------------------------------------------------

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db)
) -> User:
    """Extract and validate the current user from JWT token."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    payload = decode_access_token(token)
    if not payload:
        raise credentials_exception

    user_id = payload.get("sub")
    if not user_id:
        raise credentials_exception

    user = crud.get_user_by_id(db, uuid.UUID(user_id))
    if not user or not user.is_active:
        raise credentials_exception

    return user


def get_admin_user(current_user: User = Depends(get_current_user)) -> User:
    """Only allow tenant admin users."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
        )
    return current_user


def get_superadmin_user(current_user: User = Depends(get_current_user)) -> User:
    """Only allow platform superadmin users."""
    if current_user.role != "superadmin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Superadmin access required"
        )
    return current_user


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("/register", status_code=status.HTTP_201_CREATED)
def register(request: RegisterRequest, db: Session = Depends(get_db)):
    """Register a new tenant and admin user."""
    try:
        tenant, admin = crud.register_tenant_with_admin(
            db=db,
            email=request.email,
            plain_password=request.password,
            company_name=request.company_name
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

    emit_billing_analytics_event(build_sign_up_event(admin, tenant=tenant))

    return {
        "message": "Registration successful",
        "tenant_id": str(tenant.id),
        "user_id": str(admin.id),
        "email": admin.email,
        "role": admin.role
    }


class LoginRequest(BaseModel):
    email: str
    password: str
    totp_code: Optional[str] = None


@router.post("/login", response_model=LoginResponse)
def login(
    request: LoginRequest,
    db: Session = Depends(get_db)
):
    """Login with email and password — returns JWT token.
    Include totp_code if 2FA is enabled on the account.
    """

    # Find user
    user = crud.get_user_by_email(db, request.email)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Verify password
    if not verify_password(request.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password"
        )

    # Check account is active
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated"
        )

    # Check 2FA if enabled
    if user.totp_enabled:
        if not request.totp_code:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="2FA code required",
                headers={"X-2FA-Required": "true"},
            )
        totp = pyotp.TOTP(user.totp_secret)
        if not totp.verify(request.totp_code, valid_window=1):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid 2FA code",
            )

    # Update last login
    user.last_login = datetime.now(timezone.utc)
    db.commit()

    # Generate token
    token = create_access_token(data={
        "sub": str(user.id),
        "tenant_id": str(user.tenant_id),
        "role": user.role
    })

    seats_used = crud.count_active_users_for_tenant(db, user.tenant_id)
    return LoginResponse(
        access_token=token,
        user_id=str(user.id),
        email=user.email,
        role=user.role,
        tenant_id=str(user.tenant_id),
        subscription=serialize_subscription(
            crud.get_tenant_subscription(db, user.tenant_id),
            user.tenant,
            seats_used=seats_used,
        ),
    )


@router.get("/me", response_model=UserResponse)
def get_me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user info including settings."""
    settings = crud.get_user_settings(db, current_user.id)
    prefs = crud.get_user_preferences(db, current_user.id)
    pref_data = dict(prefs.preferences or {}) if prefs else {}
    profile = current_user.profile
    tenant_profile = current_user.tenant.profile if current_user.tenant else None
    seats_used = crud.count_active_users_for_tenant(db, current_user.tenant_id)

    return UserResponse(
        user_id=str(current_user.id),
        email=current_user.email,
        role=current_user.role,
        tenant_id=str(current_user.tenant_id),
        first_name=profile.first_name if profile else None,
        last_name=profile.last_name if profile else None,
        professional_title=profile.professional_title if profile else None,
        studio_name=(tenant_profile.legal_name or tenant_profile.display_name) if tenant_profile else None,
        avatar_url=profile.profile_image_path if profile else None,
        tone=settings.tone if settings else 2,
        standing=settings.standing if settings else 2,
        response_length=settings.response_length if settings else 2,
        dark_mode=pref_data.get("dark_mode", False),
        primary_color=pref_data.get("primary_color"),
        totp_enabled=current_user.totp_enabled or False,
        subscription=serialize_subscription(
            crud.get_tenant_subscription(db, current_user.tenant_id),
            current_user.tenant,
            seats_used=seats_used,
        ),
    )


@router.put("/me", response_model=UserResponse)
def update_profile(
    request: UpdateProfileRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update user profile — name and (admin only) studio name."""
    profile = current_user.profile
    if profile:
        if request.first_name is not None:
            profile.first_name = request.first_name
        if request.last_name is not None:
            profile.last_name = request.last_name

    if request.studio_name is not None and current_user.role == "admin":
        tenant_profile = current_user.tenant.profile if current_user.tenant else None
        if tenant_profile:
            tenant_profile.legal_name = request.studio_name
            tenant_profile.display_name = request.studio_name

    db.commit()

    settings = crud.get_user_settings(db, current_user.id)
    prefs = crud.get_user_preferences(db, current_user.id)
    pref_data = dict(prefs.preferences or {}) if prefs else {}
    tenant_profile = current_user.tenant.profile if current_user.tenant else None
    seats_used = crud.count_active_users_for_tenant(db, current_user.tenant_id)

    return UserResponse(
        user_id=str(current_user.id),
        email=current_user.email,
        role=current_user.role,
        tenant_id=str(current_user.tenant_id),
        first_name=profile.first_name if profile else None,
        last_name=profile.last_name if profile else None,
        professional_title=profile.professional_title if profile else None,
        studio_name=(tenant_profile.legal_name or tenant_profile.display_name) if tenant_profile else None,
        avatar_url=profile.profile_image_path if profile else None,
        tone=settings.tone if settings else 2,
        standing=settings.standing if settings else 2,
        response_length=settings.response_length if settings else 2,
        dark_mode=pref_data.get("dark_mode", False),
        primary_color=pref_data.get("primary_color"),
        totp_enabled=current_user.totp_enabled or False,
        subscription=serialize_subscription(
            crud.get_tenant_subscription(db, current_user.tenant_id),
            current_user.tenant,
            seats_used=seats_used,
        ),
    )


@router.put("/me/settings")
def update_settings(
    request: UpdateSettingsRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update AI conversation settings."""
    # Validate slider values
    for val, name in [
        (request.tone, "tone"),
        (request.standing, "standing"),
        (request.response_length, "response_length")
    ]:
        if not 1 <= val <= 4:
            raise HTTPException(
                status_code=400,
                detail=f"{name} must be between 1 and 4"
            )

    settings = crud.update_user_settings(
        db=db,
        user_id=current_user.id,
        tone=request.tone,
        standing=request.standing,
        response_length=request.response_length
    )

    pref_data = {}
    if request.dark_mode is not None or request.primary_color is not None:
        prefs = crud.get_user_preferences(db, current_user.id)
        pref_data = dict(prefs.preferences or {}) if prefs else {}
        if request.dark_mode is not None:
            pref_data["dark_mode"] = request.dark_mode
        if request.primary_color is not None:
            pref_data["primary_color"] = request.primary_color
        crud.update_user_preferences(db, current_user.id, pref_data)
    else:
        prefs = crud.get_user_preferences(db, current_user.id)
        pref_data = dict(prefs.preferences or {}) if prefs else {}

    return {"message": "Settings updated", "settings": {
        "tone": settings.tone,
        "standing": settings.standing,
        "response_length": settings.response_length,
        "dark_mode": pref_data.get("dark_mode", False),
        "primary_color": pref_data.get("primary_color"),
    }}


@router.put("/me/preferences")
def update_preferences(
    request: UpdatePreferencesRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Update app preferences."""
    prefs = crud.update_user_preferences(
        db=db,
        user_id=current_user.id,
        preferences=request.preferences
    )
    return {"message": "Preferences updated", "preferences": prefs.preferences}


@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_sub_user(
    request: CreateSubUserRequest,
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Admin only — create a sub user under the same tenant."""
    if crud.get_user_by_email(db, request.email):
        raise HTTPException(
            status_code=400,
            detail="Email already registered"
        )

    subscription = crud.get_tenant_subscription(db, current_user.tenant_id)
    seat_capacity = subscription.seats if subscription and subscription_is_active(subscription.status) else 1
    seats_used = crud.count_active_users_for_tenant(db, current_user.tenant_id)
    if seats_used >= seat_capacity:
        raise HTTPException(
            status_code=403,
            detail="No available team seats for this tenant"
        )

    user = crud.create_user(
        db=db,
        tenant_id=current_user.tenant_id,
        email=request.email,
        plain_password=request.password,
        role="member"
    )

    # Set name if provided
    if request.first_name or request.last_name:
        if user.profile:
            user.profile.first_name = request.first_name
            user.profile.last_name = request.last_name
            db.commit()

    return {
        "message": "User created",
        "user_id": str(user.id),
        "email": user.email,
        "role": user.role
    }


@router.get("/users")
def list_users(
    current_user: User = Depends(get_admin_user),
    db: Session = Depends(get_db)
):
    """Admin only — list all users under this tenant."""
    users = crud.get_users_by_tenant(db, current_user.tenant_id)
    return {
        "users": [
            {
                "user_id": str(u.id),
                "email": u.email,
                "role": u.role,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
                "last_login": u.last_login.isoformat() if u.last_login else None
            }
            for u in users
        ]
    }