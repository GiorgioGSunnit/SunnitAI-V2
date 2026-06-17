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


class UserResponse(BaseModel):
    user_id: str
    email: str
    role: str
    tenant_id: str
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    professional_title: Optional[str] = None
    tone: int = 2
    standing: int = 2
    response_length: int = 2
    totp_enabled: bool = False


class UpdateSettingsRequest(BaseModel):
    tone: int
    standing: int
    response_length: int


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
    """Only allow admin users."""
    if current_user.role != "admin":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin access required"
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

    return LoginResponse(
        access_token=token,
        user_id=str(user.id),
        email=user.email,
        role=user.role,
        tenant_id=str(user.tenant_id)
    )


@router.get("/me", response_model=UserResponse)
def get_me(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Get current user info including settings."""
    settings = crud.get_user_settings(db, current_user.id)
    profile = current_user.profile

    return UserResponse(
        user_id=str(current_user.id),
        email=current_user.email,
        role=current_user.role,
        tenant_id=str(current_user.tenant_id),
        first_name=profile.first_name if profile else None,
        last_name=profile.last_name if profile else None,
        professional_title=profile.professional_title if profile else None,
        tone=settings.tone if settings else 2,
        standing=settings.standing if settings else 2,
        response_length=settings.response_length if settings else 2,
        totp_enabled=current_user.totp_enabled or False
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
    return {"message": "Settings updated", "settings": {
        "tone": settings.tone,
        "standing": settings.standing,
        "response_length": settings.response_length
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