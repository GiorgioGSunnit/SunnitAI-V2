"""Staff login against the local ``pc_staff_users`` table.

Tokens issued by the parent platform are accepted by every other route, so this
endpoint only exists to make standalone operation possible.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException, status
from sqlalchemy import select

from ..config import get_settings
from ..models import StaffUser
from ..schemas.api import LoginRequest, TokenResponse
from ..security import StaffIdentity, create_access_token, verify_password
from .deps import CurrentIdentity, DbSession

router = APIRouter(prefix="/api/auth", tags=["auth"])


@router.post("/login", response_model=TokenResponse)
def login(body: LoginRequest, db: DbSession) -> TokenResponse:
    settings = get_settings()
    user = db.execute(
        select(StaffUser).where(StaffUser.email == str(body.email).lower())
    ).scalar_one_or_none()

    # Same response whether the address is unknown or the password is wrong, so
    # the endpoint cannot be used to enumerate staff accounts.
    if user is None or not user.is_active or not verify_password(body.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or password"
        )

    user.last_login_at = datetime.now(timezone.utc)
    db.commit()

    identity = StaffIdentity(
        user_id=user.id, tenant_id=user.tenant_id, email=user.email, role=user.role
    )
    return TokenResponse(
        access_token=create_access_token(identity, settings=settings),
        expires_in_minutes=settings.jwt_expire_minutes,
        tenant_id=user.tenant_id,
        email=user.email,
        role=user.role,
    )


@router.get("/me")
def whoami(identity: CurrentIdentity) -> dict:
    return {
        "user_id": str(identity.user_id) if identity.user_id else None,
        "tenant_id": str(identity.tenant_id),
        "email": identity.email,
        "role": identity.role,
        "is_admin": identity.is_admin,
    }
