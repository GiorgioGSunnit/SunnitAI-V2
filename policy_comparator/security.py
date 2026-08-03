"""Authentication and tenant identity.

Tokens are HS256 JWTs carrying ``sub`` (user id), ``tenant_id``, ``email`` and
``role`` — the same shape the parent platform issues, signed with the same
secret, so a token minted there is accepted here. The local ``pc_staff_users``
table exists so the tool can also run standalone with no parent database.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import bcrypt
from jose import JWTError, jwt

from .config import Settings, get_settings


@dataclass(frozen=True)
class StaffIdentity:
    """The authenticated caller. Every query is scoped by ``tenant_id``."""

    user_id: uuid.UUID | None
    tenant_id: uuid.UUID
    email: str | None
    role: str = "staff"

    @property
    def is_admin(self) -> bool:
        return self.role in {"admin", "superadmin"}


class AuthError(Exception):
    """Token missing, malformed, expired or lacking a tenant."""


def hash_password(password: str) -> str:
    return bcrypt.hashpw(password.encode(), bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode(), hashed.encode())
    except ValueError:
        return False


def create_access_token(
    identity: StaffIdentity,
    *,
    expires_delta: timedelta | None = None,
    settings: Settings | None = None,
) -> str:
    settings = settings or get_settings()
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=settings.jwt_expire_minutes)
    )
    claims = {
        "sub": str(identity.user_id) if identity.user_id else None,
        "tenant_id": str(identity.tenant_id),
        "email": identity.email,
        "role": identity.role,
        "exp": expire,
    }
    return jwt.encode(claims, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_token(token: str, *, settings: Settings | None = None) -> dict:
    settings = settings or get_settings()
    try:
        return jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.jwt_algorithm])
    except JWTError as exc:
        raise AuthError("Invalid or expired token") from exc


def identity_from_token(token: str, *, settings: Settings | None = None) -> StaffIdentity:
    claims = decode_token(token, settings=settings)

    raw_tenant = claims.get("tenant_id")
    if not raw_tenant:
        # Without a tenant there is nothing to scope queries by, so the token
        # is rejected rather than defaulted to some "shared" tenant.
        raise AuthError("Token does not identify a tenant")
    try:
        tenant_id = uuid.UUID(str(raw_tenant))
    except ValueError as exc:
        raise AuthError("Token tenant_id is not a valid identifier") from exc

    raw_user = claims.get("sub")
    user_id: uuid.UUID | None = None
    if raw_user:
        try:
            user_id = uuid.UUID(str(raw_user))
        except ValueError:
            user_id = None

    return StaffIdentity(
        user_id=user_id,
        tenant_id=tenant_id,
        email=claims.get("email"),
        role=claims.get("role") or "staff",
    )
