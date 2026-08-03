"""Shared FastAPI dependencies: authentication, tenant scoping, rate limiting.

Tenant scoping is enforced in one place — :func:`get_request_for_identity` — so
no route can accidentally load a record belonging to another tenant. A record
that exists but belongs elsewhere returns 404, not 403: telling a caller that a
request id exists under a different tenant is itself a leak.
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict, deque
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..db import get_db
from ..models import QuoteRequest
from ..security import AuthError, StaffIdentity, identity_from_token

DbSession = Annotated[Session, Depends(get_db)]


def get_current_identity(
    authorization: Annotated[str | None, Header()] = None,
) -> StaffIdentity:
    if not authorization or not authorization.lower().startswith("bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    token = authorization.split(" ", 1)[1].strip()
    try:
        return identity_from_token(token)
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(exc),
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc


CurrentIdentity = Annotated[StaffIdentity, Depends(get_current_identity)]


def require_admin(identity: CurrentIdentity) -> StaffIdentity:
    if not identity.is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN, detail="Administrator role required"
        )
    return identity


def get_request_for_identity(
    request_id: uuid.UUID, db: Session, identity: StaffIdentity
) -> QuoteRequest:
    """Load a quotation request, or 404 if it is not this tenant's."""
    record = db.get(QuoteRequest, request_id)
    if record is None or record.tenant_id != identity.tenant_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Quotation request not found"
        )
    return record


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------

#: tenant id -> submission timestamps. In-process, which is enough for a single
#: internal deployment; a multi-process rollout should move this to Redis.
_submissions: dict[str, deque[float]] = defaultdict(deque)


def enforce_quote_rate_limit(
    identity: StaffIdentity, settings: Settings | None = None
) -> None:
    """Cap quotation submissions per tenant per hour."""
    settings = settings or get_settings()
    limit = settings.quote_rate_limit_per_hour
    if limit <= 0:
        return

    now = time.monotonic()
    window = _submissions[str(identity.tenant_id)]
    while window and now - window[0] > 3600:
        window.popleft()

    if len(window) >= limit:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=(
                f"Quotation rate limit reached ({limit}/hour). "
                "Wait before submitting more requests."
            ),
        )
    window.append(now)


def reset_rate_limits() -> None:
    """Clear the rate-limit window. Used by tests."""
    _submissions.clear()
