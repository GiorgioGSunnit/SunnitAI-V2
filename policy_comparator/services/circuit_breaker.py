"""Per-provider circuit breaker, persisted in the database.

State lives in a table rather than in worker memory so that restarting the
worker does not hand a struggling provider a fresh set of retries, and so that
two workers agree about whether a provider is currently being skipped.

Only outcomes that indicate the *provider* is unhealthy count against the
breaker. A missing field, a CAPTCHA or a bad credential are all conditions a
retry cannot fix, so they never trip it.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from sqlalchemy import select
from sqlalchemy.orm import Session

from ..config import Settings, get_settings
from ..models import ProviderHealthState
from ..models.enums import QuoteOutcome


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _aware(value: datetime | None) -> datetime | None:
    """SQLite hands back naive datetimes; treat those as UTC."""
    if value is None:
        return None
    return value if value.tzinfo else value.replace(tzinfo=timezone.utc)


def get_state(
    db: Session, tenant_id: uuid.UUID, provider_id: str, *, create: bool = True
) -> ProviderHealthState | None:
    state = db.execute(
        select(ProviderHealthState).where(
            ProviderHealthState.tenant_id == tenant_id,
            ProviderHealthState.provider_id == provider_id,
        )
    ).scalar_one_or_none()
    if state is None and create:
        state = ProviderHealthState(tenant_id=tenant_id, provider_id=provider_id)
        db.add(state)
        db.flush()
    return state


def is_open(
    db: Session, tenant_id: uuid.UUID, provider_id: str, *, now: datetime | None = None
) -> bool:
    """Whether the provider should be skipped right now."""
    state = get_state(db, tenant_id, provider_id, create=False)
    if state is None or state.circuit_open_until is None:
        return False
    return _aware(state.circuit_open_until) > (now or _now())


def open_until(
    db: Session, tenant_id: uuid.UUID, provider_id: str
) -> datetime | None:
    state = get_state(db, tenant_id, provider_id, create=False)
    if state is None:
        return None
    return _aware(state.circuit_open_until)


def record_success(db: Session, tenant_id: uuid.UUID, provider_id: str) -> None:
    """One success closes the breaker outright — no half-open probation."""
    state = get_state(db, tenant_id, provider_id)
    assert state is not None
    state.consecutive_failures = 0
    state.circuit_open_until = None
    state.last_success_at = _now()
    state.total_successes += 1
    state.last_error_category = None


def record_failure(
    db: Session,
    tenant_id: uuid.UUID,
    provider_id: str,
    *,
    error_category: str | None = None,
    settings: Settings | None = None,
) -> bool:
    """Count a failure. Returns ``True`` if this tripped the breaker open."""
    settings = settings or get_settings()
    state = get_state(db, tenant_id, provider_id)
    assert state is not None
    state.consecutive_failures += 1
    state.total_failures += 1
    state.last_failure_at = _now()
    state.last_error_category = error_category

    if state.consecutive_failures >= settings.circuit_breaker_threshold:
        state.circuit_open_until = _now() + timedelta(
            seconds=settings.circuit_breaker_cooldown_seconds
        )
        return True
    return False


def record_outcome(
    db: Session,
    tenant_id: uuid.UUID,
    provider_id: str,
    outcome: QuoteOutcome,
    *,
    error_category: str | None = None,
    settings: Settings | None = None,
) -> None:
    """Update the breaker from a finished attempt."""
    if outcome is QuoteOutcome.QUOTED:
        record_success(db, tenant_id, provider_id)
    elif outcome.counts_against_circuit:
        record_failure(
            db, tenant_id, provider_id, error_category=error_category, settings=settings
        )
    # Everything else — missing information, manual action, auth, config — is a
    # human-gated condition, not evidence that the provider is unhealthy.


def reset(db: Session, tenant_id: uuid.UUID, provider_id: str) -> None:
    """Force the breaker closed. Used when staff explicitly retry a provider."""
    state = get_state(db, tenant_id, provider_id)
    assert state is not None
    state.consecutive_failures = 0
    state.circuit_open_until = None
