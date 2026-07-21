"""Platform administration endpoints for accounts and Stripe billing."""

import uuid
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

import stripe
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...db import crud
from ...db.base import get_db
from ...db.models import Tenant, TenantProfile, TenantSubscription, User
from ..billing import BillingConfigError, get_stripe_client, subscription_access_block_reason
from .auth import get_superadmin_user
from .billing import _stripe_error_detail, _sync_from_stripe_subscription

router = APIRouter(prefix="/admin", tags=["admin"])
ROME_TIMEZONE = ZoneInfo("Europe/Rome")


class TrialExtensionRequest(BaseModel):
    access_through: date


class TrialExtensionResponse(BaseModel):
    account: dict


def inclusive_access_date_to_trial_end(
    access_through: date,
    *,
    now: datetime | None = None,
) -> datetime:
    """Return the UTC instant when billing resumes after an inclusive local date."""
    current = now or datetime.now(timezone.utc)
    current_utc = current.astimezone(timezone.utc)
    local_trial_end = datetime.combine(access_through + timedelta(days=1), time.min, ROME_TIMEZONE)
    trial_end = local_trial_end.astimezone(timezone.utc)
    if trial_end <= current_utc:
        raise HTTPException(status_code=422, detail="La data deve essere futura")
    return trial_end


def _serialize_admin_account(user, tenant, profile, subscription) -> dict:
    billing_block_reason = subscription_access_block_reason(subscription)
    if not user.is_active:
        access_block_reason = "Utente disattivato"
    elif not tenant.is_active:
        access_block_reason = "Studio disattivato"
    else:
        access_block_reason = billing_block_reason
    return {
        "user_id": str(user.id),
        "tenant_id": str(tenant.id),
        "email": user.email,
        "role": user.role,
        "user_is_active": bool(user.is_active),
        "tenant_is_active": bool(tenant.is_active),
        "studio_name": (profile.display_name if profile else None) or tenant.email,
        "created_at": user.created_at.isoformat() if user.created_at else None,
        "last_login": user.last_login.isoformat() if user.last_login else None,
        "plan_id": subscription.plan_id if subscription else tenant.plan,
        "stripe_status": subscription.status if subscription else "inactive",
        "stripe_customer_id": subscription.stripe_customer_id if subscription else None,
        "stripe_subscription_id": subscription.stripe_subscription_id if subscription else None,
        "trial_started_at": (
            subscription.trial_started_at.isoformat()
            if subscription and subscription.trial_started_at
            else None
        ),
        "trial_ends_at": (
            subscription.trial_ends_at.isoformat()
            if subscription and subscription.trial_ends_at
            else None
        ),
        "current_period_end": (
            subscription.current_period_end.isoformat()
            if subscription and subscription.current_period_end
            else None
        ),
        "cancel_at_period_end": bool(subscription.cancel_at_period_end) if subscription else False,
        "last_payment_status": subscription.last_payment_status if subscription else None,
        "has_access": access_block_reason is None,
        "access_block_reason": access_block_reason,
        "stripe_synced_at": (
            subscription.updated_at.isoformat()
            if subscription and subscription.updated_at
            else None
        ),
    }


def _platform_accounts(db: Session) -> list[dict]:
    rows = (
        db.query(User, Tenant, TenantProfile, TenantSubscription)
        .join(Tenant, Tenant.id == User.tenant_id)
        .outerjoin(TenantProfile, TenantProfile.tenant_id == Tenant.id)
        .outerjoin(TenantSubscription, TenantSubscription.tenant_id == Tenant.id)
        .order_by(User.created_at.desc())
        .all()
    )
    return [_serialize_admin_account(*row) for row in rows]


def _admin_account_for_tenant(db: Session, tenant_id: uuid.UUID) -> dict:
    row = (
        db.query(User, Tenant, TenantProfile, TenantSubscription)
        .join(Tenant, Tenant.id == User.tenant_id)
        .outerjoin(TenantProfile, TenantProfile.tenant_id == Tenant.id)
        .outerjoin(TenantSubscription, TenantSubscription.tenant_id == Tenant.id)
        .filter(Tenant.id == tenant_id)
        .order_by(User.created_at.asc())
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Account non trovato")
    return _serialize_admin_account(*row)


@router.get("/accounts")
def list_platform_accounts(
    _current_user: User = Depends(get_superadmin_user),
    db: Session = Depends(get_db),
):
    return {"accounts": _platform_accounts(db)}


@router.post(
    "/tenants/{tenant_id}/billing/trial-extension",
    response_model=TrialExtensionResponse,
)
def extend_tenant_trial(
    tenant_id: uuid.UUID,
    request: TrialExtensionRequest,
    _current_user: User = Depends(get_superadmin_user),
    db: Session = Depends(get_db),
):
    subscription = crud.get_tenant_subscription(db, tenant_id)
    if not subscription or not subscription.stripe_subscription_id:
        raise HTTPException(
            status_code=409,
            detail="L'account non ha una subscription Stripe da prorogare",
        )

    if subscription.cancel_at_period_end:
        raise HTTPException(
            status_code=409,
            detail="La subscription ha una disdetta programmata: annullala esplicitamente prima di prorogare la trial",
        )

    trial_end = inclusive_access_date_to_trial_end(request.access_through)
    existing_access_end = max(
        (
            value.astimezone(timezone.utc)
            for value in (subscription.trial_ends_at, subscription.current_period_end)
            if value is not None
        ),
        default=None,
    )
    if existing_access_end and trial_end <= existing_access_end:
        raise HTTPException(
            status_code=422,
            detail="La nuova data deve essere successiva al periodo di accesso già concesso",
        )
    try:
        stripe_subscription = get_stripe_client().Subscription.modify(
            subscription.stripe_subscription_id,
            trial_end=int(trial_end.timestamp()),
            proration_behavior="none",
        )
        _sync_from_stripe_subscription(db, stripe_subscription)
    except BillingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except stripe.error.StripeError as exc:
        raise HTTPException(status_code=502, detail=_stripe_error_detail(exc)) from exc

    return {"account": _admin_account_for_tenant(db, tenant_id)}
