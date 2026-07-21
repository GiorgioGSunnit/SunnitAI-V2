"""Platform administration endpoints for accounts and Stripe billing."""

import uuid
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo
from typing import Literal

import stripe
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from ...db import crud
from ...db.base import get_db
from ...db.models import Tenant, TenantProfile, TenantSubscription, User
from ..billing import (
    BillingConfigError,
    get_stripe_client,
    subscription_access_block_reason,
    utc_from_unix,
)
from .auth import get_superadmin_user
from .billing import _stripe_error_detail, _sync_from_stripe_subscription

router = APIRouter(prefix="/admin", tags=["admin"])
ROME_TIMEZONE = ZoneInfo("Europe/Rome")


class TrialExtensionRequest(BaseModel):
    access_through: date


class TrialExtensionResponse(BaseModel):
    account: dict


class AdminAccountUpdateRequest(BaseModel):
    user_is_active: bool
    tenant_is_active: bool
    access_override: Literal["inherit", "allowed", "blocked"]
    access_through: date | None = None


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
        "cancel_at": (
            subscription.cancel_at.isoformat()
            if subscription and getattr(subscription, "cancel_at", None)
            else None
        ),
        "canceled_at": (
            subscription.canceled_at.isoformat()
            if subscription and getattr(subscription, "canceled_at", None)
            else None
        ),
        "ended_at": (
            subscription.ended_at.isoformat()
            if subscription and getattr(subscription, "ended_at", None)
            else None
        ),
        "last_payment_status": subscription.last_payment_status if subscription else None,
        "has_access": access_block_reason is None,
        "access_block_reason": access_block_reason,
        "stripe_synced_at": (
            subscription.updated_at.isoformat()
            if subscription and subscription.updated_at
            else None
        ),
        "admin_access_override": getattr(subscription, "admin_access_override", None),
        "admin_access_until": (
            subscription.admin_access_until.isoformat()
            if subscription and getattr(subscription, "admin_access_until", None)
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


def _admin_account_for_user(db: Session, user_id: uuid.UUID) -> dict:
    row = (
        db.query(User, Tenant, TenantProfile, TenantSubscription)
        .join(Tenant, Tenant.id == User.tenant_id)
        .outerjoin(TenantProfile, TenantProfile.tenant_id == Tenant.id)
        .outerjoin(TenantSubscription, TenantSubscription.tenant_id == Tenant.id)
        .filter(User.id == user_id)
        .first()
    )
    if not row:
        raise HTTPException(status_code=404, detail="Utente non trovato")
    return _serialize_admin_account(*row)


def _set_access_override(
    db: Session,
    tenant_id: uuid.UUID,
    mode: str,
    access_until: datetime | None,
) -> TenantSubscription:
    subscription = crud.get_tenant_subscription(db, tenant_id)
    if not subscription:
        subscription = crud.upsert_tenant_subscription(
            db,
            tenant_id,
            status="inactive",
            seats=1,
        )
    subscription.admin_access_override = None if mode == "inherit" else mode
    subscription.admin_access_until = access_until if mode == "allowed" else None
    db.commit()
    db.refresh(subscription)
    return subscription


def _stripe_datetime(value) -> str | None:
    timestamp = utc_from_unix(value)
    return timestamp.isoformat() if timestamp else None


def _stripe_collection(page) -> list:
    auto_paging_iter = getattr(page, "auto_paging_iter", None)
    if callable(auto_paging_iter):
        return list(auto_paging_iter())
    return list(getattr(page, "data", None) or [])


def _current_stripe_subscription(subscriptions: list):
    if not subscriptions:
        return None
    status_priority = {
        "active": 6,
        "trialing": 5,
        "past_due": 4,
        "unpaid": 3,
        "incomplete": 2,
        "paused": 1,
    }
    return max(
        subscriptions,
        key=lambda item: (
            status_priority.get(getattr(item, "status", ""), 0),
            getattr(item, "created", 0) or 0,
        ),
    )


def _last_payment_status(stripe_subscription) -> str | None:
    invoice = getattr(stripe_subscription, "latest_invoice", None)
    invoice_status = getattr(invoice, "status", None)
    if invoice_status == "paid" or getattr(invoice, "paid", False):
        return "paid"
    if invoice_status in {"open", "uncollectible"} and getattr(
        stripe_subscription, "status", None
    ) in {"past_due", "unpaid"}:
        return "failed"
    return None


def _stripe_object_id(value) -> str | None:
    if isinstance(value, str):
        return value
    return getattr(value, "id", None)


def _serialize_stripe_subscription(subscription) -> dict:
    items = getattr(getattr(subscription, "items", None), "data", None) or []
    first_item = items[0] if items else None
    price = getattr(first_item, "price", None)
    return {
        "id": subscription.id,
        "status": subscription.status,
        "created_at": _stripe_datetime(getattr(subscription, "created", None)),
        "current_period_start": _stripe_datetime(
            getattr(subscription, "current_period_start", None)
        ),
        "current_period_end": _stripe_datetime(
            getattr(subscription, "current_period_end", None)
        ),
        "cancel_at_period_end": bool(
            getattr(subscription, "cancel_at_period_end", False)
        ),
        "cancel_at": _stripe_datetime(getattr(subscription, "cancel_at", None)),
        "canceled_at": _stripe_datetime(getattr(subscription, "canceled_at", None)),
        "ended_at": _stripe_datetime(getattr(subscription, "ended_at", None)),
        "price_id": getattr(price, "id", None),
        "plan_id": getattr(price, "lookup_key", None),
        "quantity": getattr(first_item, "quantity", None),
    }


def _serialize_stripe_invoice(invoice) -> dict:
    subscription_id = getattr(invoice, "subscription", None)
    if not isinstance(subscription_id, str):
        subscription_id = getattr(subscription_id, "id", None)
    return {
        "id": invoice.id,
        "number": getattr(invoice, "number", None),
        "status": getattr(invoice, "status", None),
        "created_at": _stripe_datetime(getattr(invoice, "created", None)),
        "currency": getattr(invoice, "currency", None),
        "amount_due": getattr(invoice, "amount_due", 0),
        "amount_paid": getattr(invoice, "amount_paid", 0),
        "hosted_invoice_url": getattr(invoice, "hosted_invoice_url", None),
        "invoice_pdf": getattr(invoice, "invoice_pdf", None),
        "subscription_id": subscription_id,
    }


def _serialize_stripe_payment(payment) -> dict:
    invoice_id = getattr(payment, "invoice", None)
    if not isinstance(invoice_id, str):
        invoice_id = getattr(invoice_id, "id", None)
    return {
        "id": payment.id,
        "status": getattr(payment, "status", None),
        "created_at": _stripe_datetime(getattr(payment, "created", None)),
        "currency": getattr(payment, "currency", None),
        "amount": getattr(payment, "amount", 0),
        "amount_received": getattr(payment, "amount_received", 0),
        "description": getattr(payment, "description", None),
        "invoice_id": invoice_id,
    }


@router.get("/accounts")
def list_platform_accounts(
    _current_user: User = Depends(get_superadmin_user),
    db: Session = Depends(get_db),
):
    return {"accounts": _platform_accounts(db)}


@router.post(
    "/tenants/{tenant_id}/billing/sync",
    response_model=TrialExtensionResponse,
)
def sync_tenant_billing(
    tenant_id: uuid.UUID,
    _current_user: User = Depends(get_superadmin_user),
    db: Session = Depends(get_db),
):
    subscription = crud.get_tenant_subscription(db, tenant_id)
    if not subscription or not (
        subscription.stripe_customer_id or subscription.stripe_subscription_id
    ):
        raise HTTPException(status_code=422, detail="Nessun riferimento Stripe da sincronizzare")

    try:
        stripe_client = get_stripe_client()
        stripe_subscription = None
        stripe_customer_id = subscription.stripe_customer_id
        stripe_subscription_id = subscription.stripe_subscription_id

        if not stripe_customer_id and stripe_subscription_id:
            stripe_subscription = stripe_client.Subscription.retrieve(
                stripe_subscription_id,
                expand=["items.data.price", "latest_invoice"],
            )
            stripe_customer_id = _stripe_object_id(
                getattr(stripe_subscription, "customer", None)
            )

        if stripe_customer_id:
            candidates = _stripe_collection(
                stripe_client.Subscription.list(
                    customer=stripe_customer_id,
                    status="all",
                    limit=100,
                )
            )
            current_subscription = _current_stripe_subscription(candidates)
            if current_subscription:
                stripe_subscription_id = current_subscription.id
        if not stripe_subscription_id:
            raise HTTPException(status_code=422, detail="Nessuna subscription Stripe trovata")

        if (
            stripe_subscription is None
            or stripe_subscription.id != stripe_subscription_id
        ):
            stripe_subscription = stripe_client.Subscription.retrieve(
                stripe_subscription_id,
                expand=["items.data.price", "latest_invoice"],
            )
        _sync_from_stripe_subscription(
            db,
            stripe_subscription,
            fallback_tenant_id=tenant_id,
            last_payment_status=_last_payment_status(stripe_subscription),
        )
    except BillingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except stripe.error.StripeError as exc:
        raise HTTPException(status_code=502, detail=_stripe_error_detail(exc)) from exc

    return {"account": _admin_account_for_tenant(db, tenant_id)}


@router.get("/tenants/{tenant_id}/billing/history")
def get_tenant_billing_history(
    tenant_id: uuid.UUID,
    _current_user: User = Depends(get_superadmin_user),
    db: Session = Depends(get_db),
):
    subscription = crud.get_tenant_subscription(db, tenant_id)
    if not subscription or not subscription.stripe_customer_id:
        raise HTTPException(status_code=422, detail="Nessun cliente Stripe collegato")

    try:
        stripe_client = get_stripe_client()
        subscriptions = _stripe_collection(
            stripe_client.Subscription.list(
                customer=subscription.stripe_customer_id,
                status="all",
                limit=100,
            )
        )
        invoices = _stripe_collection(
            stripe_client.Invoice.list(
                customer=subscription.stripe_customer_id,
                limit=100,
            )
        )
        payments = _stripe_collection(
            stripe_client.PaymentIntent.list(
                customer=subscription.stripe_customer_id,
                limit=100,
            )
        )
    except BillingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except stripe.error.StripeError as exc:
        raise HTTPException(status_code=502, detail=_stripe_error_detail(exc)) from exc

    return {
        "subscriptions": [
            _serialize_stripe_subscription(item) for item in subscriptions
        ],
        "invoices": [_serialize_stripe_invoice(item) for item in invoices],
        "payments": [_serialize_stripe_payment(item) for item in payments],
    }


@router.patch("/accounts/{user_id}")
def update_platform_account(
    user_id: uuid.UUID,
    request: AdminAccountUpdateRequest,
    _current_user: User = Depends(get_superadmin_user),
    db: Session = Depends(get_db),
):
    user = crud.get_user_by_id(db, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="Utente non trovato")
    tenant = db.query(Tenant).filter(Tenant.id == user.tenant_id).first()
    if not tenant:
        raise HTTPException(status_code=404, detail="Studio non trovato")

    access_until = None
    if request.access_override == "allowed":
        if request.access_through is None:
            raise HTTPException(status_code=422, detail="Indica la data di fine accesso")
        access_until = inclusive_access_date_to_trial_end(request.access_through)

    user.is_active = request.user_is_active
    tenant.is_active = request.tenant_is_active
    _set_access_override(db, tenant.id, request.access_override, access_until)
    return {"account": _admin_account_for_user(db, user.id)}


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
    trial_end = inclusive_access_date_to_trial_end(request.access_through)
    subscription = crud.get_tenant_subscription(db, tenant_id)
    stripe_can_be_updated = bool(
        subscription
        and subscription.stripe_subscription_id
        and subscription.status in {"active", "trialing"}
        and not subscription.cancel_at_period_end
    )
    if stripe_can_be_updated:
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

    _set_access_override(db, tenant_id, "allowed", trial_end)

    return {"account": _admin_account_for_tenant(db, tenant_id)}
