import os
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import stripe
from fastapi import HTTPException
from sqlalchemy.orm import Session

from ..db import crud
from ..db.models import Tenant, TenantSubscription

PLAN_PRICE_ENV = {
    "plus-single-monthly": "STRIPE_PRICE_PLUS_SINGLE_MONTHLY",
    "plus-single-annual": "STRIPE_PRICE_PLUS_SINGLE_ANNUAL",
    "plus-multiuser": "STRIPE_PRICE_PLUS_MULTIUSER",
}

ACCESSIBLE_SUBSCRIPTION_STATUSES = {"active", "trialing"}
NON_BLOCKING_PENDING_STATUSES = {"checkout_pending", "checkout_completed"}
DEFAULT_TRIAL_DAYS = int(os.getenv("STRIPE_TRIAL_DAYS", "7"))
DEFAULT_MULTIUSER_MIN_SEATS = int(os.getenv("STRIPE_MULTIUSER_MIN_SEATS", "3"))


class BillingConfigError(RuntimeError):
    pass


class BillingAccessBlocked(HTTPException):
    def __init__(self, detail: str = "Subscription inactive or trial expired"):
        super().__init__(status_code=402, detail=detail)


def get_stripe_client():
    api_key = os.getenv("STRIPE_SECRET_KEY", "").strip()
    if not api_key:
        raise BillingConfigError("Missing STRIPE_SECRET_KEY")
    stripe.api_key = api_key
    return stripe


def get_webhook_secret() -> str:
    secret = os.getenv("STRIPE_WEBHOOK_SECRET", "").strip()
    if not secret:
        raise BillingConfigError("Missing STRIPE_WEBHOOK_SECRET")
    return secret


def get_price_id_for_plan(plan_id: str) -> str:
    env_name = PLAN_PRICE_ENV.get(plan_id)
    if not env_name:
        raise ValueError(f"Unsupported plan_id '{plan_id}'")
    price_id = os.getenv(env_name, "").strip()
    if not price_id:
        raise BillingConfigError(f"Missing {env_name}")
    return price_id


def get_trial_days() -> int:
    return max(0, DEFAULT_TRIAL_DAYS)


def get_multiuser_min_seats() -> int:
    return max(3, DEFAULT_MULTIUSER_MIN_SEATS)


def is_multiuser_plan(plan_id: Optional[str]) -> bool:
    return plan_id == "plus-multiuser"


def normalize_quantity(plan_id: str, quantity: Optional[int]) -> int:
    if not is_multiuser_plan(plan_id):
        return 1
    requested = get_multiuser_min_seats() if quantity is None else int(quantity)
    return max(get_multiuser_min_seats(), requested)


def utc_from_unix(timestamp: Optional[int]) -> Optional[datetime]:
    if not timestamp:
        return None
    return datetime.fromtimestamp(timestamp, tz=timezone.utc)


def trial_is_eligible(subscription: Optional[TenantSubscription]) -> bool:
    return subscription is None or subscription.trial_started_at is None


def subscription_is_active(status: Optional[str]) -> bool:
    return status in ACCESSIBLE_SUBSCRIPTION_STATUSES


def validate_return_url(url: str) -> str:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("success_url/cancel_url must be absolute http(s) URLs")
    return url


def tenant_seat_capacity(subscription: Optional[TenantSubscription]) -> int:
    if not subscription:
        return 1
    return max(1, int(subscription.seats or 1))


def tenant_seat_usage(db: Session, tenant_id, subscription: Optional[TenantSubscription] = None) -> dict:
    used = crud.count_active_users_for_tenant(db, tenant_id)
    total = tenant_seat_capacity(subscription)
    available = max(0, total - used)
    over_limit = max(0, used - total)
    return {
        "seats": total,
        "seats_used": used,
        "seats_available": available,
        "seats_over_limit": over_limit,
    }


def subscription_access_block_reason(subscription: Optional[TenantSubscription]) -> Optional[str]:
    if not subscription:
        return None
    if subscription.status in NON_BLOCKING_PENDING_STATUSES:
        return None
    now = datetime.now(timezone.utc)
    if subscription.status == "trialing":
        trial_end = subscription.trial_ends_at or subscription.current_period_end
        if trial_end and now >= trial_end:
            return "La prova gratuita è scaduta. Attiva il pagamento per continuare a usare Astrea."
        return None
    if subscription.status == "active":
        if subscription.last_payment_status == "failed":
            return "Pagamento non riuscito. Aggiorna il metodo di pagamento per continuare a usare Astrea."
        return None
    if subscription.status in {"past_due", "unpaid", "incomplete", "incomplete_expired"}:
        return "Pagamento richiesto. Aggiorna il metodo di pagamento per continuare a usare Astrea."
    if subscription.status in {"canceled", "cancelled", "inactive"}:
        return "Abbonamento non attivo. Riattiva un piano per continuare a usare Astrea."
    return None


def enforce_tenant_product_access(db: Session, tenant_id) -> None:
    subscription = crud.get_tenant_subscription(db, tenant_id)
    reason = subscription_access_block_reason(subscription)
    if reason:
        raise BillingAccessBlocked(reason)


def serialize_subscription(
    subscription: Optional[TenantSubscription],
    tenant: Optional[Tenant] = None,
    *,
    seats_used: Optional[int] = None,
) -> dict:
    plan_id = subscription.plan_id if subscription else (tenant.plan if tenant else "basic")
    status = subscription.status if subscription else ("inactive")
    current_period_end = (
        subscription.current_period_end if subscription else (tenant.subscription_end if tenant else None)
    )
    seats_total = tenant_seat_capacity(subscription)
    seats_used_value = seats_total if seats_used is None else max(0, int(seats_used))
    access_block_reason = subscription_access_block_reason(subscription)
    return {
        "plan_id": plan_id,
        "status": status,
        "seats": seats_total,
        "seats_used": seats_used_value,
        "seats_available": max(0, seats_total - seats_used_value),
        "seats_over_limit": max(0, seats_used_value - seats_total),
        "min_team_seats": get_multiuser_min_seats(),
        "stripe_customer_id": subscription.stripe_customer_id if subscription else None,
        "stripe_subscription_id": subscription.stripe_subscription_id if subscription else None,
        "stripe_checkout_session_id": subscription.stripe_checkout_session_id if subscription else None,
        "trial_started_at": subscription.trial_started_at.isoformat() if subscription and subscription.trial_started_at else None,
        "trial_ends_at": subscription.trial_ends_at.isoformat() if subscription and subscription.trial_ends_at else None,
        "current_period_end": current_period_end.isoformat() if current_period_end else None,
        "cancel_at_period_end": subscription.cancel_at_period_end if subscription else False,
        "last_payment_status": subscription.last_payment_status if subscription else None,
        "is_active": subscription_is_active(status) and access_block_reason is None,
        "access_block_reason": access_block_reason,
        "trial_days": get_trial_days(),
    }
