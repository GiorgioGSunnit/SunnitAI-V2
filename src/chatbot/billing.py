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
NON_BLOCKING_PENDING_STATUSES = {"checkout_completed"}

PLACEHOLDER_VALUES = {
    "sk_test_your_key_here",
    "sk_live_your_key_here",
    "whsec_your_webhook_secret_here",
    "price_test_single_monthly",
    "price_test_single_annual",
    "price_test_multiuser",
    "price_live_plus_single_monthly",
    "price_live_plus_single_annual",
    "price_live_plus_multiuser",
}


class BillingConfigError(RuntimeError):
    pass


class BillingAccessBlocked(HTTPException):
    def __init__(self, detail: str = "Subscription inactive or trial expired"):
        super().__init__(status_code=402, detail=detail)


def _looks_like_placeholder(value: str) -> bool:
    normalized = value.strip().lower()
    return (
        not normalized
        or normalized in PLACEHOLDER_VALUES
        or "your-" in normalized
        or "your_" in normalized
        or "placeholder" in normalized
        or normalized.endswith("_here")
    )


def require_stripe_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if _looks_like_placeholder(value):
        raise BillingConfigError(f"Missing or placeholder {name}")
    return value


def stripe_bool_env(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def stripe_int_env(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise BillingConfigError(f"{name} must be an integer") from exc
    return max(minimum, value)


def get_stripe_client():
    api_key = require_stripe_env("STRIPE_SECRET_KEY")
    stripe.api_key = api_key
    return stripe


def get_webhook_secret() -> str:
    return require_stripe_env("STRIPE_WEBHOOK_SECRET")


def get_price_id_for_plan(plan_id: str) -> str:
    env_name = PLAN_PRICE_ENV.get(plan_id)
    if not env_name:
        raise ValueError(f"Unsupported plan_id '{plan_id}'")
    return require_stripe_env(env_name)


def get_trial_days() -> int:
    return stripe_int_env("STRIPE_TRIAL_DAYS", default=7, minimum=0)


def get_multiuser_min_seats() -> int:
    return stripe_int_env("STRIPE_MULTIUSER_MIN_SEATS", default=3, minimum=3)


def get_automatic_tax_enabled() -> bool:
    return stripe_bool_env("STRIPE_AUTOMATIC_TAX_ENABLED", default=False)


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


def subscription_allows_access(status: Optional[str], access_block_reason: Optional[str]) -> bool:
    if access_block_reason is not None:
        return False
    return subscription_is_active(status) or status in NON_BLOCKING_PENDING_STATUSES


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
        return "Nessun piano attivo. Apri Piani per continuare a usare Astrea."
    now = datetime.now(timezone.utc)
    admin_override = getattr(subscription, "admin_access_override", None)
    admin_access_until = getattr(subscription, "admin_access_until", None)
    if admin_override == "blocked":
        return "Accesso sospeso manualmente da un amministratore."
    if admin_override == "allowed":
        if admin_access_until is None or now < admin_access_until:
            return None
    if subscription.status == "checkout_pending":
        return "Checkout non completato. Apri Piani per attivare la prova gratuita."
    if subscription.status in NON_BLOCKING_PENDING_STATUSES:
        return None
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
        "is_active": (
            getattr(subscription, "admin_access_override", None) == "allowed"
            and (
                getattr(subscription, "admin_access_until", None) is None
                or datetime.now(timezone.utc) < subscription.admin_access_until
            )
        ) or subscription_allows_access(status, access_block_reason),
        "access_block_reason": access_block_reason,
        "trial_days": get_trial_days(),
    }
