import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

import stripe
from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from ...db import crud
from ...db.base import get_db
from ...db.models import Tenant, TenantSubscription, User
from ..billing import (
    BillingConfigError,
    get_automatic_tax_enabled,
    get_multiuser_min_seats,
    get_price_id_for_plan,
    get_stripe_client,
    get_trial_days,
    get_webhook_secret,
    normalize_quantity,
    serialize_subscription,
    subscription_is_active,
    tenant_seat_usage,
    trial_is_eligible,
    utc_from_unix,
    validate_return_url,
)
from .auth import get_current_user

router = APIRouter(prefix="/billing", tags=["billing"])
logger = logging.getLogger(__name__)


class CheckoutSessionRequest(BaseModel):
    plan_id: str
    success_url: str
    cancel_url: str
    source: Optional[str] = None
    return_to: Optional[str] = None
    quantity: Optional[int] = None


class CheckoutSessionResponse(BaseModel):
    checkout_url: str
    session_id: str


class SyncCheckoutSessionRequest(BaseModel):
    session_id: str


class SubscriptionResponse(BaseModel):
    plan_id: Optional[str] = None
    status: str
    seats: int
    seats_used: int
    seats_available: int
    seats_over_limit: int = 0
    min_team_seats: int = 3
    stripe_customer_id: Optional[str] = None
    stripe_subscription_id: Optional[str] = None
    stripe_checkout_session_id: Optional[str] = None
    trial_started_at: Optional[str] = None
    trial_ends_at: Optional[str] = None
    current_period_end: Optional[str] = None
    cancel_at_period_end: bool = False
    last_payment_status: Optional[str] = None
    is_active: bool = False
    access_block_reason: Optional[str] = None
    trial_days: int = 7


class BillingSnapshot(BaseModel):
    subscription: SubscriptionResponse


def _object_id(value) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return getattr(value, "id", None)


def _metadata(obj) -> dict:
    metadata = getattr(obj, "metadata", None)
    if not metadata:
        return {}
    if isinstance(metadata, dict):
        return dict(metadata)

    to_dict_recursive = getattr(metadata, "to_dict_recursive", None)
    if callable(to_dict_recursive):
        return dict(to_dict_recursive())

    to_dict = getattr(metadata, "to_dict", None)
    if callable(to_dict):
        return dict(to_dict())

    try:
        return dict(metadata)
    except (KeyError, TypeError, ValueError):
        return {}


def _stripe_error_detail(exc: stripe.error.StripeError) -> str:
    message = getattr(exc, "user_message", None) or str(exc)
    return f"Stripe billing error: {message}"


def _billing_storage_unavailable() -> HTTPException:
    logger.exception("Billing storage unavailable")
    return HTTPException(
        status_code=503,
        detail="Billing storage unavailable. Run database migrations and retry checkout.",
    )


def _subscription_response(db: Session, tenant_id: uuid.UUID) -> dict:
    tenant = db.query(Tenant).filter(Tenant.id == tenant_id).first()
    subscription = crud.get_tenant_subscription(db, tenant_id)
    usage = tenant_seat_usage(db, tenant_id, subscription)
    return serialize_subscription(subscription, tenant, seats_used=usage["seats_used"])


def _resolve_local_subscription(
    db: Session,
    *,
    stripe_subscription_id: Optional[str] = None,
    stripe_customer_id: Optional[str] = None,
    stripe_checkout_session_id: Optional[str] = None,
    tenant_id: Optional[uuid.UUID] = None,
) -> Optional[TenantSubscription]:
    if stripe_subscription_id:
        subscription = crud.get_tenant_subscription_by_stripe_subscription_id(db, stripe_subscription_id)
        if subscription:
            return subscription
    if stripe_checkout_session_id:
        subscription = crud.get_tenant_subscription_by_checkout_session_id(db, stripe_checkout_session_id)
        if subscription:
            return subscription
    if stripe_customer_id:
        subscription = crud.get_tenant_subscription_by_customer_id(db, stripe_customer_id)
        if subscription:
            return subscription
    if tenant_id:
        return crud.get_tenant_subscription(db, tenant_id)
    return None


def _checkout_completed_fallback_status(local_subscription: Optional[TenantSubscription]) -> str:
    if not local_subscription:
        return "checkout_completed"
    if subscription_is_active(local_subscription.status):
        return local_subscription.status

    trial_end = local_subscription.trial_ends_at or local_subscription.current_period_end
    if local_subscription.trial_started_at and trial_end and datetime.now(timezone.utc) < trial_end:
        return "trialing"

    if local_subscription.status in {"checkout_pending", "checkout_completed"}:
        return local_subscription.status
    return "checkout_completed"


def _get_or_create_customer_id(db: Session, user: User) -> str:
    stripe_client = get_stripe_client()
    local_subscription = crud.get_tenant_subscription(db, user.tenant_id)
    if local_subscription and local_subscription.stripe_customer_id:
        try:
            customer = stripe_client.Customer.retrieve(local_subscription.stripe_customer_id)
            if _object_id(customer) and not getattr(customer, "deleted", False):
                return customer.id
        except stripe.error.InvalidRequestError:
            pass

    customers = stripe_client.Customer.list(email=user.email, limit=10)
    for customer in customers.data:
        metadata = _metadata(customer)
        if metadata.get("tenant_id") == str(user.tenant_id):
            return customer.id

    customer = stripe_client.Customer.create(
        email=user.email,
        metadata={
            "user_id": str(user.id),
            "tenant_id": str(user.tenant_id),
        },
    )
    return customer.id


def _sync_from_stripe_subscription(
    db: Session,
    stripe_subscription,
    *,
    fallback_user_id: Optional[uuid.UUID] = None,
    fallback_tenant_id: Optional[uuid.UUID] = None,
    fallback_plan_id: Optional[str] = None,
    fallback_checkout_session_id: Optional[str] = None,
    last_payment_status: Optional[str] = None,
) -> TenantSubscription:
    metadata = _metadata(stripe_subscription)
    stripe_subscription_id = _object_id(stripe_subscription)
    stripe_customer_id = _object_id(getattr(stripe_subscription, "customer", None))
    local_subscription = _resolve_local_subscription(
        db,
        stripe_subscription_id=stripe_subscription_id,
        stripe_customer_id=stripe_customer_id,
        stripe_checkout_session_id=fallback_checkout_session_id,
        tenant_id=fallback_tenant_id,
    )

    tenant_id = metadata.get("tenant_id")
    user_id = metadata.get("user_id")
    plan_id = metadata.get("plan_id") or fallback_plan_id

    tenant_uuid = None
    if tenant_id:
        tenant_uuid = uuid.UUID(tenant_id)
    elif local_subscription:
        tenant_uuid = local_subscription.tenant_id
    elif fallback_tenant_id:
        tenant_uuid = fallback_tenant_id

    if not tenant_uuid:
        raise HTTPException(status_code=400, detail="Unable to resolve tenant for subscription")

    user_uuid = None
    if user_id:
        user_uuid = uuid.UUID(user_id)
    elif local_subscription and local_subscription.user_id:
        user_uuid = local_subscription.user_id
    elif fallback_user_id:
        user_uuid = fallback_user_id

    seats = 1
    items = getattr(getattr(stripe_subscription, "items", None), "data", None) or []
    if items:
        seats = normalize_quantity(plan_id or "plus-single-monthly", getattr(items[0], "quantity", None) or 1)
    elif local_subscription and local_subscription.seats:
        seats = local_subscription.seats

    if not plan_id and local_subscription:
        plan_id = local_subscription.plan_id

    return crud.upsert_tenant_subscription(
        db,
        tenant_uuid,
        user_id=user_uuid,
        plan_id=plan_id,
        status=getattr(stripe_subscription, "status", None) or "inactive",
        seats=seats,
        stripe_customer_id=stripe_customer_id,
        stripe_subscription_id=stripe_subscription_id,
        stripe_checkout_session_id=fallback_checkout_session_id,
        trial_started_at=utc_from_unix(getattr(stripe_subscription, "trial_start", None)),
        trial_ends_at=utc_from_unix(getattr(stripe_subscription, "trial_end", None)),
        current_period_end=utc_from_unix(getattr(stripe_subscription, "current_period_end", None)),
        cancel_at_period_end=bool(getattr(stripe_subscription, "cancel_at_period_end", False)),
        last_payment_status=last_payment_status,
    )


def _sync_from_checkout_session(
    db: Session,
    checkout_session,
    *,
    current_user: Optional[User] = None,
    last_payment_status: Optional[str] = None,
) -> TenantSubscription:
    metadata = _metadata(checkout_session)
    tenant_uuid = uuid.UUID(metadata["tenant_id"]) if metadata.get("tenant_id") else None
    user_uuid = uuid.UUID(metadata["user_id"]) if metadata.get("user_id") else None
    plan_id = metadata.get("plan_id")

    subscription_obj = getattr(checkout_session, "subscription", None)
    stripe_subscription_id = _object_id(subscription_obj)
    if not stripe_subscription_id:
        tenant_id = tenant_uuid or (current_user.tenant_id if current_user else None)
        if not tenant_id:
            raise HTTPException(status_code=400, detail="Checkout session has no subscription")

        local_subscription = _resolve_local_subscription(
            db,
            stripe_customer_id=_object_id(getattr(checkout_session, "customer", None)),
            stripe_checkout_session_id=checkout_session.id,
            tenant_id=tenant_id,
        )
        resolved_plan_id = plan_id or (local_subscription.plan_id if local_subscription else None)
        resolved_quantity = metadata.get("seats") or (local_subscription.seats if local_subscription else 1)
        return crud.upsert_tenant_subscription(
            db,
            tenant_id,
            user_id=user_uuid or (current_user.id if current_user else None),
            plan_id=resolved_plan_id,
            status=_checkout_completed_fallback_status(local_subscription),
            seats=normalize_quantity(resolved_plan_id or "plus-single-monthly", resolved_quantity),
            stripe_customer_id=_object_id(getattr(checkout_session, "customer", None)),
            stripe_checkout_session_id=checkout_session.id,
            last_payment_status=last_payment_status or getattr(checkout_session, "payment_status", None),
        )

    stripe_client = get_stripe_client()
    stripe_subscription = stripe_client.Subscription.retrieve(
        stripe_subscription_id,
        expand=["items.data.price"],
    )
    return _sync_from_stripe_subscription(
        db,
        stripe_subscription,
        fallback_user_id=user_uuid or (current_user.id if current_user else None),
        fallback_tenant_id=tenant_uuid or (current_user.tenant_id if current_user else None),
        fallback_plan_id=plan_id,
        fallback_checkout_session_id=checkout_session.id,
        last_payment_status=last_payment_status or getattr(checkout_session, "payment_status", None),
    )


def _verify_checkout_session_ownership(checkout_session, current_user: User) -> None:
    metadata = _metadata(checkout_session)
    if metadata.get("user_id") != str(current_user.id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Checkout session does not belong to this user",
        )
    if metadata.get("tenant_id") != str(current_user.tenant_id):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Checkout session does not belong to this tenant",
        )


@router.post("/checkout/sessions", response_model=CheckoutSessionResponse)
def create_checkout_session(
    request: CheckoutSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        existing_subscription = crud.get_tenant_subscription(db, current_user.tenant_id)
        validate_return_url(request.success_url)
        validate_return_url(request.cancel_url)
        price_id = get_price_id_for_plan(request.plan_id)
        quantity = normalize_quantity(request.plan_id, request.quantity)
        stripe_client = get_stripe_client()
        customer_id = _get_or_create_customer_id(db, current_user)
        automatic_tax_enabled = get_automatic_tax_enabled()

        metadata = {
            "user_id": str(current_user.id),
            "tenant_id": str(current_user.tenant_id),
            "plan_id": request.plan_id,
            "seats": str(quantity),
        }
        if request.source:
            metadata["source"] = request.source
        if request.return_to:
            metadata["return_to"] = request.return_to

        subscription_data = {"metadata": metadata}
        if trial_is_eligible(existing_subscription):
            subscription_data["trial_period_days"] = get_trial_days()

        checkout_session_params = {
            "mode": "subscription",
            "customer": customer_id,
            "client_reference_id": str(current_user.id),
            "success_url": request.success_url,
            "cancel_url": request.cancel_url,
            "payment_method_collection": "always",
            "automatic_tax": {"enabled": automatic_tax_enabled},
            "line_items": [{"price": price_id, "quantity": quantity}],
            "metadata": metadata,
            "subscription_data": subscription_data,
        }
        if automatic_tax_enabled:
            checkout_session_params["billing_address_collection"] = "required"
            checkout_session_params["customer_update"] = {"address": "auto"}

        checkout_session = stripe_client.checkout.Session.create(**checkout_session_params)

        crud.upsert_tenant_subscription(
            db,
            current_user.tenant_id,
            user_id=current_user.id,
            plan_id=(existing_subscription.plan_id if existing_subscription and subscription_is_active(existing_subscription.status) else request.plan_id),
            status=(existing_subscription.status if existing_subscription and subscription_is_active(existing_subscription.status) else "checkout_pending"),
            seats=(existing_subscription.seats if existing_subscription and subscription_is_active(existing_subscription.status) else quantity),
            stripe_customer_id=customer_id,
            stripe_checkout_session_id=checkout_session.id,
        )
    except BillingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except stripe.error.StripeError as exc:
        raise HTTPException(status_code=502, detail=_stripe_error_detail(exc))
    except SQLAlchemyError:
        db.rollback()
        raise _billing_storage_unavailable()

    if not getattr(checkout_session, "url", None):
        raise HTTPException(status_code=502, detail="Stripe did not return a checkout URL")

    return CheckoutSessionResponse(checkout_url=checkout_session.url, session_id=checkout_session.id)


@router.get("/subscription", response_model=SubscriptionResponse)
def get_subscription(
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    return _subscription_response(db, current_user.tenant_id)


@router.post("/checkout/sessions/sync", response_model=SubscriptionResponse)
def sync_checkout_session(
    request: SyncCheckoutSessionRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        stripe_client = get_stripe_client()
        checkout_session = stripe_client.checkout.Session.retrieve(
            request.session_id,
            expand=["subscription"],
        )
    except BillingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except stripe.error.StripeError as exc:
        raise HTTPException(status_code=502, detail=_stripe_error_detail(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Unable to retrieve checkout session: {exc}")

    try:
        _verify_checkout_session_ownership(checkout_session, current_user)
        subscription = _sync_from_checkout_session(db, checkout_session, current_user=current_user)
    except stripe.error.StripeError as exc:
        raise HTTPException(status_code=502, detail=_stripe_error_detail(exc))

    tenant = db.query(Tenant).filter(Tenant.id == subscription.tenant_id).first()
    usage = tenant_seat_usage(db, subscription.tenant_id, subscription)
    return serialize_subscription(subscription, tenant, seats_used=usage["seats_used"])


@router.post("/webhook")
async def stripe_webhook(request: Request, db: Session = Depends(get_db)):
    payload = await request.body()
    signature = request.headers.get("stripe-signature")
    if not signature:
        raise HTTPException(status_code=400, detail="Missing Stripe signature")

    try:
        event = get_stripe_client().Webhook.construct_event(payload, signature, get_webhook_secret())
    except BillingConfigError as exc:
        raise HTTPException(status_code=503, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid webhook: {exc}")

    event_type = event["type"]
    data_object = event["data"]["object"]

    try:
        if event_type == "checkout.session.completed":
            _sync_from_checkout_session(db, data_object)
        elif event_type in {"customer.subscription.created", "customer.subscription.updated", "customer.subscription.deleted"}:
            _sync_from_stripe_subscription(db, data_object)
        elif event_type == "invoice.payment_succeeded":
            stripe_subscription_id = _object_id(data_object.get("subscription"))
            if stripe_subscription_id:
                stripe_subscription = get_stripe_client().Subscription.retrieve(
                    stripe_subscription_id,
                    expand=["items.data.price"],
                )
                _sync_from_stripe_subscription(db, stripe_subscription, last_payment_status="paid")
        elif event_type == "invoice.payment_failed":
            stripe_subscription_id = _object_id(data_object.get("subscription"))
            if stripe_subscription_id:
                stripe_subscription = get_stripe_client().Subscription.retrieve(
                    stripe_subscription_id,
                    expand=["items.data.price"],
                )
                _sync_from_stripe_subscription(db, stripe_subscription, last_payment_status="failed")
    except stripe.error.StripeError as exc:
        raise HTTPException(status_code=502, detail=_stripe_error_detail(exc))

    return {"received": True, "trial_days": get_trial_days(), "min_team_seats": get_multiuser_min_seats()}
