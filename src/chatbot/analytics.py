import logging
from collections.abc import Mapping
from typing import Any, Optional

from .tracking_ws import queue_tracking_event

logger = logging.getLogger(__name__)

DEFAULT_CURRENCY = "EUR"


def _stripe_value(obj: Any, key: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(key, default)
    getter = getattr(obj, "get", None)
    if callable(getter):
        try:
            return getter(key, default)
        except TypeError:
            pass
    return getattr(obj, key, default)


def _stripe_object_id(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return _stripe_value(value, "id")


def _stripe_metadata(obj: Any) -> dict[str, Any]:
    metadata = _stripe_value(obj, "metadata")
    if not metadata:
        return {}
    if isinstance(metadata, Mapping):
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


def _minor_to_major(value: Any) -> float:
    try:
        return round(int(value or 0) / 100, 2)
    except (TypeError, ValueError):
        return 0.0


def _positive_minor_amount(value: Any) -> bool:
    try:
        return int(value or 0) > 0
    except (TypeError, ValueError):
        return False


def _clean_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in payload.items() if value is not None}


def _optional_str(value: Any) -> Optional[str]:
    if value is None or value == "":
        return None
    return str(value)


def build_free_trial_event(
    subscription: Any,
    *,
    checkout_session: Any = None,
    stripe_event_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    if _stripe_value(subscription, "status") != "trialing":
        return None

    subscription_metadata = _stripe_metadata(subscription)
    checkout_metadata = _stripe_metadata(checkout_session)
    metadata = {**subscription_metadata, **checkout_metadata}

    return _clean_payload(
        {
            "event": "free_trial",
            "event_id": stripe_event_id,
            "transaction_id": _stripe_object_id(checkout_session),
            "plan_id": _stripe_value(subscription, "plan_id") or metadata.get("plan_id"),
            "subscription_id": _stripe_value(subscription, "stripe_subscription_id"),
            "checkout_session_id": _stripe_value(subscription, "stripe_checkout_session_id")
            or _stripe_object_id(checkout_session),
            "tenant_id": _optional_str(_stripe_value(subscription, "tenant_id") or metadata.get("tenant_id")),
            "user_id": _optional_str(_stripe_value(subscription, "user_id") or metadata.get("user_id")),
        }
    )


def _coupon_code(invoice: Any) -> Optional[str]:
    discount = _stripe_value(invoice, "discount")
    discounts = _stripe_value(invoice, "discounts") or []
    if discount is None and discounts:
        discount = discounts[0]
    coupon = _stripe_value(discount, "coupon")
    if isinstance(coupon, str):
        return coupon
    return _stripe_value(coupon, "name") or _stripe_value(coupon, "id")


def _invoice_items(invoice: Any, metadata: Mapping[str, Any]) -> list[dict[str, Any]]:
    lines = _stripe_value(invoice, "lines")
    line_data = _stripe_value(lines, "data") or []
    items = []

    for line in line_data:
        quantity = int(_stripe_value(line, "quantity") or 1)
        price = _stripe_value(line, "price")
        plan = _stripe_value(line, "plan")
        product_id = _stripe_object_id(_stripe_value(price, "product")) or _stripe_object_id(
            _stripe_value(plan, "product")
        )
        price_id = _stripe_object_id(price) or _stripe_object_id(plan)
        line_amount = _stripe_value(line, "amount")
        unit_amount = _stripe_value(price, "unit_amount")
        if unit_amount is None and quantity > 0 and line_amount is not None:
            unit_amount = int(line_amount) / quantity

        item_name = (
            _stripe_value(line, "description")
            or metadata.get("plan_id")
            or price_id
            or product_id
            or "Astrea subscription"
        )
        item_id = price_id or product_id or metadata.get("plan_id") or item_name

        items.append(
            {
                "item_name": str(item_name),
                "item_id": str(item_id),
                "price": _minor_to_major(unit_amount),
                "quantity": quantity,
            }
        )

    if items:
        return items

    item_name = metadata.get("plan_id") or "Astrea subscription"
    return [
        {
            "item_name": str(item_name),
            "item_id": str(metadata.get("plan_id") or item_name),
            "price": _minor_to_major(_stripe_value(invoice, "amount_paid") or _stripe_value(invoice, "total")),
            "quantity": 1,
        }
    ]


def build_purchase_event(
    invoice: Any,
    *,
    subscription: Any = None,
    stripe_event_id: Optional[str] = None,
) -> Optional[dict[str, Any]]:
    amount_paid = _stripe_value(invoice, "amount_paid")
    if amount_paid is None:
        amount_paid = _stripe_value(invoice, "total")
    if not _positive_minor_amount(amount_paid):
        return None

    metadata = {**_stripe_metadata(subscription), **_stripe_metadata(invoice)}
    total_details = _stripe_value(invoice, "total_details") or {}
    currency = (_stripe_value(invoice, "currency") or DEFAULT_CURRENCY).upper()
    transaction_id = _stripe_object_id(invoice) or _stripe_object_id(_stripe_value(invoice, "payment_intent"))
    coupon = _coupon_code(invoice)

    ecommerce = _clean_payload(
        {
            "currency": currency,
            "value": _minor_to_major(amount_paid),
            "tax": _minor_to_major(_stripe_value(total_details, "amount_tax")),
            "shipping": _minor_to_major(
                _stripe_value(total_details, "amount_shipping")
                or _stripe_value(invoice, "amount_shipping")
            ),
            "transaction_id": transaction_id,
            "coupon": coupon,
            "items": _invoice_items(invoice, metadata),
        }
    )

    return _clean_payload(
        {
            "event": "purchase",
            "event_id": stripe_event_id,
            "ecommerce": ecommerce,
            "plan_id": metadata.get("plan_id"),
            "subscription_id": _stripe_value(subscription, "stripe_subscription_id")
            or _stripe_object_id(_stripe_value(invoice, "subscription")),
            "tenant_id": _optional_str(_stripe_value(subscription, "tenant_id") or metadata.get("tenant_id")),
            "user_id": _optional_str(_stripe_value(subscription, "user_id") or metadata.get("user_id")),
        }
    )


def emit_billing_analytics_event(payload: Optional[dict[str, Any]]) -> bool:
    if not payload:
        return False

    user_id = payload.get("user_id")
    if not user_id:
        logger.info("Billing analytics event skipped: payload has no user_id")
        return False

    return queue_tracking_event(user_id, payload)
