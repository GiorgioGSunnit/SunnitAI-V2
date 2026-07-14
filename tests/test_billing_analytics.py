import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import stripe

from src.chatbot.analytics import (
    build_free_trial_event,
    build_purchase_event,
    emit_billing_analytics_event,
)
from src.chatbot.routes import billing as billing_routes


def test_build_free_trial_event_uses_webhook_subscription_state():
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    subscription = SimpleNamespace(
        tenant_id=tenant_id,
        user_id=user_id,
        plan_id="plus-single-monthly",
        status="trialing",
        stripe_subscription_id="sub_trial",
        stripe_checkout_session_id="cs_trial",
        trial_started_at=datetime.now(timezone.utc),
        trial_ends_at=datetime.now(timezone.utc) + timedelta(days=7),
    )
    checkout_session = SimpleNamespace(
        id="cs_trial",
        metadata={
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "plan_id": "plus-single-monthly",
        },
    )

    payload = build_free_trial_event(
        subscription,
        checkout_session=checkout_session,
        stripe_event_id="evt_trial",
    )

    assert payload == {
        "event": "free_trial",
        "event_id": "evt_trial",
        "transaction_id": "cs_trial",
        "plan_id": "plus-single-monthly",
        "subscription_id": "sub_trial",
        "checkout_session_id": "cs_trial",
        "tenant_id": str(tenant_id),
        "user_id": str(user_id),
    }


def test_build_purchase_event_maps_paid_invoice_to_ecommerce_payload():
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    subscription = SimpleNamespace(
        tenant_id=tenant_id,
        user_id=user_id,
        plan_id="plus-single-annual",
        stripe_subscription_id="sub_paid",
    )
    invoice = {
        "id": "in_paid",
        "subscription": "sub_paid",
        "currency": "eur",
        "amount_paid": 120780,
        "total_details": {"amount_tax": 21780, "amount_shipping": 0},
        "discount": {"coupon": {"id": "WELCOME10"}},
        "lines": {
            "data": [
                {
                    "description": "Astrea Plus annuale",
                    "quantity": 1,
                    "amount": 99000,
                    "price": {
                        "id": "price_plus_annual",
                        "product": "prod_astrea_plus",
                        "unit_amount": 99000,
                    },
                }
            ]
        },
        "metadata": {
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "plan_id": "plus-single-annual",
        },
    }

    payload = build_purchase_event(invoice, subscription=subscription, stripe_event_id="evt_paid")

    assert payload == {
        "event": "purchase",
        "event_id": "evt_paid",
        "ecommerce": {
            "currency": "EUR",
            "value": 1207.8,
            "tax": 217.8,
            "shipping": 0.0,
            "transaction_id": "in_paid",
            "coupon": "WELCOME10",
            "items": [
                {
                    "item_name": "Astrea Plus annuale",
                    "item_id": "price_plus_annual",
                    "price": 990.0,
                    "quantity": 1,
                }
            ],
        },
        "plan_id": "plus-single-annual",
        "subscription_id": "sub_paid",
        "tenant_id": str(tenant_id),
        "user_id": str(user_id),
    }


def test_build_purchase_event_ignores_zero_value_trial_invoice():
    assert build_purchase_event({"id": "in_trial", "amount_paid": 0}) is None


def test_emit_billing_analytics_event_queues_for_payload_user_id(monkeypatch):
    captured = {}

    def fake_queue_tracking_event(user_id, payload):
        captured["user_id"] = user_id
        captured["payload"] = payload
        return True

    monkeypatch.setattr("src.chatbot.analytics.queue_tracking_event", fake_queue_tracking_event)

    payload = {"event": "free_trial", "user_id": "user_123"}

    assert emit_billing_analytics_event(payload) is True
    assert captured == {"user_id": "user_123", "payload": payload}


def test_emit_billing_analytics_event_skips_payload_without_user_id():
    assert emit_billing_analytics_event({"event": "free_trial"}) is False


def test_stripe_webhook_emits_free_trial_from_checkout_completed(monkeypatch):
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    checkout_session = SimpleNamespace(
        id="cs_trial",
        metadata={
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "plan_id": "plus-single-monthly",
        },
    )
    subscription = SimpleNamespace(
        tenant_id=tenant_id,
        user_id=user_id,
        plan_id="plus-single-monthly",
        status="trialing",
        stripe_subscription_id="sub_trial",
        stripe_checkout_session_id="cs_trial",
    )
    captured = []

    fake_stripe = SimpleNamespace(
        Webhook=SimpleNamespace(
            construct_event=lambda _payload, _signature, _secret: {
                "id": "evt_checkout",
                "type": "checkout.session.completed",
                "data": {"object": checkout_session},
            }
        )
    )

    monkeypatch.setattr(billing_routes, "get_stripe_client", lambda: fake_stripe)
    monkeypatch.setattr(billing_routes, "get_webhook_secret", lambda: "whsec_test")
    monkeypatch.setattr(
        billing_routes,
        "_sync_from_checkout_session",
        lambda _db, _checkout_session: subscription,
    )
    monkeypatch.setattr(
        billing_routes,
        "emit_billing_analytics_event",
        lambda payload: captured.append(payload) or True,
    )

    request = SimpleNamespace(
        body=lambda: asyncio.sleep(0, result=b"{}"),
        headers={"stripe-signature": "sig_test"},
    )

    result = asyncio.run(billing_routes.stripe_webhook(request, db=SimpleNamespace()))

    assert result["received"] is True
    assert captured == [
        {
            "event": "free_trial",
            "event_id": "evt_checkout",
            "transaction_id": "cs_trial",
            "plan_id": "plus-single-monthly",
            "subscription_id": "sub_trial",
            "checkout_session_id": "cs_trial",
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
        }
    ]


def test_stripe_webhook_emits_purchase_from_paid_invoice(monkeypatch):
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    invoice = stripe.Invoice.construct_from(
        {
            "id": "in_paid",
            "subscription": "sub_paid",
            "currency": "eur",
            "amount_paid": 9900,
            "total_details": {"amount_tax": 1800, "amount_shipping": 0},
            "lines": {
                "data": [
                    {
                        "description": "Astrea Plus mensile",
                        "quantity": 1,
                        "price": {"id": "price_plus_monthly", "unit_amount": 8100},
                    }
                ]
            },
            "metadata": {
                "tenant_id": str(tenant_id),
                "user_id": str(user_id),
                "plan_id": "plus-single-monthly",
            },
        },
        key=None,
    )
    subscription = SimpleNamespace(
        tenant_id=tenant_id,
        user_id=user_id,
        plan_id="plus-single-monthly",
        stripe_subscription_id="sub_paid",
    )
    captured = []

    fake_stripe = SimpleNamespace(
        Webhook=SimpleNamespace(
            construct_event=lambda _payload, _signature, _secret: {
                "id": "evt_invoice",
                "type": "invoice.payment_succeeded",
                "data": {"object": invoice},
            }
        ),
        Subscription=SimpleNamespace(retrieve=lambda *_args, **_kwargs: SimpleNamespace(id="sub_paid")),
    )

    monkeypatch.setattr(billing_routes, "get_stripe_client", lambda: fake_stripe)
    monkeypatch.setattr(billing_routes, "get_webhook_secret", lambda: "whsec_test")
    monkeypatch.setattr(
        billing_routes,
        "_sync_from_stripe_subscription",
        lambda _db, _stripe_subscription, last_payment_status=None: subscription,
    )
    monkeypatch.setattr(
        billing_routes,
        "emit_billing_analytics_event",
        lambda payload: captured.append(payload) or True,
    )

    request = SimpleNamespace(
        body=lambda: asyncio.sleep(0, result=b"{}"),
        headers={"stripe-signature": "sig_test"},
    )

    result = asyncio.run(billing_routes.stripe_webhook(request, db=SimpleNamespace()))

    assert result["received"] is True
    assert captured == [
        {
            "event": "purchase",
            "event_id": "evt_invoice",
            "ecommerce": {
                "currency": "EUR",
                "value": 99.0,
                "tax": 18.0,
                "shipping": 0.0,
                "transaction_id": "in_paid",
                "items": [
                    {
                        "item_name": "Astrea Plus mensile",
                        "item_id": "price_plus_monthly",
                        "price": 81.0,
                        "quantity": 1,
                    }
                ],
            },
            "plan_id": "plus-single-monthly",
            "subscription_id": "sub_paid",
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
        }
    ]
