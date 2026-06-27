import uuid
from types import SimpleNamespace

import pytest
import stripe
from fastapi import HTTPException
from sqlalchemy.exc import SQLAlchemyError

from src.chatbot.routes import billing


def test_get_or_create_customer_id_reads_stripe_object_metadata(monkeypatch):
    user = SimpleNamespace(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        email="user@example.com",
    )
    customer = stripe.Customer.construct_from(
        {
            "id": "cus_existing",
            "email": user.email,
            "metadata": {"tenant_id": str(user.tenant_id)},
        },
        key=None,
    )

    class CustomerApi:
        @staticmethod
        def list(**_kwargs):
            return SimpleNamespace(data=[customer])

        @staticmethod
        def create(**_kwargs):
            raise AssertionError("should reuse the matching Stripe customer")

    monkeypatch.setattr(billing.crud, "get_tenant_subscription", lambda _db, _tenant_id: None)
    monkeypatch.setattr(billing, "get_stripe_client", lambda: SimpleNamespace(Customer=CustomerApi))

    assert billing._get_or_create_customer_id(SimpleNamespace(), user) == "cus_existing"


def test_create_checkout_session_uses_fake_stripe_client(monkeypatch):
    user = SimpleNamespace(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        email="user@example.com",
    )
    db = SimpleNamespace()
    existing_customer = stripe.Customer.construct_from(
        {
            "id": "cus_existing",
            "email": user.email,
            "metadata": {"tenant_id": str(user.tenant_id)},
        },
        key=None,
    )
    checkout_calls = []
    upserts = []

    class CustomerApi:
        @staticmethod
        def list(**kwargs):
            assert kwargs == {"email": user.email, "limit": 10}
            return SimpleNamespace(data=[existing_customer])

        @staticmethod
        def create(**_kwargs):
            raise AssertionError("should reuse the matching Stripe customer")

    class CheckoutSessionApi:
        @staticmethod
        def create(**kwargs):
            checkout_calls.append(kwargs)
            return SimpleNamespace(
                id="cs_test_123",
                url="https://checkout.stripe.test/session/cs_test_123",
            )

    fake_stripe = SimpleNamespace(
        Customer=CustomerApi,
        checkout=SimpleNamespace(Session=CheckoutSessionApi),
    )

    def fake_upsert(*args, **kwargs):
        upserts.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace()

    monkeypatch.setenv("STRIPE_PRICE_PLUS_MULTIUSER", "price_multiuser_123")
    monkeypatch.setenv("STRIPE_TRIAL_DAYS", "7")
    monkeypatch.setenv("STRIPE_MULTIUSER_MIN_SEATS", "3")
    monkeypatch.setenv("STRIPE_AUTOMATIC_TAX_ENABLED", "false")
    monkeypatch.setattr(billing, "get_stripe_client", lambda: fake_stripe)
    monkeypatch.setattr(billing.crud, "get_tenant_subscription", lambda _db, _tenant_id: None)
    monkeypatch.setattr(billing.crud, "upsert_tenant_subscription", fake_upsert)

    response = billing.create_checkout_session(
        billing.CheckoutSessionRequest(
            plan_id="plus-multiuser",
            success_url="https://app.astrea.sunnit.ai/plans?billing=success",
            cancel_url="https://app.astrea.sunnit.ai/plans?billing=cancelled",
            source="fake-local-test",
            return_to="/plans",
            quantity=2,
        ),
        current_user=user,
        db=db,
    )

    assert response.checkout_url == "https://checkout.stripe.test/session/cs_test_123"
    assert response.session_id == "cs_test_123"
    assert checkout_calls == [
        {
            "mode": "subscription",
            "customer": "cus_existing",
            "client_reference_id": str(user.id),
            "success_url": "https://app.astrea.sunnit.ai/plans?billing=success",
            "cancel_url": "https://app.astrea.sunnit.ai/plans?billing=cancelled",
            "payment_method_collection": "always",
            "automatic_tax": {"enabled": False},
            "line_items": [{"price": "price_multiuser_123", "quantity": 3}],
            "metadata": {
                "user_id": str(user.id),
                "tenant_id": str(user.tenant_id),
                "plan_id": "plus-multiuser",
                "seats": "3",
                "source": "fake-local-test",
                "return_to": "/plans",
            },
            "subscription_data": {
                "metadata": {
                    "user_id": str(user.id),
                    "tenant_id": str(user.tenant_id),
                    "plan_id": "plus-multiuser",
                    "seats": "3",
                    "source": "fake-local-test",
                    "return_to": "/plans",
                },
                "trial_period_days": 7,
            },
        }
    ]
    assert len(upserts) == 1
    assert upserts[0]["args"][:2] == (db, user.tenant_id)
    assert upserts[0]["kwargs"]["stripe_customer_id"] == "cus_existing"
    assert upserts[0]["kwargs"]["stripe_checkout_session_id"] == "cs_test_123"
    assert upserts[0]["kwargs"]["status"] == "checkout_pending"
    assert upserts[0]["kwargs"]["seats"] == 3


def test_create_checkout_session_returns_503_when_billing_storage_is_unavailable(monkeypatch):
    user = SimpleNamespace(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        email="user@example.com",
    )

    class FakeDb:
        rolled_back = False

        def rollback(self):
            self.rolled_back = True

    db = FakeDb()

    def raise_storage_error(_db, _tenant_id):
        raise SQLAlchemyError("relation tenant_subscriptions does not exist")

    monkeypatch.setattr(billing.crud, "get_tenant_subscription", raise_storage_error)

    with pytest.raises(HTTPException) as exc_info:
        billing.create_checkout_session(
            billing.CheckoutSessionRequest(
                plan_id="plus-single-monthly",
                success_url="https://app.astrea.sunnit.ai/plans?billing=success",
                cancel_url="https://app.astrea.sunnit.ai/plans?billing=cancelled",
            ),
            current_user=user,
            db=db,
        )

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == "Billing storage unavailable. Run database migrations and retry checkout."
    assert db.rolled_back is True
