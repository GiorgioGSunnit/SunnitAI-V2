import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi.testclient import TestClient

from src.chatbot.api import app
from src.chatbot.routes import billing
from src.chatbot.routes.auth import get_current_user
from src.db.base import get_db


class _FakeQuery:
    def __init__(self, tenant):
        self._tenant = tenant

    def filter(self, *_args, **_kwargs):
        return self

    def first(self):
        return self._tenant


class _FakeDb:
    def __init__(self, tenant):
        self._tenant = tenant

    def query(self, _model):
        return _FakeQuery(self._tenant)


def test_sync_checkout_session_preserves_trialing_status_when_paid_session_lacks_embedded_subscription(
    monkeypatch,
):
    user = SimpleNamespace(
        id=uuid.uuid4(),
        tenant_id=uuid.uuid4(),
        email="user@example.com",
    )
    tenant = SimpleNamespace(
        id=user.tenant_id,
        plan="plus-single-monthly",
        subscription_end=datetime.now(timezone.utc) + timedelta(days=6),
    )
    existing_subscription = SimpleNamespace(
        tenant_id=user.tenant_id,
        user_id=user.id,
        plan_id="plus-single-monthly",
        status="trialing",
        seats=1,
        stripe_customer_id="cus_existing",
        stripe_subscription_id="sub_existing",
        stripe_checkout_session_id="cs_old",
        trial_started_at=datetime.now(timezone.utc) - timedelta(days=1),
        trial_ends_at=datetime.now(timezone.utc) + timedelta(days=6),
        current_period_end=datetime.now(timezone.utc) + timedelta(days=6),
        cancel_at_period_end=False,
        last_payment_status="paid",
    )
    checkout_session = SimpleNamespace(
        id="cs_paid_without_embedded_subscription",
        subscription=None,
        customer="cus_existing",
        payment_status="paid",
        metadata={
            "user_id": str(user.id),
            "tenant_id": str(user.tenant_id),
            "plan_id": "plus-single-monthly",
            "seats": "1",
        },
    )

    class CheckoutSessionApi:
        @staticmethod
        def retrieve(session_id, **kwargs):
            assert session_id == "cs_paid_without_embedded_subscription"
            assert kwargs == {"expand": ["subscription"]}
            return checkout_session

    def fake_upsert(_db, tenant_id_arg, **kwargs):
        merged = existing_subscription.__dict__.copy()
        merged["tenant_id"] = tenant_id_arg
        merged.update({key: value for key, value in kwargs.items() if value is not None})
        return SimpleNamespace(**merged)

    def override_current_user():
        return user

    def override_get_db():
        yield _FakeDb(tenant)

    monkeypatch.setattr(
        billing,
        "get_stripe_client",
        lambda: SimpleNamespace(checkout=SimpleNamespace(Session=CheckoutSessionApi)),
    )
    monkeypatch.setattr(billing.crud, "upsert_tenant_subscription", fake_upsert)
    monkeypatch.setattr(
        billing,
        "_resolve_local_subscription",
        lambda *_args, **_kwargs: existing_subscription,
    )
    monkeypatch.setattr(
        billing,
        "tenant_seat_usage",
        lambda *_args, **_kwargs: {"seats_used": 1, "seats_available": 0, "seats_over_limit": 0},
    )

    app.dependency_overrides[get_current_user] = override_current_user
    app.dependency_overrides[get_db] = override_get_db

    try:
        with TestClient(app) as client:
            response = client.post(
                "/api/billing/checkout/sessions/sync",
                json={"session_id": "cs_paid_without_embedded_subscription"},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "trialing"
    assert payload["is_active"] is True
    assert payload["trial_started_at"] is not None
    assert payload["trial_ends_at"] is not None
