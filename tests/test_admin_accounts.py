import uuid
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace

from src.chatbot.routes import admin as admin_routes


def test_inclusive_access_date_bills_from_next_day_in_rome():
    trial_end = admin_routes.inclusive_access_date_to_trial_end(
        date(2026, 9, 15),
        now=datetime(2026, 7, 21, 10, 0, tzinfo=timezone.utc),
    )

    assert trial_end.isoformat() == "2026-09-15T22:00:00+00:00"


def test_extend_trial_updates_stripe_and_persists_returned_state(monkeypatch):
    tenant_id = uuid.uuid4()
    local_subscription = SimpleNamespace(
        tenant_id=tenant_id,
        stripe_subscription_id="sub_123",
        trial_ends_at=None,
        current_period_end=None,
        cancel_at_period_end=False,
        status="trialing",
    )
    stripe_subscription = SimpleNamespace(id="sub_123")
    captured = {}

    monkeypatch.setattr(
        admin_routes.crud,
        "get_tenant_subscription",
        lambda _db, requested_tenant_id: local_subscription
        if requested_tenant_id == tenant_id
        else None,
    )

    class FakeSubscriptionApi:
        @staticmethod
        def modify(subscription_id, **kwargs):
            captured.update({"subscription_id": subscription_id, **kwargs})
            return stripe_subscription

    monkeypatch.setattr(
        admin_routes,
        "get_stripe_client",
        lambda: SimpleNamespace(Subscription=FakeSubscriptionApi),
    )
    monkeypatch.setattr(
        admin_routes,
        "_sync_from_stripe_subscription",
        lambda _db, value, **_kwargs: value,
    )
    monkeypatch.setattr(
        admin_routes,
        "_admin_account_for_tenant",
        lambda _db, _tenant_id: {"tenant_id": str(tenant_id), "stripe_status": "trialing"},
    )
    monkeypatch.setattr(
        admin_routes,
        "_set_access_override",
        lambda _db, _tenant_id, mode, access_until: captured.update(
            {"override": mode, "access_until": access_until}
        ),
    )

    access_through = date.today() + timedelta(days=45)
    expected_trial_end = admin_routes.inclusive_access_date_to_trial_end(access_through)

    response = admin_routes.extend_tenant_trial(
        tenant_id=tenant_id,
        request=admin_routes.TrialExtensionRequest(access_through=access_through),
        _current_user=SimpleNamespace(role="superadmin"),
        db=SimpleNamespace(),
    )

    assert captured["subscription_id"] == "sub_123"
    assert captured["trial_end"] == int(expected_trial_end.timestamp())
    assert captured["proration_behavior"] == "none"
    assert captured["override"] == "allowed"
    assert captured["access_until"] == expected_trial_end
    assert response["account"]["stripe_status"] == "trialing"


def test_extend_trial_grants_local_access_without_a_stripe_subscription(monkeypatch):
    tenant_id = uuid.uuid4()
    captured = {}
    monkeypatch.setattr(admin_routes.crud, "get_tenant_subscription", lambda *_args: None)
    monkeypatch.setattr(
        admin_routes,
        "_set_access_override",
        lambda _db, _tenant_id, mode, access_until: captured.update(
            {"mode": mode, "access_until": access_until}
        ),
    )
    monkeypatch.setattr(
        admin_routes,
        "_admin_account_for_tenant",
        lambda _db, _tenant_id: {"tenant_id": str(tenant_id), "has_access": True},
    )

    response = admin_routes.extend_tenant_trial(
        tenant_id=tenant_id,
        request=admin_routes.TrialExtensionRequest(
            access_through=date.today() + timedelta(days=30)
        ),
        _current_user=SimpleNamespace(role="superadmin"),
        db=SimpleNamespace(),
    )

    assert captured["mode"] == "allowed"
    assert response["account"]["has_access"] is True


def test_admin_account_marks_an_inactive_user_as_blocked():
    now = datetime.now(timezone.utc)
    account = admin_routes._serialize_admin_account(
        SimpleNamespace(
            id=uuid.uuid4(),
            email="inactive@example.com",
            role="member",
            is_active=False,
            created_at=now,
            last_login=None,
        ),
        SimpleNamespace(id=uuid.uuid4(), email="studio@example.com", is_active=True, plan="plus"),
        SimpleNamespace(display_name="Studio Test"),
        SimpleNamespace(
            plan_id="plus-single-monthly",
            status="active",
            stripe_customer_id="cus_123",
            stripe_subscription_id="sub_123",
            trial_started_at=None,
            trial_ends_at=None,
            current_period_end=now + timedelta(days=30),
            cancel_at_period_end=False,
            last_payment_status="paid",
            updated_at=now,
        ),
    )

    assert account["has_access"] is False
    assert account["access_block_reason"] == "Utente disattivato"
