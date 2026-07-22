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
    stripe_subscription = SimpleNamespace(
        id="sub_123",
        status="active",
        created=1_752_496_800,
        latest_invoice=SimpleNamespace(status="paid", paid=True),
    )
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


def test_sync_tenant_billing_refreshes_account_from_stripe(monkeypatch):
    tenant_id = uuid.uuid4()
    local_subscription = SimpleNamespace(
        tenant_id=tenant_id,
        stripe_subscription_id="sub_123",
        stripe_customer_id="cus_123",
    )
    stripe_subscription = SimpleNamespace(
        id="sub_123",
        status="active",
        created=1_752_496_800,
        latest_invoice=SimpleNamespace(status="paid", paid=True),
    )
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
        def list(**kwargs):
            captured["list"] = kwargs
            return SimpleNamespace(data=[stripe_subscription])

        @staticmethod
        def retrieve(subscription_id, **kwargs):
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
        lambda _db, value, **kwargs: captured.update({"synced": value, "sync_kwargs": kwargs}),
    )
    monkeypatch.setattr(
        admin_routes,
        "_admin_account_for_tenant",
        lambda _db, _tenant_id: {
            "tenant_id": str(tenant_id),
            "stripe_status": "canceled",
            "canceled_at": "2026-07-21T14:00:00+00:00",
        },
    )

    response = admin_routes.sync_tenant_billing(
        tenant_id=tenant_id,
        _current_user=SimpleNamespace(role="superadmin"),
        db=SimpleNamespace(),
    )

    assert captured["subscription_id"] == "sub_123"
    assert captured["list"] == {"customer": "cus_123", "status": "all", "limit": 100}
    assert captured["expand"] == ["items.data.price", "latest_invoice"]
    assert captured["synced"] is stripe_subscription
    assert captured["sync_kwargs"]["last_payment_status"] == "paid"
    assert response["account"]["stripe_status"] == "canceled"
    assert response["account"]["canceled_at"] == "2026-07-21T14:00:00+00:00"


def test_sync_tenant_billing_discovers_new_active_subscription(monkeypatch):
    tenant_id = uuid.uuid4()
    local_subscription = SimpleNamespace(
        tenant_id=tenant_id,
        stripe_subscription_id="sub_old",
        stripe_customer_id="cus_123",
    )
    old_subscription = SimpleNamespace(id="sub_old", status="canceled", created=100)
    active_subscription = SimpleNamespace(
        id="sub_current",
        status="active",
        created=200,
        latest_invoice=None,
    )
    captured = {}

    monkeypatch.setattr(admin_routes.crud, "get_tenant_subscription", lambda *_: local_subscription)

    class FakeSubscriptionApi:
        @staticmethod
        def list(**_kwargs):
            return SimpleNamespace(data=[old_subscription, active_subscription])

        @staticmethod
        def retrieve(subscription_id, **_kwargs):
            captured["retrieved"] = subscription_id
            return active_subscription

    monkeypatch.setattr(
        admin_routes,
        "get_stripe_client",
        lambda: SimpleNamespace(Subscription=FakeSubscriptionApi),
    )
    monkeypatch.setattr(admin_routes, "_sync_from_stripe_subscription", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(admin_routes, "_admin_account_for_tenant", lambda *_: {"tenant_id": str(tenant_id)})

    admin_routes.sync_tenant_billing(
        tenant_id=tenant_id,
        _current_user=SimpleNamespace(role="superadmin"),
        db=SimpleNamespace(),
    )

    assert captured["retrieved"] == "sub_current"


def test_sync_tenant_billing_uses_stored_subscription_to_discover_customer(monkeypatch):
    tenant_id = uuid.uuid4()
    local_subscription = SimpleNamespace(
        tenant_id=tenant_id,
        stripe_subscription_id="sub_old",
        stripe_customer_id=None,
    )
    old_subscription = SimpleNamespace(
        id="sub_old",
        customer="cus_123",
        status="canceled",
        created=100,
        latest_invoice=None,
    )
    active_subscription = SimpleNamespace(
        id="sub_current",
        customer="cus_123",
        status="active",
        created=200,
        latest_invoice=None,
    )
    captured = {"retrieved": []}

    monkeypatch.setattr(
        admin_routes.crud,
        "get_tenant_subscription",
        lambda *_: local_subscription,
    )

    class FakeSubscriptionApi:
        @staticmethod
        def list(**kwargs):
            assert kwargs == {"customer": "cus_123", "status": "all", "limit": 100}
            return SimpleNamespace(data=[old_subscription, active_subscription])

        @staticmethod
        def retrieve(subscription_id, **_kwargs):
            captured["retrieved"].append(subscription_id)
            return old_subscription if subscription_id == "sub_old" else active_subscription

    monkeypatch.setattr(
        admin_routes,
        "get_stripe_client",
        lambda: SimpleNamespace(Subscription=FakeSubscriptionApi),
    )
    monkeypatch.setattr(
        admin_routes,
        "_sync_from_stripe_subscription",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        admin_routes,
        "_admin_account_for_tenant",
        lambda *_: {"tenant_id": str(tenant_id)},
    )

    admin_routes.sync_tenant_billing(
        tenant_id=tenant_id,
        _current_user=SimpleNamespace(role="superadmin"),
        db=SimpleNamespace(),
    )

    assert captured["retrieved"] == ["sub_old", "sub_current"]


def test_tenant_billing_history_returns_stripe_subscriptions_and_invoices(monkeypatch):
    tenant_id = uuid.uuid4()
    local_subscription = SimpleNamespace(stripe_customer_id="cus_123")

    monkeypatch.setattr(
        admin_routes.crud,
        "get_tenant_subscription",
        lambda _db, requested_tenant_id: local_subscription
        if requested_tenant_id == tenant_id
        else None,
    )

    class FakeSubscriptionApi:
        @staticmethod
        def list(**kwargs):
            assert kwargs == {"customer": "cus_123", "status": "all", "limit": 100}
            return SimpleNamespace(
                data=[
                    SimpleNamespace(
                        id="sub_old",
                        status="canceled",
                        created=1_752_496_800,
                        current_period_start=1_752_496_800,
                        current_period_end=1_755_175_200,
                        cancel_at_period_end=False,
                        cancel_at=None,
                        canceled_at=1_753_274_400,
                        ended_at=1_753_274_400,
                        items=SimpleNamespace(
                            data=[
                                SimpleNamespace(
                                    quantity=1,
                                    price=SimpleNamespace(id="price_old", lookup_key="plus-single"),
                                )
                            ]
                        ),
                    )
                ]
            )

    class FakeInvoiceApi:
        @staticmethod
        def list(**kwargs):
            assert kwargs == {"customer": "cus_123", "limit": 100}
            return SimpleNamespace(
                data=[
                    SimpleNamespace(
                        id="in_123",
                        number="AST-001",
                        status="paid",
                        created=1_752_496_800,
                        currency="eur",
                        amount_due=9900,
                        amount_paid=9900,
                        hosted_invoice_url="https://stripe.example/invoice",
                        invoice_pdf="https://stripe.example/invoice.pdf",
                        subscription="sub_old",
                    )
                ]
            )

    class FakePaymentIntentApi:
        @staticmethod
        def list(**kwargs):
            assert kwargs == {"customer": "cus_123", "limit": 100}
            return SimpleNamespace(
                data=[
                    SimpleNamespace(
                        id="pi_123",
                        status="succeeded",
                        created=1_752_496_800,
                        currency="eur",
                        amount=9900,
                        amount_received=9900,
                        description="Astrea Plus",
                        invoice="in_123",
                    )
                ]
            )

    monkeypatch.setattr(
        admin_routes,
        "get_stripe_client",
        lambda: SimpleNamespace(
            Subscription=FakeSubscriptionApi,
            Invoice=FakeInvoiceApi,
            PaymentIntent=FakePaymentIntentApi,
        ),
    )

    response = admin_routes.get_tenant_billing_history(
        tenant_id=tenant_id,
        _current_user=SimpleNamespace(role="superadmin"),
        db=SimpleNamespace(),
    )

    assert response["subscriptions"][0]["id"] == "sub_old"
    assert response["subscriptions"][0]["status"] == "canceled"
    assert response["subscriptions"][0]["canceled_at"] == "2025-07-23T12:40:00+00:00"
    assert response["invoices"][0]["id"] == "in_123"
    assert response["invoices"][0]["amount_paid"] == 9900
    assert response["invoices"][0]["subscription_id"] == "sub_old"
    assert response["payments"][0]["id"] == "pi_123"
    assert response["payments"][0]["amount_received"] == 9900


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
