import asyncio
import uuid
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from src.chatbot import billing
from src.chatbot.routes import billing as billing_routes


def _subscription(**overrides):
    now = datetime.now(timezone.utc)
    base = {
        "tenant_id": uuid.uuid4(),
        "user_id": uuid.uuid4(),
        "plan_id": "plus-single-monthly",
        "status": "trialing",
        "seats": 1,
        "stripe_customer_id": "cus_test",
        "stripe_subscription_id": "sub_test",
        "stripe_checkout_session_id": "cs_test",
        "trial_started_at": now - timedelta(days=2),
        "trial_ends_at": now + timedelta(days=5),
        "current_period_end": now + timedelta(days=5),
        "cancel_at_period_end": False,
        "last_payment_status": None,
        "admin_access_override": None,
        "admin_access_until": None,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def test_subscription_access_reason_allows_active_and_trialing_states():
    assert billing.subscription_access_block_reason(_subscription(status="trialing")) is None
    assert billing.subscription_access_block_reason(_subscription(status="active")) is None


def test_subscription_access_reason_blocks_expired_trial_and_failed_payment():
    expired_trial = _subscription(trial_ends_at=datetime.now(timezone.utc) - timedelta(seconds=1))
    failed_payment = _subscription(status="active", last_payment_status="failed")

    assert "prova gratuita" in billing.subscription_access_block_reason(expired_trial)
    assert "Pagamento non riuscito" in billing.subscription_access_block_reason(failed_payment)


def test_subscription_access_reason_blocks_canceled_and_unpaid_states():
    assert "Abbonamento non attivo" in billing.subscription_access_block_reason(
        _subscription(status="canceled")
    )
    assert "Pagamento richiesto" in billing.subscription_access_block_reason(
        _subscription(status="past_due")
    )


def test_subscription_access_reason_blocks_when_subscription_is_missing():
    assert "Nessun piano attivo" in billing.subscription_access_block_reason(None)


def test_admin_override_can_allow_or_block_access_independently_from_stripe():
    allowed = _subscription(
        status="inactive",
        admin_access_override="allowed",
        admin_access_until=datetime.now(timezone.utc) + timedelta(days=5),
    )
    blocked = _subscription(status="active", admin_access_override="blocked")

    assert billing.subscription_access_block_reason(allowed) is None
    assert "sospeso manualmente" in billing.subscription_access_block_reason(blocked)


def test_expired_admin_override_falls_back_to_stripe_status():
    expired = _subscription(
        status="inactive",
        admin_access_override="allowed",
        admin_access_until=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    assert "Abbonamento non attivo" in billing.subscription_access_block_reason(expired)


def test_subscription_access_reason_blocks_abandoned_checkout_pending():
    pending_checkout = _subscription(
        status="checkout_pending",
        stripe_subscription_id=None,
        trial_started_at=None,
        trial_ends_at=None,
        current_period_end=None,
        last_payment_status=None,
    )

    assert billing.subscription_access_block_reason(pending_checkout) == (
        "Checkout non completato. Apri Piani per attivare la prova gratuita."
    )
    assert billing.subscription_allows_access(
        pending_checkout.status,
        billing.subscription_access_block_reason(pending_checkout),
    ) is False


def test_sync_from_stripe_subscription_preserves_trial_cancellation_until_expiry(monkeypatch):
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    stripe_subscription = SimpleNamespace(
        id="sub_trial_cancel",
        customer="cus_trial_cancel",
        status="trialing",
        trial_start=int((now - timedelta(days=2)).timestamp()),
        trial_end=int((now + timedelta(days=2)).timestamp()),
        current_period_end=int((now + timedelta(days=2)).timestamp()),
        cancel_at_period_end=True,
        cancel_at=int((now + timedelta(days=2)).timestamp()),
        canceled_at=int((now - timedelta(minutes=10)).timestamp()),
        ended_at=None,
        items=SimpleNamespace(data=[SimpleNamespace(quantity=1)]),
        metadata={
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "plan_id": "plus-single-monthly",
        },
    )

    def fake_upsert(_db, tenant_id_arg, **kwargs):
        return SimpleNamespace(tenant_id=tenant_id_arg, **kwargs)

    monkeypatch.setattr(billing.crud, "get_tenant_subscription_by_stripe_subscription_id", lambda *_: None)
    monkeypatch.setattr(billing.crud, "get_tenant_subscription_by_checkout_session_id", lambda *_: None)
    monkeypatch.setattr(billing.crud, "get_tenant_subscription_by_customer_id", lambda *_: None)
    monkeypatch.setattr(billing.crud, "get_tenant_subscription", lambda *_: None)
    monkeypatch.setattr(billing.crud, "upsert_tenant_subscription", fake_upsert)

    result = billing_routes._sync_from_stripe_subscription(SimpleNamespace(), stripe_subscription)

    assert result.status == "trialing"
    assert result.cancel_at_period_end is True
    assert result.cancel_at == billing.utc_from_unix(stripe_subscription.cancel_at)
    assert result.canceled_at == billing.utc_from_unix(stripe_subscription.canceled_at)
    assert result.ended_at is None
    assert result.trial_ends_at is not None
    assert billing.subscription_access_block_reason(result) is None


def test_sync_from_stripe_subscription_marks_failed_payment(monkeypatch):
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    stripe_subscription = SimpleNamespace(
        id="sub_failed_payment",
        customer="cus_failed_payment",
        status="active",
        trial_start=None,
        trial_end=None,
        current_period_end=int((now + timedelta(days=30)).timestamp()),
        cancel_at_period_end=False,
        items=SimpleNamespace(data=[SimpleNamespace(quantity=1)]),
        metadata={
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "plan_id": "plus-single-monthly",
        },
    )
    captured = {}

    def fake_upsert(_db, tenant_id_arg, **kwargs):
        captured.update({"tenant_id": tenant_id_arg, **kwargs})
        return SimpleNamespace(tenant_id=tenant_id_arg, **kwargs)

    monkeypatch.setattr(billing.crud, "get_tenant_subscription_by_stripe_subscription_id", lambda *_: None)
    monkeypatch.setattr(billing.crud, "get_tenant_subscription_by_checkout_session_id", lambda *_: None)
    monkeypatch.setattr(billing.crud, "get_tenant_subscription_by_customer_id", lambda *_: None)
    monkeypatch.setattr(billing.crud, "get_tenant_subscription", lambda *_: None)
    monkeypatch.setattr(billing.crud, "upsert_tenant_subscription", fake_upsert)

    result = billing_routes._sync_from_stripe_subscription(
        SimpleNamespace(),
        stripe_subscription,
        last_payment_status="failed",
    )

    assert captured["status"] == "active"
    assert captured["last_payment_status"] == "failed"
    assert result.last_payment_status == "failed"
    assert billing.subscription_access_block_reason(result) == (
        "Pagamento non riuscito. Aggiorna il metodo di pagamento per continuare a usare Astrea."
    )


def test_stripe_webhook_handles_deleted_subscription(monkeypatch):
    tenant_id = uuid.uuid4()
    user_id = uuid.uuid4()
    now = datetime.now(timezone.utc)
    stripe_subscription = SimpleNamespace(
        id="sub_deleted",
        customer="cus_deleted",
        status="canceled",
        trial_start=None,
        trial_end=None,
        current_period_end=int((now - timedelta(days=1)).timestamp()),
        cancel_at_period_end=False,
        items=SimpleNamespace(data=[SimpleNamespace(quantity=1)]),
        metadata={
            "tenant_id": str(tenant_id),
            "user_id": str(user_id),
            "plan_id": "plus-single-monthly",
        },
    )
    captured = {}

    def fake_upsert(_db, tenant_id_arg, **kwargs):
        captured.update({"tenant_id": tenant_id_arg, **kwargs})
        return SimpleNamespace(tenant_id=tenant_id_arg, **kwargs)

    fake_stripe = SimpleNamespace(
        Webhook=SimpleNamespace(
            construct_event=lambda _payload, _signature, _secret: {
                "type": "customer.subscription.deleted",
                "data": {"object": stripe_subscription},
            }
        ),
        Subscription=SimpleNamespace(retrieve=lambda *_args, **_kwargs: stripe_subscription),
    )

    monkeypatch.setattr(billing_routes, "get_stripe_client", lambda: fake_stripe)
    monkeypatch.setattr(billing_routes, "get_webhook_secret", lambda: "whsec_test")
    monkeypatch.setattr(billing.crud, "get_tenant_subscription_by_stripe_subscription_id", lambda *_: None)
    monkeypatch.setattr(billing.crud, "get_tenant_subscription_by_checkout_session_id", lambda *_: None)
    monkeypatch.setattr(billing.crud, "get_tenant_subscription_by_customer_id", lambda *_: None)
    monkeypatch.setattr(billing.crud, "get_tenant_subscription", lambda *_: None)
    monkeypatch.setattr(billing.crud, "upsert_tenant_subscription", fake_upsert)

    request = SimpleNamespace(
        body=lambda: asyncio.sleep(0, result=b"{}"),
        headers={"stripe-signature": "sig_test"},
    )

    result = asyncio.run(billing_routes.stripe_webhook(request, db=SimpleNamespace()))

    assert result["received"] is True
    assert captured["status"] == "canceled"
    assert billing.subscription_access_block_reason(
        SimpleNamespace(**captured)
    ) == "Abbonamento non attivo. Riattiva un piano per continuare a usare Astrea."
