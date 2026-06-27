import uuid
from types import SimpleNamespace

import pytest
import stripe

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
