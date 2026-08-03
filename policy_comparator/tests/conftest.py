"""Shared fixtures.

Every test gets a clean schema, so no test depends on another's leftovers.
"""

from __future__ import annotations

import uuid
from datetime import date, timedelta

import pytest
from fastapi.testclient import TestClient

from policy_comparator.api import deps
from policy_comparator.api.app import create_app
from policy_comparator.db import Base, create_all, get_engine, session_scope
from policy_comparator.models import StaffUser
from policy_comparator.security import StaffIdentity, create_access_token, hash_password

TENANT_A = uuid.UUID("aaaaaaaa-0000-4000-8000-000000000001")
TENANT_B = uuid.UUID("bbbbbbbb-0000-4000-8000-000000000002")


@pytest.fixture(autouse=True)
def fresh_schema():
    """Drop and recreate every table around each test."""
    Base.metadata.drop_all(bind=get_engine())
    create_all()
    deps.reset_rate_limits()
    yield
    Base.metadata.drop_all(bind=get_engine())


@pytest.fixture
def db():
    session = session_scope()
    try:
        yield session
        session.commit()
    finally:
        session.close()


@pytest.fixture
def client():
    with TestClient(create_app()) as test_client:
        yield test_client


def make_identity(tenant_id: uuid.UUID = TENANT_A, role: str = "admin") -> StaffIdentity:
    return StaffIdentity(
        user_id=uuid.uuid4(), tenant_id=tenant_id, email="staff@example.com", role=role
    )


def auth_headers(identity: StaffIdentity | None = None) -> dict[str, str]:
    identity = identity or make_identity()
    return {"Authorization": f"Bearer {create_access_token(identity)}"}


@pytest.fixture
def identity() -> StaffIdentity:
    return make_identity()


@pytest.fixture
def headers(identity: StaffIdentity) -> dict[str, str]:
    return auth_headers(identity)


@pytest.fixture
def staff_user(db) -> StaffUser:
    user = StaffUser(
        tenant_id=TENANT_A,
        email="staff@example.com",
        hashed_password=hash_password("correct-horse"),
        role="admin",
    )
    db.add(user)
    db.commit()
    return user


@pytest.fixture
def new_request_body() -> dict:
    """The minimal initial form, with both mandatory consents granted."""
    return {
        "vehicle_plate": "AB123CD",
        "owner_date_of_birth": "1985-03-04",
        "customer_email": "cliente@esempio.it",
        "policy_start_date": (date.today() + timedelta(days=7)).isoformat(),
        "privacy_accepted": True,
        "provider_data_transfer_accepted": True,
        "selected_provider_ids": ["zurich", "allianz", "generali", "cercassicurazioni"],
    }


#: Answers to everything the four mock providers ask for in their second stage.
FULL_ANSWERS = {
    "customer.tax_code": "RSSMRA85C04H501Z",
    "customer.municipality": "Roma",
    "customer.postcode": "00184",
    "customer.first_name": "Mario",
    "customer.last_name": "Rossi",
    "customer.mobile_number": "3331234567",
    "vehicle.first_registration_date": "2019-05-10",
    "vehicle.make": "Fiat",
    "vehicle.model": "Panda",
    "history.universal_merit_class": "3",
    "preferences.driving_formula": "expert",
}
