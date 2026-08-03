"""Validation and monetary-precision guarantees."""

from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import StatementError

from policy_comparator.models import NormalizedQuote, QuoteRequest
from policy_comparator.schemas.profile import (
    CoveragePreferenceData,
    CustomerProfileData,
    QuotationProfile,
    VehicleData,
)
from policy_comparator.schemas.quotes import CoverageData, NormalizedQuoteData
from policy_comparator.tests.conftest import TENANT_A


def _profile(**overrides) -> QuotationProfile:
    base = dict(
        customer_email="cliente@esempio.it",
        policy_start_date=date(2026, 9, 1),
        customer=CustomerProfileData(owner_date_of_birth=date(1985, 3, 4)),
        vehicle=VehicleData(plate="AB123CD"),
    )
    base.update(overrides)
    return QuotationProfile(**base)


class TestPlateNormalization:
    def test_separators_and_case_are_stripped(self):
        assert VehicleData(plate="ab 123-cd").plate == "AB123CD"

    def test_too_short_is_rejected(self):
        with pytest.raises(ValidationError):
            VehicleData(plate="AB1")

    def test_too_long_is_rejected(self):
        with pytest.raises(ValidationError):
            VehicleData(plate="ABCDEFGHIJKL")


class TestProfilePaths:
    def test_get_path_reads_nested_values(self):
        profile = _profile()
        assert profile.get_path("vehicle.plate") == "AB123CD"
        assert profile.get_path("customer.owner_date_of_birth") == date(1985, 3, 4)

    def test_unknown_path_is_none_not_an_error(self):
        # An adapter asking about a field we do not model must not crash a run.
        assert _profile().get_path("vehicle.nonexistent.deep") is None

    def test_blank_string_counts_as_missing(self):
        profile = _profile(customer=CustomerProfileData(first_name="   "))
        assert profile.has_path("customer.first_name") is False

    def test_missing_paths_reports_only_absent_fields(self):
        profile = _profile()
        missing = profile.missing_paths(["vehicle.plate", "vehicle.make", "customer.tax_code"])
        assert missing == ["vehicle.make", "customer.tax_code"]


class TestMoneyIsNeverFloat:
    def test_quote_rejects_a_float_premium(self):
        with pytest.raises(ValidationError):
            NormalizedQuoteData(
                provider_id="zurich", insurer_name="Zurich", annual_total_premium=342.31
            )

    def test_coverage_rejects_a_float_price(self):
        with pytest.raises(ValidationError):
            CoverageData(code="kasko", label="Kasko", price=395.0)

    def test_preference_limits_reject_floats(self):
        with pytest.raises(ValidationError):
            CoveragePreferenceData(max_acceptable_deductible=500.0)

    def test_decimal_strings_are_accepted_exactly(self):
        quote = NormalizedQuoteData(
            provider_id="zurich", insurer_name="Zurich", annual_total_premium="342.31"
        )
        assert quote.annual_total_premium == Decimal("342.31")


def _request_and_attempt(db):
    """Real request and attempt rows, so the foreign keys hold."""
    from policy_comparator.crypto import blind_index
    from policy_comparator.models import Customer, ProviderAttempt

    customer = Customer(
        tenant_id=TENANT_A,
        email="cliente@esempio.it",
        email_fingerprint=blind_index("cliente@esempio.it"),
    )
    db.add(customer)
    db.flush()

    request = QuoteRequest(
        tenant_id=TENANT_A,
        customer_id=customer.id,
        customer_profile_id=uuid.uuid4(),
        vehicle_id=uuid.uuid4(),
        insurance_history_id=uuid.uuid4(),
        coverage_preference_id=uuid.uuid4(),
        policy_start_date=date(2026, 9, 1),
    )
    db.add(request)
    db.flush()

    attempt = ProviderAttempt(
        tenant_id=TENANT_A,
        quote_request_id=request.id,
        provider_id="zurich",
        idempotency_key="test-key",
    )
    db.add(attempt)
    db.flush()
    return request, attempt


class TestMoneyColumnRoundTrip:
    def test_decimal_survives_the_database(self, db):
        request, attempt = _request_and_attempt(db)

        # Values with cents are exactly where binary floats drift.
        quote = NormalizedQuote(
            tenant_id=TENANT_A,
            quote_request_id=request.id,
            provider_attempt_id=attempt.id,
            provider_id="zurich",
            insurer_name="Zurich",
            annual_total_premium=Decimal("1234.55"),
            deductible=Decimal("0.30"),
        )
        db.add(quote)
        db.commit()
        db.expire_all()

        stored = db.get(NormalizedQuote, quote.id)
        assert stored.annual_total_premium == Decimal("1234.55")
        assert stored.annual_total_premium + stored.deductible == Decimal("1234.85")

    def test_float_is_refused_at_the_column(self, db):
        request, attempt = _request_and_attempt(db)

        db.add(
            NormalizedQuote(
                tenant_id=TENANT_A,
                quote_request_id=request.id,
                provider_attempt_id=attempt.id,
                provider_id="zurich",
                insurer_name="Zurich",
                annual_total_premium=342.31,
            )
        )
        # SQLAlchemy wraps the column's TypeError in a StatementError, but the
        # write is still refused rather than silently rounded.
        with pytest.raises(StatementError) as exc_info:
            db.flush()
        assert isinstance(exc_info.value.orig, TypeError)
        db.rollback()


class TestEncryptedColumns:
    def test_email_is_ciphertext_at_rest_but_plaintext_in_python(self, db):
        from sqlalchemy import text

        from policy_comparator.crypto import blind_index
        from policy_comparator.models import Customer

        customer = Customer(
            tenant_id=TENANT_A,
            email="cliente@esempio.it",
            email_fingerprint=blind_index("cliente@esempio.it"),
        )
        db.add(customer)
        db.commit()

        raw = db.execute(
            text("SELECT email FROM pc_customers WHERE id = :id"),
            {"id": str(customer.id)},
        ).scalar_one()
        assert "cliente@esempio.it" not in raw
        assert raw.startswith("pcenc1:")

        db.expire_all()
        assert db.get(Customer, customer.id).email == "cliente@esempio.it"

    def test_blind_index_is_stable_and_case_insensitive(self):
        from policy_comparator.crypto import blind_index

        assert blind_index("Cliente@Esempio.IT ") == blind_index("cliente@esempio.it")
        assert blind_index("a@b.it") != blind_index("c@d.it")
