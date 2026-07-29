"""Numeric inputs must not silently lose information, and counts must be bounded.

Two defects with the same shape — a value the platform cannot honestly
represent is accepted anyway:

  - a JSON float beyond 2**53 has already been rounded by the JSON parser
    before any code here sees it, so `Decimal(str(v))` faithfully records
    the WRONG number (9007199254740993.0 arrives as ...992);
  - an integer count with no declared maximum accepts 1e23, which the
    procedural-deadline strategy then walks day by day until date.max.
"""

import time

import pytest
from fastapi.testclient import TestClient

from app.core.errors import InputValidationError
from app.core.validators import _coerce_integer, validate_inputs
from app.main import app, engine

client = TestClient(app)

# 2**53 + 1: the smallest positive integer a float cannot represent.
UNREPRESENTABLE = 9007199254740993


# ---------------------------------------------------------------------------
# Finding C — a float that has already lost precision must be refused
# ---------------------------------------------------------------------------

def test_unrepresentable_float_is_rejected_not_silently_corrected():
    with pytest.raises(ValueError):
        _coerce_integer(float(UNREPRESENTABLE))


def test_unrepresentable_float_rejected_over_http():
    response = client.post("/calculate", json={
        "calculator_id": "legal_it.late_payment_interest",
        "inputs": {"capitale": 1, "tasso_riferimento_bce": 0, "giorni": float(UNREPRESENTABLE)},
    })
    body = response.json()
    assert body["status"] == "error", f"accepted a corrupted float: {body.get('inputs_used')}"
    assert body["errors"][0]["code"] == "input_invalid"
    assert str(UNREPRESENTABLE - 1) not in str(body.get("inputs_used"))


def test_exact_integer_forms_still_accepted():
    """The same magnitude sent losslessly stays valid — this is about
    representability, not about size."""
    assert _coerce_integer(UNREPRESENTABLE) == UNREPRESENTABLE
    assert _coerce_integer(str(UNREPRESENTABLE)) == UNREPRESENTABLE
    assert _coerce_integer(2.0) == 2
    assert _coerce_integer(1e15) == 10**15


# ---------------------------------------------------------------------------
# Finding D — unbounded counts permit CPU amplification
# ---------------------------------------------------------------------------

def test_absurd_day_count_is_rejected_by_validation_not_by_exhaustion():
    definition = engine.registry.get("legal_it.termini_processuali_civili")
    with pytest.raises(InputValidationError):
        validate_inputs(definition, {"data_decorrenza": "2026-01-15", "giorni": "1e23"})


def test_absurd_day_count_fails_fast_over_http():
    started = time.monotonic()
    response = client.post("/calculate", json={
        "calculator_id": "legal_it.termini_processuali_civili",
        "inputs": {"data_decorrenza": "2026-01-15", "giorni": "1e23"},
    })
    elapsed = time.monotonic() - started
    body = response.json()
    assert body["errors"][0]["code"] == "input_invalid", (
        f"reached the strategy instead of failing validation: {body['errors'][0]}"
    )
    assert elapsed < 0.1, f"took {elapsed:.3f}s — work was done before rejecting"


def test_realistic_deadlines_still_compute():
    """The bound must not touch any legally plausible term."""
    for giorni in (1, 30, 180, 3650):
        response = client.post("/calculate", json={
            "calculator_id": "legal_it.termini_processuali_civili",
            "inputs": {"data_decorrenza": "2026-01-15", "giorni": giorni},
        })
        assert response.json()["status"] == "success", f"{giorni} days rejected"


def test_loan_months_and_circumstance_counts_are_bounded():
    months = client.post("/calculate", json={
        "calculator_id": "business.loan_payment",
        "inputs": {"principal": 100000, "annual_rate": 0.06, "months": 10**9},
    }).json()
    assert months["errors"][0]["code"] == "input_invalid"

    definition = engine.registry.get("business.loan_payment")
    assert validate_inputs(definition, {
        "principal": 100000, "annual_rate": 0.06, "months": 480,
    }).values["months"] == 480
