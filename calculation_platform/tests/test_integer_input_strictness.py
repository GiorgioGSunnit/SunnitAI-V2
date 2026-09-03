"""An input declared `integer` must reject a fractional value, not truncate it.

`int(1.9)` is 1. On a procedural deadline that silently turns a 2-day term
into a 1-day term — a wrong legal answer delivered with full confidence and
no warning. The same coercer backs late-interest day counts, loan months and
penal circumstance counts, so the rejection is enforced at the validator
rather than per calculator.
"""

from decimal import Decimal

import pytest

from app.core.errors import InputValidationError
from app.core.validators import validate_inputs
from app.main import engine
from app.schemas.calculation_request import CalculationRequest
from app.schemas.calculator_definition import CalculatorDefinition, InputSpec

TERMINI_ID = "legal_it.termini_processuali_civili"


def _definition() -> CalculatorDefinition:
    return CalculatorDefinition(
        id="test.integer_strictness", name="int", category="test", strategy="expression",
        inputs=[InputSpec(name="n", type="integer", required=True)],
        formula={"expression": "n"},
        output={"name": "result"},
    )


def _coerce(value):
    return validate_inputs(_definition(), {"n": value}).values["n"]


# ---------------------------------------------------------------------------
# Rejected: anything whose integer form loses information
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value", [1.9, "1.9", -0.5, 2.0001, Decimal("1.9"), "1,9"])
def test_fractional_values_are_rejected(value):
    with pytest.raises(InputValidationError) as excinfo:
        _coerce(value)
    assert excinfo.value.code == "input_invalid"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf"), "NaN", "Infinity"])
def test_non_finite_values_are_rejected(value):
    # int(float("inf")) raises OverflowError, which _coerce_scalar does not
    # catch — it used to escape the engine as an unhandled 500.
    with pytest.raises(InputValidationError):
        _coerce(value)


@pytest.mark.parametrize("value", [True, False])
def test_booleans_are_not_silently_counted_as_1_or_0(value):
    with pytest.raises(InputValidationError):
        _coerce(value)


@pytest.mark.parametrize("value", ["", "abc", None, [2], {"n": 2}])
def test_non_numeric_values_are_rejected(value):
    with pytest.raises(InputValidationError):
        _coerce(value)


def test_truncation_no_longer_reaches_a_procedural_deadline():
    result = engine.calculate(CalculationRequest(
        calculator_id=TERMINI_ID,
        inputs={"data_decorrenza": "2026-03-02", "giorni": 1.9},
    ))
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"
    assert "giorni" in result.errors[0].message


# ---------------------------------------------------------------------------
# Accepted: values that are exactly an integer
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    (2, 2), (2.0, 2), ("2", 2), (" 2 ", 2), (Decimal("2"), 2), (Decimal("2.00"), 2), (-3, -3),
])
def test_exact_integers_are_accepted(value, expected):
    coerced = _coerce(value)
    assert coerced == expected
    assert isinstance(coerced, int)


def test_whole_day_count_still_computes():
    result = engine.calculate(CalculationRequest(
        calculator_id=TERMINI_ID,
        inputs={"data_decorrenza": "2026-03-02", "giorni": 20},
    ))
    assert result.status == "success"
