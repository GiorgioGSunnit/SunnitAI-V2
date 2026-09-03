"""Structured clarification payloads: when required inputs are missing or
invalid, the error's `details` must carry machine-actionable specs — an LLM
integration formulates the question from these, never by parsing prose."""

from app.main import engine
from app.schemas.calculation_request import CalculationRequest


def _calculate(calculator_id, inputs=None, **kwargs):
    return engine.calculate(CalculationRequest(
        calculator_id=calculator_id, inputs=inputs or {}, **kwargs
    ))


def test_missing_inputs_carry_full_specs():
    result = _calculate("legal_it.irpef", tax_year=2026)
    assert result.status == "error"
    error = result.errors[0]
    assert error.code == "input_invalid"
    assert error.details["missing_inputs"] == ["taxable_income"]
    (spec,) = error.details["missing"]
    assert spec["name"] == "taxable_income"
    assert spec["type"] == "decimal"
    assert spec["required"] is True
    assert spec["description"]  # human hint for the question
    assert spec["min_value"] == 0


def test_multiple_missing_inputs_each_get_a_spec():
    result = _calculate("legal_it.ravvedimento_operoso")
    error = result.errors[0]
    names = [s["name"] for s in error.details["missing"]]
    assert names == error.details["missing_inputs"]
    assert len(names) >= 2
    by_name = {s["name"]: s for s in error.details["missing"]}
    assert by_name["scadenza_originaria"]["type"] == "date"


def test_missing_period_is_a_structured_object_with_date_fields():
    result = _calculate("legal_it.legal_interest", inputs={"capital": 10000})
    error = result.errors[0]
    assert error.details["missing_inputs"] == ["period"]
    (spec,) = error.details["missing"]
    assert spec["name"] == "period"
    assert spec["type"] == "period"
    field_names = [f["name"] for f in spec["fields"]]
    assert field_names == ["start_date", "end_date"]
    assert all(f["type"] == "date" for f in spec["fields"])


def test_invalid_value_error_carries_the_expected_spec():
    result = _calculate("legal_it.irpef", inputs={"taxable_income": "abc"}, tax_year=2026)
    error = result.errors[0]
    assert error.code == "input_invalid"
    assert error.details["input"] == "taxable_income"
    assert error.details["expected"]["type"] == "decimal"
    assert error.details["expected"]["min_value"] == 0
