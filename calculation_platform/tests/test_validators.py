import pytest

from app.core.errors import InputValidationError
from app.core.validators import validate_inputs
from app.schemas.calculator_definition import CalculatorDefinition, InputSpec


def _definition(**overrides):
    inputs = overrides.pop("inputs", [InputSpec(name="amount", type="decimal", required=True)])
    return CalculatorDefinition(
        id="test.validators", name="v", category="test", strategy="expression",
        inputs=inputs, formula={"expression": "amount"}, **overrides,
    )


def test_valid_inputs_coerce_correctly():
    from decimal import Decimal

    definition = _definition()
    validated = validate_inputs(definition, {"amount": "123.45"})
    assert validated.values["amount"] == Decimal("123.45")
    assert validated.assumptions == []


def test_missing_required_input_raises_with_details():
    definition = _definition()
    with pytest.raises(InputValidationError) as exc_info:
        validate_inputs(definition, {})
    assert exc_info.value.code == "input_invalid"
    assert exc_info.value.details["missing_inputs"] == ["amount"]


def test_non_numeric_value_for_decimal_input_raises():
    definition = _definition()
    with pytest.raises(InputValidationError) as exc_info:
        validate_inputs(definition, {"amount": "not-a-number"})
    assert exc_info.value.details["input"] == "amount"


def test_optional_input_default_recorded_as_assumption():
    definition = _definition(inputs=[
        InputSpec(name="amount", type="decimal", required=True),
        InputSpec(name="discount", type="decimal", required=False, default=0),
    ])
    validated = validate_inputs(definition, {"amount": 100})
    assert validated.values["discount"] == 0
    assert len(validated.assumptions) == 1
    assert "discount" in validated.assumptions[0]


def test_omitted_optional_input_with_no_default_is_simply_absent():
    definition = _definition(inputs=[
        InputSpec(name="amount", type="decimal", required=True),
        InputSpec(name="note", type="string", required=False),
    ])
    validated = validate_inputs(definition, {"amount": 100})
    assert "note" not in validated.values
    assert validated.assumptions == []


def test_boolean_input_coerces_from_various_truthy_values():
    definition = _definition(inputs=[InputSpec(name="flag", type="boolean", required=True)])
    assert validate_inputs(definition, {"flag": True}).values["flag"] is True
    assert validate_inputs(definition, {"flag": False}).values["flag"] is False


@pytest.mark.parametrize("raw", ["false", "False", "FALSE", "no", "0", 0])
def test_boolean_false_spellings_coerce_to_false(raw):
    definition = _definition(inputs=[InputSpec(name="flag", type="boolean", required=True)])
    assert validate_inputs(definition, {"flag": raw}).values["flag"] is False


@pytest.mark.parametrize("raw", ["true", "True", "sì", "si", "yes", "1", 1])
def test_boolean_true_spellings_coerce_to_true(raw):
    definition = _definition(inputs=[InputSpec(name="flag", type="boolean", required=True)])
    assert validate_inputs(definition, {"flag": raw}).values["flag"] is True


@pytest.mark.parametrize("raw", ["falsey", "nope", "2", 2, 3.5, [], {}])
def test_boolean_unrecognized_value_raises(raw):
    definition = _definition(inputs=[InputSpec(name="flag", type="boolean", required=True)])
    with pytest.raises(InputValidationError) as exc_info:
        validate_inputs(definition, {"flag": raw})
    assert exc_info.value.details["input"] == "flag"


def test_date_input_coerces_from_iso_string():
    from datetime import date

    definition = _definition(inputs=[InputSpec(name="when", type="date", required=True)])
    validated = validate_inputs(definition, {"when": "2026-01-15"})
    assert validated.values["when"] == date(2026, 1, 15)


def test_min_value_rejects_out_of_range_input():
    definition = _definition(inputs=[InputSpec(name="amount", type="decimal", required=True, min_value=0)])
    with pytest.raises(InputValidationError) as exc_info:
        validate_inputs(definition, {"amount": -100})
    assert exc_info.value.details["input"] == "amount"
    assert exc_info.value.details["min_value"] == 0


def test_min_value_accepts_boundary_value():
    definition = _definition(inputs=[InputSpec(name="amount", type="decimal", required=True, min_value=0)])
    validated = validate_inputs(definition, {"amount": 0})
    assert validated.values["amount"] == 0


def test_max_value_rejects_out_of_range_input():
    definition = _definition(inputs=[InputSpec(name="amount", type="integer", required=True, max_value=100)])
    with pytest.raises(InputValidationError) as exc_info:
        validate_inputs(definition, {"amount": 101})
    assert exc_info.value.details["max_value"] == 100
