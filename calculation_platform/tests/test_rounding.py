from decimal import Decimal

import pytest

from app.core.definition_validator import validate_definition
from app.core.errors import DefinitionValidationError
from app.core.result_builder import round_decimal, round_output
from app.schemas.calculator_definition import CalculatorDefinition, InputSpec


def test_default_mode_is_half_up():
    assert round_decimal(Decimal("2.345")) == Decimal("2.35")
    assert round_decimal(Decimal("2.5"), 0) == Decimal("3")


@pytest.mark.parametrize(
    "mode, value, places, expected",
    [
        ("half_up", "2.345", 2, "2.35"),
        ("half_even", "2.345", 2, "2.34"),
        ("down", "2.349", 2, "2.34"),
        ("down", "-2.349", 2, "-2.34"),
        ("up", "2.341", 2, "2.35"),
        ("floor", "-2.341", 2, "-2.35"),
        ("ceiling", "2.341", 2, "2.35"),
        # Tax rounding to the whole euro (e.g. truncation vs half-up differ)
        ("down", "99.99", 0, "99"),
        ("half_up", "99.50", 0, "100"),
    ],
)
def test_rounding_modes(mode, value, places, expected):
    assert round_decimal(Decimal(value), places, mode) == Decimal(expected)


def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        round_decimal(Decimal("1"), 2, "banker")


def test_round_output_reads_output_spec():
    assert round_output(Decimal("2.345"), {"round_to": 2, "rounding": "half_even"}) == Decimal("2.34")
    assert round_output(Decimal("2.345"), {}) == Decimal("2.35")


def test_definition_with_unknown_rounding_mode_fails_at_load():
    definition = CalculatorDefinition(
        id="test.rounding", name="r", category="test", strategy="expression",
        inputs=[InputSpec(name="amount", type="decimal", required=True)],
        formula={"expression": "amount"},
        output={"name": "result", "round_to": 2, "rounding": "bankers"},
    )
    with pytest.raises(DefinitionValidationError):
        validate_definition(definition)


def test_definition_with_valid_rounding_mode_loads():
    definition = CalculatorDefinition(
        id="test.rounding", name="r", category="test", strategy="expression",
        inputs=[InputSpec(name="amount", type="decimal", required=True)],
        formula={"expression": "amount"},
        output={"name": "result", "round_to": 0, "rounding": "down"},
    )
    validate_definition(definition)
