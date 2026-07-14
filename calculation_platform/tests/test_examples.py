"""Runs every worked example declared in every formula pack's YAML through
the real engine and asserts it matches the declared expected_result.

This makes the `examples` metadata in each YAML self-verifying — if a
future change to a formula, a strategy, or a parameter table silently
changes a calculator's behavior, the corresponding example catches it,
without needing a bespoke test for every documented example.
"""

import pytest

from app.main import engine
from app.schemas.calculation_request import CalculationRequest


def _iter_examples():
    for calculator_id, definition in engine.registry._definitions.items():
        for index, example in enumerate(definition.examples):
            yield pytest.param(calculator_id, example, id=f"{calculator_id}[{index}]")


@pytest.mark.parametrize("calculator_id,example", list(_iter_examples()))
def test_declared_example_matches_engine_output(calculator_id, example):
    request = CalculationRequest(
        calculator_id=calculator_id,
        inputs=example.inputs,
        tax_year=example.tax_year,
        as_of_date=example.as_of_date,
        period=example.period,
        caller_supplied_values=example.caller_supplied_values,
    )
    result = engine.calculate(request)
    assert result.status == "success", f"{calculator_id} example {example.description!r} failed: {result.errors}"
    for key, expected_value in example.expected_result.items():
        assert result.result.get(key) == expected_value, (
            f"{calculator_id} example {example.description!r}: "
            f"expected {key}={expected_value}, got {result.result.get(key)}"
        )


def test_every_calculator_has_at_least_one_example():
    missing = [
        calculator_id
        for calculator_id, definition in engine.registry._definitions.items()
        if not definition.examples
    ]
    assert not missing, f"Calculators with no worked examples: {missing}"
