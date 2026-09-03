from decimal import Decimal

from app.resolvers.parameter_store import ParameterStore
from app.schemas.calculation_request import CalculationRequest
from app.schemas.calculator_definition import CalculatorDefinition, InputSpec
from app.strategies.expression import ExpressionStrategy
from pathlib import Path

PARAMETERS_DIR = Path(__file__).resolve().parent.parent / "parameters"


def _strategy():
    return ExpressionStrategy(ParameterStore(PARAMETERS_DIR))


def test_output_clamps_to_minimum():
    definition = CalculatorDefinition(
        id="test.clamp_min", name="clamp", category="test", strategy="expression",
        inputs=[InputSpec(name="a", type="decimal", required=True), InputSpec(name="b", type="decimal", required=True)],
        formula={"expression": "a - b"},
        output={"name": "result", "round_to": 2, "min": 0},
    )
    outcome = _strategy().run(definition, {"a": Decimal("10"), "b": Decimal("50")}, CalculationRequest(calculator_id="test.clamp_min"))
    assert outcome.result["result"] == Decimal("0.00")
    assert any(s["type"] == "clamped_to_minimum" for s in outcome.steps)


def test_output_clamps_to_maximum():
    definition = CalculatorDefinition(
        id="test.clamp_max", name="clamp", category="test", strategy="expression",
        inputs=[InputSpec(name="a", type="decimal", required=True)],
        formula={"expression": "a"},
        output={"name": "result", "round_to": 2, "max": 100},
    )
    outcome = _strategy().run(definition, {"a": Decimal("500")}, CalculationRequest(calculator_id="test.clamp_max"))
    assert outcome.result["result"] == Decimal("100.00")
    assert any(s["type"] == "clamped_to_maximum" for s in outcome.steps)


def test_output_within_bounds_is_unaffected():
    definition = CalculatorDefinition(
        id="test.clamp_none", name="clamp", category="test", strategy="expression",
        inputs=[InputSpec(name="a", type="decimal", required=True)],
        formula={"expression": "a"},
        output={"name": "result", "round_to": 2, "min": 0, "max": 100},
    )
    outcome = _strategy().run(definition, {"a": Decimal("50")}, CalculationRequest(calculator_id="test.clamp_none"))
    assert outcome.result["result"] == Decimal("50.00")
    assert not any(s["type"].startswith("clamped") for s in outcome.steps)


def test_round_in_formula_flows_through_strategy():
    # One-arg round() used to return an int (Decimal.__round__ contract),
    # which crashed round_decimal's quantize with AttributeError; the tie at
    # 2.5 also discriminates half_up (3) from half-even (2).
    definition = CalculatorDefinition(
        id="test.round_expr", name="round", category="test", strategy="expression",
        inputs=[InputSpec(name="a", type="decimal", required=True)],
        formula={"expression": "round(a)"},
        output={"name": "result", "round_to": 2},
    )
    outcome = _strategy().run(definition, {"a": Decimal("2.5")}, CalculationRequest(calculator_id="test.round_expr"))
    assert outcome.result["result"] == Decimal("3.00")


def test_chained_derived_variables_evaluate_in_order():
    definition = CalculatorDefinition(
        id="test.chain", name="chain", category="test", strategy="expression",
        inputs=[InputSpec(name="a", type="decimal", required=True)],
        derived_variables={"b": "a * 2", "c": "b + 1"},
        formula={"expression": "c"},
        output={"name": "result"},
    )
    outcome = _strategy().run(definition, {"a": Decimal("5")}, CalculationRequest(calculator_id="test.chain"))
    assert outcome.derived_values["b"] == Decimal("10")
    assert outcome.derived_values["c"] == Decimal("11")
    assert outcome.result["result"] == Decimal("11.00")
