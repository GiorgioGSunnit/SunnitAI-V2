import pytest

from app.core.errors import StrategyExecutionError
from app.schemas.calculation_request import CalculationRequest
from app.schemas.calculator_definition import CalculatorDefinition, InputSpec
from app.strategies.decision_table import DecisionTableStrategy


def _definition(rules):
    return CalculatorDefinition(
        id="test.decision", name="decision", category="test", strategy="decision_table",
        inputs=[InputSpec(name="score", type="decimal", required=False, default=0)],
        formula={"rules": rules},
        output={"name": "tier"},
    )


def test_equals_comparator_matches():
    definition = _definition([{"when": {"input": "score", "equals": 10}, "value": "exact"}])
    strategy = DecisionTableStrategy(parameter_store=None)
    outcome = strategy.run(definition, {"score": 10}, CalculationRequest(calculator_id="test.decision"))
    assert outcome.result["tier"] == "exact"


@pytest.mark.parametrize(
    "comparator,threshold,score,expected_match",
    [
        ("greater_than", 50, 60, True),
        ("greater_than", 50, 50, False),
        ("less_than", 50, 40, True),
        ("less_than", 50, 50, False),
        ("at_least", 50, 50, True),
        ("at_least", 50, 49, False),
        ("at_most", 50, 50, True),
        ("at_most", 50, 51, False),
    ],
)
def test_numeric_comparators(comparator, threshold, score, expected_match):
    definition = _definition([
        {"when": {"input": "score", comparator: threshold}, "value": "matched"},
        {"value": "default"},
    ])
    strategy = DecisionTableStrategy(parameter_store=None)
    outcome = strategy.run(definition, {"score": score}, CalculationRequest(calculator_id="test.decision"))
    assert outcome.result["tier"] == ("matched" if expected_match else "default")


def test_rule_with_no_when_acts_as_default():
    definition = _definition([
        {"when": {"input": "score", "greater_than": 1000}, "value": "never"},
        {"value": "fallback"},
    ])
    strategy = DecisionTableStrategy(parameter_store=None)
    outcome = strategy.run(definition, {"score": 5}, CalculationRequest(calculator_id="test.decision"))
    assert outcome.result["tier"] == "fallback"


def test_no_match_and_no_default_raises_structured_error():
    definition = _definition([{"when": {"input": "score", "greater_than": 1000}, "value": "never"}])
    strategy = DecisionTableStrategy(parameter_store=None)
    with pytest.raises(StrategyExecutionError) as exc_info:
        strategy.run(definition, {"score": 5}, CalculationRequest(calculator_id="test.decision"))
    assert exc_info.value.code == "strategy_execution_failed"


def test_unrecognized_comparator_raises_structured_error():
    definition = _definition([{"when": {"input": "score", "not_a_real_comparator": 1}, "value": "x"}])
    strategy = DecisionTableStrategy(parameter_store=None)
    with pytest.raises(StrategyExecutionError):
        strategy.run(definition, {"score": 5}, CalculationRequest(calculator_id="test.decision"))
