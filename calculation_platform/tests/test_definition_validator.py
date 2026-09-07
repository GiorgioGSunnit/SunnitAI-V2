import pytest

from app.core.definition_validator import validate_definition
from app.core.errors import DefinitionValidationError
from app.schemas.calculator_definition import CalculatorDefinition, InputSpec, ParameterRef


def test_valid_definition_passes_silently():
    definition = CalculatorDefinition(
        id="test.valid", name="valid", category="test", strategy="expression",
        inputs=[InputSpec(name="a", type="decimal")],
        formula={"expression": "a + 1"},
    )
    validate_definition(definition)  # must not raise


def test_rejects_expression_referencing_undeclared_variable():
    definition = CalculatorDefinition(
        id="test.bad", name="bad", category="test", strategy="expression",
        inputs=[InputSpec(name="a", type="decimal")],
        formula={"expression": "a + b"},
    )
    with pytest.raises(DefinitionValidationError) as exc_info:
        validate_definition(definition)
    assert "b" in exc_info.value.details["unknown_variables"]


def test_rejects_unknown_strategy():
    definition = CalculatorDefinition(id="test.bad", name="bad", category="test", strategy="not_a_real_strategy")
    with pytest.raises(DefinitionValidationError):
        validate_definition(definition)


def test_rejects_missing_required_formula_key():
    definition = CalculatorDefinition(
        id="test.bad", name="bad", category="test", strategy="progressive_brackets", formula={},
    )
    with pytest.raises(DefinitionValidationError) as exc_info:
        validate_definition(definition)
    assert exc_info.value.details["missing_formula_key"] == "base_input"


def test_rejects_unresolvable_parameter():
    definition = CalculatorDefinition(
        id="test.bad", name="bad", category="test", strategy="expression",
        parameters=[ParameterRef(name="rate")],
        formula={"expression": "rate"},
    )
    with pytest.raises(DefinitionValidationError) as exc_info:
        validate_definition(definition)
    assert exc_info.value.details["parameter"] == "rate"


def test_rejects_disallowed_function_call():
    definition = CalculatorDefinition(
        id="test.bad", name="bad", category="test", strategy="expression",
        inputs=[InputSpec(name="a", type="decimal")],
        formula={"expression": "eval(a)"},
    )
    with pytest.raises(DefinitionValidationError):
        validate_definition(definition)


def test_rejects_malformed_citation():
    definition = CalculatorDefinition(
        id="test.bad", name="bad", category="test", strategy="expression",
        inputs=[InputSpec(name="a", type="decimal")],
        formula={"expression": "a"},
        citations=[{"not_a_reference_field": "oops"}],
    )
    with pytest.raises(DefinitionValidationError):
        validate_definition(definition)


def test_chained_derived_variables_can_reference_earlier_ones():
    definition = CalculatorDefinition(
        id="test.chained", name="chained", category="test", strategy="expression",
        inputs=[InputSpec(name="a", type="decimal")],
        derived_variables={"b": "a * 2", "c": "b + 1"},
        formula={"expression": "c"},
    )
    validate_definition(definition)  # must not raise — c legitimately depends on b
