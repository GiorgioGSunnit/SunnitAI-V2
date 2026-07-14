from datetime import date
from decimal import Decimal

import pytest

from app.core.errors import ParameterResolutionError
from app.main import engine
from app.resolvers.date_parameter_resolver import resolve_parameters
from app.schemas.calculation_request import CalculationRequest


def test_parameter_store_origin_carries_citation_and_effective_range():
    definition = engine.registry.get("legal_it.irpef")
    request = CalculationRequest(calculator_id="legal_it.irpef", inputs={"taxable_income": 42000}, tax_year=2026)
    resolution = resolve_parameters(definition, engine.parameter_store, request)

    brackets = resolution.resolved["brackets"]
    assert brackets.origin == "parameter_store"
    assert brackets.effective_from == "2026-01-01"
    assert brackets.effective_to == "2026-12-31"
    assert brackets.official is True
    assert len(brackets.citations) >= 1


def test_caller_supplied_value_overrides_parameter_store():
    definition = engine.registry.get("legal_it.registration_tax_leases")
    request = CalculationRequest(
        calculator_id="legal_it.registration_tax_leases",
        inputs={"annual_rent": 9600, "years": 4, "first_registration": True},
        caller_supplied_values={"rate": 0.05},
    )
    resolution = resolve_parameters(definition, engine.parameter_store, request)
    assert resolution.resolved["rate"].origin == "caller_supplied"
    assert resolution.values["rate"] == Decimal("0.05")
    # the untouched parameter still resolves from the store
    assert resolution.resolved["minimum"].origin == "parameter_store"


def test_static_default_origin_when_no_caller_value_or_parameter_id():
    from app.schemas.calculator_definition import CalculatorDefinition, ParameterRef

    definition = CalculatorDefinition(
        id="test.static", name="static", category="test", strategy="expression",
        parameters=[ParameterRef(name="rate", default=0.1)],
        formula={"expression": "rate"},
    )
    request = CalculationRequest(calculator_id="test.static", inputs={})
    resolution = resolve_parameters(definition, engine.parameter_store, request)
    assert resolution.resolved["rate"].origin == "static_default"


def test_missing_parameter_raises_structured_error():
    from app.schemas.calculator_definition import CalculatorDefinition, ParameterRef

    definition = CalculatorDefinition(
        id="test.missing", name="missing", category="test", strategy="expression",
        parameters=[ParameterRef(name="rate", parameter_id="does.not.exist")],
        formula={"expression": "rate"},
    )
    request = CalculationRequest(calculator_id="test.missing", inputs={})
    with pytest.raises(ParameterResolutionError) as exc_info:
        resolve_parameters(definition, engine.parameter_store, request)
    assert exc_info.value.code == "parameter_unresolved"


def test_date_resolution_reflects_explicit_as_of_date():
    definition = engine.registry.get("legal_it.irpef")
    request = CalculationRequest(
        calculator_id="legal_it.irpef", inputs={"taxable_income": 20000}, as_of_date=date(2024, 6, 15),
    )
    resolution = resolve_parameters(definition, engine.parameter_store, request)
    assert resolution.date_resolution["source"] == "explicit_as_of_date"
    assert resolution.date_resolution["as_of_date"] == "2024-06-15"


def test_date_resolution_defaults_to_today_when_nothing_given():
    definition = engine.registry.get("legal_it.irpef")
    request = CalculationRequest(calculator_id="legal_it.irpef", inputs={"taxable_income": 20000})
    resolution = resolve_parameters(definition, engine.parameter_store, request)
    assert resolution.date_resolution["source"] == "defaulted_to_today"
    assert resolution.date_resolution["as_of_date"] == date.today().isoformat()


def test_date_resolution_absent_when_calculator_has_no_date_versioned_parameters():
    definition = engine.registry.get("business.invoice_total")
    request = CalculationRequest(
        calculator_id="business.invoice_total",
        inputs={"net_amount": 1000, "vat_rate": 0.22},
    )
    resolution = resolve_parameters(definition, engine.parameter_store, request)
    assert resolution.date_resolution is None
