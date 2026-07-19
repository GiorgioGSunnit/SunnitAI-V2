"""Golden tests for legal_it.rivalutazione_istat (FOI monetary revaluation).

The FOI months used here are the SCHEMA PLACEHOLDERS from
parameters/legal_it/foi_indices.yml (synthetic values, verified: false) —
the arithmetic below is exact for those placeholders and must be re-verified
once the real ISTAT series is loaded.
"""

from decimal import Decimal

from app.main import engine
from app.schemas.calculation_request import CalculationRequest

CALC_ID = "legal_it.rivalutazione_istat"


def _calculate(**overrides):
    payload = {
        "importo": 1000,
        "data_iniziale": "2024-11-15",
        "data_finale": "2026-02-10",
    }
    payload.update(overrides)
    return engine.calculate(CalculationRequest(calculator_id=CALC_ID, inputs=payload))


def test_rivalutazione_golden_case_with_placeholder_months():
    """Hand calculation with the placeholder FOI months:

        FOI(2024-11) = 100.0   (placeholder)
        FOI(2026-02) = 102.5   (placeholder)
        coefficiente = 102.5 / 100.0 = 1.025 (exact in Decimal)
        importo_rivalutato = 1000 x 1.025 = 1025.00

    verified_against: TODO (official calculator/source check pending)
    """
    result = _calculate()
    assert result.status == "success"
    assert result.result["importo_rivalutato"] == "1025.00"
    assert result.derived_values["coefficiente_rivalutazione"] == "1.025"


def test_result_carries_both_indices_with_source_refs():
    result = _calculate()
    params = result.parameters_used
    assert set(params) == {"foi_index_2024_11", "foi_index_2026_02"}
    assert params["foi_index_2024_11"]["value"] == "100.0"
    assert params["foi_index_2026_02"]["value"] == "102.5"
    for p in params.values():
        assert p["origin"] == "parameter_store"
        assert p["source"]  # source_ref surfaced for audit


def test_coefficient_is_an_explicit_step():
    result = _calculate()
    coefficient_steps = [s for s in result.steps if s["type"] == "revaluation_coefficient"]
    assert len(coefficient_steps) == 1
    step = coefficient_steps[0]
    assert step["index_initial"] == "100.0"
    assert step["index_final"] == "102.5"
    assert step["coefficient"] == "1.025"


def test_placeholder_months_produce_a_warning():
    result = _calculate()
    assert any("SEGNAPOSTO" in w.message for w in result.warnings)


def test_missing_month_is_a_structured_error_naming_param_and_month():
    result = _calculate(data_finale="2026-03-10")
    assert result.status == "error"
    error = result.errors[0]
    assert error.code == "parameter_unresolved"
    assert "legal_it.foi_index" in error.message
    assert "2026-03" in error.message
    assert error.details["year"] == 2026
    assert error.details["month"] == 3


def test_reversed_dates_are_an_error():
    result = _calculate(data_iniziale="2026-02-10", data_finale="2024-11-15")
    assert result.status == "error"


def test_same_month_coefficient_is_one():
    result = _calculate(data_iniziale="2024-11-01", data_finale="2024-11-30")
    assert result.status == "success"
    assert result.result["importo_rivalutato"] == "1000.00"
    assert Decimal(result.derived_values["coefficiente_rivalutazione"]) == Decimal("1")
