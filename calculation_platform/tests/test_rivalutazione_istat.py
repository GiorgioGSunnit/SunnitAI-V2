"""Golden tests for legal_it.rivalutazione_istat (FOI monetary revaluation).

The FOI months used here are the SCHEMA PLACEHOLDERS from
parameters/legal_it/foi_indices.yml (synthetic values, verified: false) —
the arithmetic below is exact for those placeholders and must be re-verified
once the real ISTAT series is loaded.
"""

from decimal import Decimal

from support import engine
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


def test_cross_base_uses_the_base_link_coefficient():
    """FOI(2021-03) = 103.3 in base 2015, FOI(2026-06) = 102.8 in base 2025.
    The bases differ, so the end index is relinked into base 2015 with the
    store's coefficient 1.214 before dividing:

        coefficiente = (102.8 x 1.214) / 103.3 = 124.7992 / 103.3 = 1.208123...
        importo 1000 x 1.208123... = 1208.12 (full-precision coefficient policy)

    The official ISTAT calculator, which rounds the coefficient to 3 decimals
    (1.208), returns 1208.00 — the cents difference is the documented policy.
    """
    result = _calculate(data_iniziale="2021-03-15", data_finale="2026-06-15")
    assert result.status == "success"
    assert result.result["importo_rivalutato"] == "1208.12"
    coefficient = Decimal(result.derived_values["coefficiente_rivalutazione"])
    assert coefficient == (Decimal("102.8") * Decimal("1.214")) / Decimal("103.3")
    step = next(s for s in result.steps if s["type"] == "revaluation_coefficient")
    assert step["index_initial_base"] == 2015
    assert step["index_final_base"] == 2025


def test_cross_base_without_a_link_is_a_structured_error():
    """If two months carry different bases and no base-link coefficient is
    registered, the division is refused rather than silently mixing bases."""
    from support import build_engine

    foi = """
values:
  - {parameter_id: legal_it.foi_index, year: 2021, month: 3, value: "103.3", base_year: 2015, unit: index, placeholder: true, source_ref: fixture}
  - {parameter_id: legal_it.foi_index, year: 2026, month: 6, value: "102.8", base_year: 2025, unit: index, placeholder: true, source_ref: fixture}
"""
    unlinked_engine, _ = build_engine(foi)
    result = unlinked_engine.calculate(CalculationRequest(
        calculator_id=CALC_ID,
        inputs={"importo": 1000, "data_iniziale": "2021-03-15", "data_finale": "2026-06-15"},
    ))
    assert result.status == "error"
    assert result.errors[0].code == "parameter_unresolved"
    assert "base" in result.errors[0].message.lower()
