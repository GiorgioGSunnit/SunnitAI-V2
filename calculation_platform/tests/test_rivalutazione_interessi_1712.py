"""Tests for legal_it.rivalutazione_interessi_1712 (Cass. SS.UU. 1712/1995).

All FOI values are the SCHEMA PLACEHOLDERS from foi_indices.yml
(2024-11: 100.0, 2024-12: 100.5, 2025-12: 102.0, 2026-02: 102.5) —
synthetic, verified: false. Legal rates: 2024 2.5%, 2025 2%, 2026 1.6%.
"""

from decimal import Decimal

from support import engine
from app.schemas.calculation_request import CalculationRequest
from app.strategies.foi_revaluation_interest import year_slices

CALC_ID = "legal_it.rivalutazione_interessi_1712"


def _calculate(importo=1000, data_iniziale="2024-11-15", data_finale="2026-02-10"):
    return engine.calculate(CalculationRequest(
        calculator_id=CALC_ID,
        inputs={"importo": importo, "data_iniziale": data_iniziale, "data_finale": data_finale},
    ))


def test_single_year_slice():
    """Hand calculation, single partial slice inside 2024 (leap year):

        FOI(2024-11) = 100.0, FOI(2024-12) = 100.5 (placeholders)
        capitale 1000 -> 1000 x 100.5/100.0 = 1005 (esatto)
        media = (1000 + 1005) / 2 = 1002.5
        giorni 2024-11-15..2024-12-31 = 16 + 31 = 47, divisore 366 (bisestile)
        interessi = 1002.5 x 0.025 x 47/366 = 3.21840846994535...
        arrotondato half_up a 2 decimali = 3.22
        totale = 1005 + 3.2184... = 1008.22

    verified_against: TODO (official calculator/source check pending)
    """
    result = _calculate(data_iniziale="2024-11-15", data_finale="2024-12-31")
    assert result.status == "success"
    assert result.result["capitale_rivalutato"] == "1005.00"
    assert result.result["interessi_totali"] == "3.22"
    assert result.result["totale"] == "1008.22"


def test_multi_year_with_rate_change_and_leap_year():
    """Hand calculation, three slices spanning leap year 2024, the 2025->2026
    rate change, and partial first/last years:

        Slice 2024 (2024-11-15..2024-12-31, 47 gg, divisore 366, saggio 2,5%):
            capitale 1000 -> 1005 (FOI 100.0 -> 100.5), media 1002.5
            interessi = 1002.5 x 0.025 x 47/366  = 3.218408469945355...
        Slice 2025 (intero anno, 365 gg, saggio 2%):
            capitale 1005 -> 1005 x 102.0/100.5 = 1020 (esatto), media 1012.5
            interessi = 1012.5 x 0.02 x 365/365 = 20.25
        Slice 2026 (2026-01-01..2026-02-10, 41 gg, divisore 365, saggio 1,6%):
            capitale 1020 -> 1020 x 102.5/102.0 = 1025 (esatto), media 1022.5
            interessi = 1022.5 x 0.016 x 41/365 = 1.837698630136986...
        interessi_totali = 25.30610710008234... -> 25.31
        capitale_rivalutato = 1025.00
        totale = 1025 + 25.3061... = 1050.31

    verified_against: TODO (official calculator/source check pending)
    """
    result = _calculate()
    assert result.status == "success"
    assert result.result["capitale_rivalutato"] == "1025.00"
    assert result.result["interessi_totali"] == "25.31"
    assert result.result["totale"] == "1050.31"

    slices = [s for s in result.steps if s["type"] == "year_slice"]
    assert len(slices) == 3
    assert slices[0]["rate"] == "0.025"
    assert slices[0]["divisor"] == "366"  # 2024 is a leap year
    assert slices[0]["days"] == 47
    assert slices[1]["rate"] == "0.02"
    assert slices[1]["days"] == 365
    assert slices[1]["mean_base"] == "1012.5"
    assert slices[2]["rate"] == "0.016"
    assert slices[2]["divisor"] == "365"
    assert slices[2]["days"] == 41
    # readable Italian computation line on every slice
    for s in slices:
        assert "capitale rivalutato" in s["note"]
        assert "giorni" in s["note"]


def test_partial_first_and_last_years_are_pro_rata_by_days():
    result = _calculate()
    slices = [s for s in result.steps if s["type"] == "year_slice"]
    # first slice: only 47 of 366 days of 2024; last: only 41 of 365 of 2026
    first, full, last = slices
    assert first["days"] == 47
    assert last["days"] == 41
    # pro-rata: interest of the partial slices is days/divisor of a full year
    mean_first = Decimal(first["mean_base"])
    expected_first = mean_first * Decimal("0.025") * 47 / 366
    assert Decimal(first["interest"]) == expected_first


def test_interest_never_compounds_into_capital():
    """Property: the revalued capital must equal importo x FOI(final)/FOI(initial)
    exactly (the FOI chain telescopes) — independent of any interest, proving
    interest never enters the capital base (no anatocismo)."""
    for importo in (1000, 25000, 7):
        result = _calculate(importo=importo)
        assert result.status == "success"
        expected = Decimal(importo) * Decimal("102.5") / Decimal("100.0")
        assert Decimal(result.result["capitale_rivalutato"]) == expected.quantize(Decimal("0.01"))
        # and the total is exactly capital + interest, nothing more
        assert Decimal(result.result["totale"]) == (
            Decimal(result.result["capitale_rivalutato"]) + Decimal(result.result["interessi_totali"])
        )


def test_criterion_warning_is_present():
    result = _calculate()
    assert any("criterio della media" in w.message.lower() or "MEDIA" in w.message for w in result.warnings)


def test_year_slices_splits_on_calendar_years():
    from datetime import date

    assert year_slices(date(2024, 11, 15), date(2026, 2, 10)) == [
        (date(2024, 11, 15), date(2024, 12, 31)),
        (date(2025, 1, 1), date(2025, 12, 31)),
        (date(2026, 1, 1), date(2026, 2, 10)),
    ]
    assert year_slices(date(2025, 3, 1), date(2025, 3, 1)) == [(date(2025, 3, 1), date(2025, 3, 1))]


def test_missing_foi_month_is_a_structured_error():
    result = _calculate(data_finale="2026-03-15")
    assert result.status == "error"
    assert result.errors[0].code == "parameter_unresolved"
    assert "2026-03" in result.errors[0].message


def test_parameters_used_carry_indices_and_rates():
    result = _calculate()
    params = result.parameters_used
    assert "foi_index_2024_11" in params
    assert "foi_index_2024_12" in params
    assert "foi_index_2025_12" in params
    assert "foi_index_2026_02" in params
    assert "legal_interest_rate_from_2024-01-01" in params
    assert "legal_interest_rate_from_2025-01-01" in params
    assert "legal_interest_rate_from_2026-01-01" in params
