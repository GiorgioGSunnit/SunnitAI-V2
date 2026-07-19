"""Tests for legal_it.compensi_dm55 (parametri forensi DM 55/2014).

The fase values are the SCHEMA PLACEHOLDERS from dm55_compensi.yml for the
scaglione 26.000,01-52.000 (studio 2000, introduttiva 1500, istruttoria
5000, decisionale 3500 — synthetic, verified: false).
"""

from decimal import Decimal

from app.main import engine
from app.schemas.calculation_request import CalculationRequest

CALC_ID = "legal_it.compensi_dm55"


def _calculate(**inputs):
    payload = {"valore_causa": 30000, "fasi": ["studio", "introduttiva", "decisionale"]}
    payload.update(inputs)
    return engine.calculate(CalculationRequest(calculator_id=CALC_ID, inputs=payload))


def test_full_chain_multi_fase_golden_case():
    """Hand calculation with the placeholder values:

        compenso = studio 2000 + introduttiva 1500 + decisionale 3500 = 7000
        spese generali 15%   = 1050.00 -> subtotale  8050.00
        CPA 4% di 8050       =  322.00 -> subtotale  8372.00
        IVA 22% di 8372      = 1841.84 -> totale    10213.84

    verified_against: TODO (official calculator/source check pending)
    """
    result = _calculate()
    assert result.status == "success"
    r = result.result
    assert r["compenso_tabellare"] == "7000.00"
    assert r["compenso_adeguato"] == "7000.00"
    assert r["spese_generali"] == "1050.00"
    assert r["subtotale_con_spese"] == "8050.00"
    assert r["cpa"] == "322.00"
    assert r["subtotale_con_cpa"] == "8372.00"
    assert r["iva"] == "1841.84"
    assert r["totale"] == "10213.84"
    # every stage is an explicit step
    step_types = [s["type"] for s in result.steps]
    for expected in ("fase", "compenso_base", "spese_generali", "cpa", "iva"):
        assert expected in step_types


def test_single_fase_chain():
    """istruttoria only: 5000 -> +15% = 5750 -> +4% = 5980 -> +22% = 7295.60.

    verified_against: TODO (official calculator/source check pending)
    """
    result = _calculate(fasi=["istruttoria"])
    assert result.status == "success"
    assert result.result["compenso_tabellare"] == "5000.00"
    assert result.result["totale"] == "7295.60"


def test_fasi_accepts_comma_separated_string():
    result = _calculate(fasi="studio, decisionale")
    assert result.status == "success"
    assert result.result["compenso_tabellare"] == "5500.00"


def test_aumento_within_bound_is_applied():
    # 7000 + 20% = 8400 -> +15% = 9660 -> +4% = 10046.40 -> +22% = 12256.608 -> 12256.61
    result = _calculate(aumento_pct=20)
    assert result.status == "success"
    assert result.result["compenso_adeguato"] == "8400.00"
    assert result.result["totale"] == "12256.61"


def test_aumento_above_80_percent_is_rejected():
    result = _calculate(aumento_pct=90)
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"
    assert "80" in result.errors[0].message


def test_riduzione_above_50_percent_is_rejected_for_general_fasi():
    result = _calculate(riduzione_pct=60)
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"
    assert "50" in result.errors[0].message


def test_riduzione_up_to_70_percent_allowed_for_sole_istruttoria():
    result = _calculate(fasi=["istruttoria"], riduzione_pct=70)
    assert result.status == "success"
    # 5000 - 70% = 1500
    assert result.result["compenso_adeguato"] == "1500.00"


def test_riduzione_50_percent_applied():
    result = _calculate(riduzione_pct=50)
    assert result.status == "success"
    assert result.result["compenso_adeguato"] == "3500.00"


def test_aumento_and_riduzione_together_are_rejected():
    result = _calculate(aumento_pct=10, riduzione_pct=10)
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"


def test_unknown_fase_is_rejected():
    result = _calculate(fasi=["studio", "appello"])
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"
    assert "appello" in result.errors[0].message


def test_unpopulated_scaglione_fails_loudly_as_to_verify_stub():
    result = _calculate(valore_causa=10000)
    assert result.status == "error"
    assert result.errors[0].code == "parameter_unresolved"
    assert "TO_VERIFY" in result.errors[0].message


def test_placeholder_values_produce_a_warning():
    result = _calculate()
    assert any("SEGNAPOSTO" in w.message for w in result.warnings)


def test_totale_is_consistent_with_the_chain():
    result = _calculate()
    r = result.result
    compenso = Decimal(r["compenso_adeguato"])
    expected = compenso * Decimal("1.15") * Decimal("1.04") * Decimal("1.22")
    assert Decimal(r["totale"]) == expected.quantize(Decimal("0.01"))
