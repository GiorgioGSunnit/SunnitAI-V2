"""Tests for legal_it.compensi_dm55 (parametri forensi DM 55/2014).

The fase values are the real Tabella 2 (DM 147/2022) figures from
dm55_compensi.yml, verified against the official G.U. annex (verified: true).
Scaglione 26.000,01-52.000: studio 1701, introduttiva 1204, istruttoria 1806,
decisionale 2905.
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
    """Hand calculation with the real Tabella 2 values (scaglione 26.000,01-52.000):

        compenso = studio 1701 + introduttiva 1204 + decisionale 2905 = 5810
        spese generali 15%   =  871.50 -> subtotale  6681.50
        CPA 4% di 6681.50    =  267.26 -> subtotale  6948.76
        IVA 22% di 6948.76   = 1528.73 -> totale     8477.49
    """
    result = _calculate()
    assert result.status == "success"
    r = result.result
    assert r["compenso_tabellare"] == "5810.00"
    assert r["compenso_adeguato"] == "5810.00"
    assert r["spese_generali"] == "871.50"
    assert r["subtotale_con_spese"] == "6681.50"
    assert r["cpa"] == "267.26"
    assert r["subtotale_con_cpa"] == "6948.76"
    assert r["iva"] == "1528.73"
    assert r["totale"] == "8477.49"
    # every stage is an explicit step
    step_types = [s["type"] for s in result.steps]
    for expected in ("fase", "compenso_base", "spese_generali", "cpa", "iva"):
        assert expected in step_types


def test_single_fase_chain():
    """istruttoria only (1806): -> +15% = 2076.90 -> +4% = 2159.976 ->
    +22% = 2635.17."""
    result = _calculate(fasi=["istruttoria"])
    assert result.status == "success"
    assert result.result["compenso_tabellare"] == "1806.00"
    assert result.result["totale"] == "2635.17"


def test_fasi_accepts_comma_separated_string():
    # studio 1701 + decisionale 2905 = 4606
    result = _calculate(fasi="studio, decisionale")
    assert result.status == "success"
    assert result.result["compenso_tabellare"] == "4606.00"


def test_aumento_within_bound_is_applied():
    # 5810 + 20% = 6972 -> +15% = 8017.80 -> +4% = 8338.512 -> +22% = 10172.98
    result = _calculate(aumento_pct=20)
    assert result.status == "success"
    assert result.result["compenso_adeguato"] == "6972.00"
    assert result.result["totale"] == "10172.98"


# --- Adjustment limits: DM 55/2014 art. 4 as amended by DM 147/2022 (±50%) ---

def test_aumento_50_percent_accepted():
    # 5810 + 50% = 8715 (boundary accepted)
    result = _calculate(aumento_pct=50)
    assert result.status == "success"
    assert result.result["compenso_adeguato"] == "8715.00"


def test_aumento_above_50_percent_is_rejected():
    result = _calculate(aumento_pct=Decimal("50.01"))
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"
    assert "50" in result.errors[0].message


def test_riduzione_50_percent_accepted():
    # 5810 - 50% = 2905 (boundary accepted)
    result = _calculate(riduzione_pct=50)
    assert result.status == "success"
    assert result.result["compenso_adeguato"] == "2905.00"


def test_riduzione_above_50_percent_is_rejected():
    result = _calculate(riduzione_pct=Decimal("50.01"))
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"
    assert "50" in result.errors[0].message


def test_former_70_percent_istruttoria_reduction_is_now_rejected():
    # DM 147/2022 removed the special 70% reduction for the fase istruttoria.
    result = _calculate(fasi=["istruttoria"], riduzione_pct=70)
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"
    assert "50" in result.errors[0].message


def test_aumento_and_riduzione_together_are_rejected():
    result = _calculate(aumento_pct=10, riduzione_pct=10)
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"


def test_unknown_fase_is_rejected():
    result = _calculate(fasi=["studio", "appello"])
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"
    assert "appello" in result.errors[0].message


def test_lower_scaglione_is_now_populated():
    # valore 10.000 -> scaglione 5.200,01-26.000: studio 919 + introduttiva
    # 777 + decisionale 1701 = 3397 (previously an unpopulated TO_VERIFY stub).
    result = _calculate(valore_causa=10000)
    assert result.status == "success"
    assert result.result["compenso_tabellare"] == "3397.00"


def test_value_above_top_scaglione_is_unsupported():
    # No seventh table row above 520.000 EUR (art. 6 progressive rule): refuse
    # honestly rather than invent a value.
    result = _calculate(valore_causa=600000)
    assert result.status == "error"
    assert result.errors[0].code == "parameter_unresolved"
    assert result.errors[0].details.get("unsupported_range") is True
    assert "520000" in str(result.errors[0].details.get("highest_scaglione_max"))


def test_verified_values_produce_no_data_quality_warning():
    # Table 2 values are source-verified (verified: true) -> no placeholder or
    # pending-verification warning.
    result = _calculate()
    assert not any("SEGNAPOSTO" in w.message for w in result.warnings)
    assert not any("non ancora verificat" in w.message.lower() for w in result.warnings)


def test_totale_is_consistent_with_the_chain():
    result = _calculate()
    r = result.result
    compenso = Decimal(r["compenso_adeguato"])
    expected = compenso * Decimal("1.15") * Decimal("1.04") * Decimal("1.22")
    assert Decimal(r["totale"]) == expected.quantize(Decimal("0.01"))


# --- Exact bracket boundaries and one cent below/above -----------------------

def test_bracket_boundary_at_and_across_5200():
    # studio only: 5.200,00 -> scaglione 1.100,01-5.200 (425); 5.200,01 ->
    # scaglione 5.200,01-26.000 (919).
    at = _calculate(valore_causa=Decimal("5200.00"), fasi=["studio"])
    above = _calculate(valore_causa=Decimal("5200.01"), fasi=["studio"])
    assert at.result["compenso_tabellare"] == "425.00"
    assert above.result["compenso_tabellare"] == "919.00"


def test_bracket_boundary_at_and_across_52000():
    # studio only: 52.000,00 -> scaglione 26.000,01-52.000 (1701); 52.000,01 ->
    # scaglione 52.000,01-260.000 (2552).
    at = _calculate(valore_causa=Decimal("52000.00"), fasi=["studio"])
    above = _calculate(valore_causa=Decimal("52000.01"), fasi=["studio"])
    assert at.result["compenso_tabellare"] == "1701.00"
    assert above.result["compenso_tabellare"] == "2552.00"


def test_top_bracket_boundary_520000_supported_and_one_cent_above_unsupported():
    at = _calculate(valore_causa=Decimal("520000.00"), fasi=["studio"])
    above = _calculate(valore_causa=Decimal("520000.01"), fasi=["studio"])
    assert at.status == "success"
    assert at.result["compenso_tabellare"] == "3544.00"
    assert above.status == "error"
    assert above.errors[0].details.get("unsupported_range") is True


# --- CPA / IVA explicit flags: all four combinations -------------------------
# Base golden case: compenso 5810 -> +15% = subtotale_con_spese 6681.50.

def test_cpa_on_iva_on_is_the_default():
    result = _calculate()  # flags omitted -> both default true
    r = result.result
    assert r["cpa_applicata"] is True and r["iva_applicata"] is True
    assert r["cpa"] == "267.26"
    assert r["iva"] == "1528.73"
    assert r["totale"] == "8477.49"
    # the default was recorded as an assumption (not a hidden universal)
    assert any("applica_cpa" in a.message for a in result.assumptions)
    assert any("applica_iva" in a.message for a in result.assumptions)


def test_cpa_on_iva_off():
    result = _calculate(applica_iva=False)
    r = result.result
    assert r["cpa_applicata"] is True and r["iva_applicata"] is False
    assert r["cpa"] == "267.26"
    assert r["subtotale_con_cpa"] == "6948.76"
    assert r["iva"] == "0.00"
    assert r["totale"] == "6948.76"


def test_cpa_off_iva_on():
    result = _calculate(applica_cpa=False)
    r = result.result
    assert r["cpa_applicata"] is False and r["iva_applicata"] is True
    assert r["cpa"] == "0.00"
    assert r["subtotale_con_cpa"] == "6681.50"
    # IVA on the subtotale with spese only: 6681.50 x 0.22 = 1469.93
    assert r["iva"] == "1469.93"
    assert r["totale"] == "8151.43"


def test_cpa_off_iva_off():
    result = _calculate(applica_cpa=False, applica_iva=False)
    r = result.result
    assert r["cpa_applicata"] is False and r["iva_applicata"] is False
    assert r["cpa"] == "0.00"
    assert r["iva"] == "0.00"
    assert r["totale"] == "6681.50"
