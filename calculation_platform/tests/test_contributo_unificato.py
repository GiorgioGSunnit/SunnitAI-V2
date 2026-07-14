from app.main import engine
from app.schemas.calculation_request import CalculationRequest


def _calculate(**inputs):
    return engine.calculate(CalculationRequest(
        calculator_id="legal_it.contributo_unificato_civile", inputs=inputs,
    ))


def test_band_boundaries_are_inclusive():
    # 1100 is the top of the first band, 1100.01 falls into the second.
    assert _calculate(valore_causa=1100).result["contributo_dovuto"] == 43.00
    assert _calculate(valore_causa="1100.01").result["contributo_dovuto"] == 98.00


def test_every_band_matches_expected_amount():
    expectations = {
        500: 43.00,
        5200: 98.00,
        26000: 237.00,
        52000: 518.00,
        260000: 759.00,
        520000: 1214.00,
        520001: 1686.00,
    }
    for value, expected in expectations.items():
        result = _calculate(valore_causa=value)
        assert result.status == "success"
        assert result.result["contributo_dovuto"] == expected, f"valore {value}"


def test_appello_increases_by_half_and_cassazione_doubles():
    assert _calculate(valore_causa=30000, grado="appello").result["contributo_dovuto"] == 777.00
    assert _calculate(valore_causa=30000, grado="cassazione").result["contributo_dovuto"] == 1036.00


def test_unknown_grado_is_an_error_not_a_guess():
    result = _calculate(valore_causa=30000, grado="revocazione")
    assert result.status == "error"
    assert "revocazione" in result.errors[0].message


def test_indeterminable_value_uses_dedicated_row_and_ignores_amount():
    result = _calculate(valore_causa=999999, valore_indeterminabile=True)
    assert result.result["contributo_dovuto"] == 518.00
    assert any("ignored" in a.message for a in result.assumptions)


def test_missing_both_value_and_indeterminable_flag_is_an_error():
    result = _calculate()
    assert result.status == "error"


def test_exemption_forces_zero_with_warning():
    result = _calculate(valore_causa=30000, esente=True)
    assert result.result["contributo_dovuto"] == 0.00
    assert any("art. 10" in w.message for w in result.warnings)


def test_esente_as_string_false_does_not_exempt():
    # Regression guard for the strict boolean coercion: "false" must not
    # silently become an exemption (or its opposite).
    result = _calculate(valore_causa=30000, esente="false")
    assert result.result["contributo_dovuto"] == 518.00


def test_result_carries_official_citation_and_band_step():
    result = _calculate(valore_causa=15000)
    assert any(c.official for c in result.citations)
    assert any(s.get("type") == "band_matched" for s in result.steps)
