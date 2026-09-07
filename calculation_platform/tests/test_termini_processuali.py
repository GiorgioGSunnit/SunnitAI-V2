from app.main import engine
from app.schemas.calculation_request import CalculationRequest


def _calculate(**inputs):
    return engine.calculate(CalculationRequest(
        calculator_id="legal_it.termini_processuali_civili", inputs=inputs,
    ))


def test_dies_a_quo_excluded_dies_ad_quem_included():
    result = _calculate(data_decorrenza="2026-01-10", giorni=30)
    assert result.status == "success"
    assert result.result["scadenza"] == "2026-02-09"


def test_term_starting_inside_august_defers_to_september():
    # Notification on Aug 10 with feriale suspension: counting effectively
    # starts Sep 1, so 10 days land on Sep 10.
    result = _calculate(data_decorrenza="2026-08-10", giorni=10)
    assert result.result["scadenza"] == "2026-09-10"
    assert result.result["giorni_feriale_sospesi"] == 21


def test_term_straddling_august_skips_31_days():
    result = _calculate(data_decorrenza="2026-06-15", giorni=90)
    assert result.result["scadenza"] == "2026-10-14"
    assert result.result["giorni_feriale_sospesi"] == 31


def test_term_ending_before_august_is_unaffected_by_feriale():
    result = _calculate(data_decorrenza="2026-05-04", giorni=30)
    assert result.result["scadenza"] == "2026-06-03"
    assert result.result["giorni_feriale_sospesi"] == 0


def test_feriale_disabled_for_excluded_rites():
    result = _calculate(data_decorrenza="2026-06-15", giorni=90, sospensione_feriale=False)
    assert result.result["scadenza"] == "2026-09-14"  # Sun Sep 13 rolled to Monday


def test_saturday_and_holiday_chain_rolls_to_next_working_day():
    # Raw deadline Sat 2026-04-04 -> Easter Sunday -> Easter Monday -> Tue 7.
    result = _calculate(data_decorrenza="2026-03-05", giorni=30)
    assert result.result["scadenza_senza_rinvii"] == "2026-04-04"
    assert result.result["scadenza"] == "2026-04-07"
    assert any(s.get("type") == "holiday_roll" for s in result.steps)


def test_giorni_liberi_adds_one_day_forward():
    plain = _calculate(data_decorrenza="2026-01-12", giorni=10)
    free = _calculate(data_decorrenza="2026-01-12", giorni=10, giorni_liberi=True)
    assert plain.result["scadenza"] == "2026-01-22"
    assert free.result["scadenza"] == "2026-01-23"


def test_backward_term_with_free_days_anticipates_from_saturday():
    result = _calculate(
        data_decorrenza="2026-05-20", giorni=10,
        giorni_liberi=True, termine_a_ritroso=True,
    )
    assert result.result["scadenza_senza_rinvii"] == "2026-05-09"
    assert result.result["scadenza"] == "2026-05-08"
    assert any("a ritroso" in w.message for w in result.warnings)


def test_backward_term_crossing_august_counts_no_feriale_days():
    # 30 days backward from a Sep 15 hearing must skip all of August:
    # Sep 1..14 = 14 counted days, August skipped, Jul 16..31 = 16 more.
    result = _calculate(data_decorrenza="2026-09-15", giorni=30, termine_a_ritroso=True)
    assert result.result["scadenza"] == "2026-07-16"
    assert result.result["giorni_feriale_sospesi"] == 31


def test_deadline_beyond_holiday_calendar_coverage_warns():
    result = _calculate(data_decorrenza="2027-12-01", giorni=60)
    assert result.status == "success"
    assert any("calendario" in w.message for w in result.warnings)


def test_deadline_before_holiday_calendar_coverage_warns():
    # The calendar starts at 2025-01-01: terms touching 2024 must flag that
    # holiday rolling may not have been applied.
    result = _calculate(data_decorrenza="2024-04-01", giorni=30)
    assert result.status == "success"
    assert any("anteriori" in w.message for w in result.warnings)


def test_zero_days_is_rejected():
    result = _calculate(data_decorrenza="2026-01-10", giorni=0)
    assert result.status == "error"
