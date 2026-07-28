"""End-to-end tests for the DRAFT penal-range calculator: the tested
arithmetic core exposed through the real engine, planner and simulated
conversation — mechanics only, counts instead of the legally-gated
circumstance catalog."""

from app.main import engine
from app.schemas.calculation_request import CalculationRequest
from simulation.conversation import SimulatedConversation
from simulation.planner import plan_sentence

CALC_ID = "legal_it.omicidio_pena_draft"


def _calculate(**inputs):
    return engine.calculate(CalculationRequest(calculator_id=CALC_ID, inputs=inputs))


# ---------------------------------------------------------------------------
# Engine level
# ---------------------------------------------------------------------------

def test_plain_homicide_returns_base_range():
    result = _calculate(aggravanti_comuni=0, attenuanti_comuni=0)
    assert result.status == "success"
    assert result.result == {"specie": "reclusione", "pena_minima": "21 anni", "pena_massima": "24 anni"}
    messages = [w.message for w in result.warnings]
    assert any("BOZZA" in m for m in messages)
    assert any("previsione della sentenza" in m for m in messages)


def test_two_aggravating_hit_the_30_year_cap():
    result = _calculate(aggravanti_comuni=2, attenuanti_comuni=0)
    assert result.result["pena_massima"] == "30 anni"  # 24 * (4/3)^2 = 42a8m -> capped
    assert any(s["type"] == "tetto_applicato" for s in result.steps)


def test_one_mitigating_matches_document_envelope():
    result = _calculate(aggravanti_comuni=0, attenuanti_comuni=1)
    assert result.result["pena_minima"] == "14 anni"
    assert result.result["pena_massima"] == "24 anni"


def test_both_sides_without_scenario_return_full_matrix():
    result = _calculate(aggravanti_comuni=1, attenuanti_comuni=1)
    assert result.result["tipo"] == "matrice_scenari_art_69"
    assert result.result["equivalenza"] == {"specie": "reclusione", "pena_minima": "21 anni", "pena_massima": "24 anni"}
    assert result.result["aggravanti_prevalenti"]["pena_massima"] == "30 anni"  # 32 capped
    assert result.result["attenuanti_prevalenti"]["pena_minima"] == "14 anni"


def test_selected_scenario_returns_single_range():
    result = _calculate(aggravanti_comuni=1, attenuanti_comuni=1, scenario_art_69="attenuanti_prevalenti")
    assert result.result == {"specie": "reclusione", "pena_minima": "14 anni", "pena_massima": "24 anni"}


def test_invalid_scenario_is_a_structured_error():
    result = _calculate(aggravanti_comuni=1, attenuanti_comuni=1, scenario_art_69="boh")
    assert result.status == "error"
    assert result.errors[0].code == "strategy_execution_failed"


def test_abbreviato_reduces_by_one_third_after_everything_else():
    result = _calculate(aggravanti_comuni=0, attenuanti_comuni=0, rito_abbreviato=True)
    assert result.result == {"specie": "reclusione", "pena_minima": "14 anni", "pena_massima": "16 anni"}
    assert any(s["type"] == "rito_abbreviato" for s in result.steps)


def test_negative_counts_are_rejected():
    result = _calculate(aggravanti_comuni=-1, attenuanti_comuni=0)
    assert result.status == "error"
    assert result.errors[0].code == "input_invalid"


def test_steps_carry_norm_references():
    result = _calculate(aggravanti_comuni=2, attenuanti_comuni=0)
    norms = {s.get("norm") for s in result.steps if s.get("norm")}
    assert any("63" in n for n in norms)
    assert any("66" in n for n in norms)


# ---------------------------------------------------------------------------
# Planner / conversation level (sentence -> result, as in the UI)
# ---------------------------------------------------------------------------

def test_sentence_with_counts_is_ready_to_calculate():
    plan = plan_sentence(
        "pena per omicidio con 2 aggravanti e 0 attenuanti",
        engine.registry.definitions(),
    )
    assert plan.status == "ready_to_calculate"
    assert plan.calculator_id == CALC_ID
    assert plan.inputs == {"aggravanti_comuni": 2, "attenuanti_comuni": 0}


def test_sentence_without_counts_asks_for_them():
    plan = plan_sentence("che pena rischia per omicidio", engine.registry.definitions())
    assert plan.status == "needs_clarification"
    assert plan.calculator_id == CALC_ID
    assert plan.missing_inputs == ["aggravanti_comuni", "attenuanti_comuni"]


def test_full_conversation_flow_through_clarification():
    conversation = SimulatedConversation(engine)
    first = conversation.send("quanti anni si rischiano per un omicidio?")
    assert first.kind == "question"
    second = conversation.send("2 aggravanti e 0 attenuanti")
    assert second.kind == "answer"
    assert second.calculation.result["pena_massima"] == "30 anni"


# ---------------------------------------------------------------------------
# The other draft offences — routing and fractional base ranges
# ---------------------------------------------------------------------------

def test_furto_routes_and_handles_fractional_base():
    plan = plan_sentence("che pena rischia chi ruba", engine.registry.definitions())
    assert plan.calculator_id == "legal_it.furto_pena_draft"
    result = engine.calculate(CalculationRequest(
        calculator_id="legal_it.furto_pena_draft",
        inputs={"aggravanti_comuni": 0, "attenuanti_comuni": 0},
    ))
    assert result.result["pena_minima"] == "6 mesi"  # exact Fraction(1,2) years
    assert result.result["pena_massima"] == "3 anni"


def test_rapina_routes_and_computes_abbreviato():
    plan = plan_sentence("pena per rapina", engine.registry.definitions())
    assert plan.calculator_id == "legal_it.rapina_pena_draft"
    result = engine.calculate(CalculationRequest(
        calculator_id="legal_it.rapina_pena_draft",
        inputs={"aggravanti_comuni": 0, "attenuanti_comuni": 0, "rito_abbreviato": True},
    ))
    assert result.result["pena_minima"] == "3 anni e 4 mesi"
    assert result.result["pena_massima"] == "6 anni e 8 mesi"


def test_the_three_offences_do_not_steal_each_others_sentences():
    definitions = engine.registry.definitions()
    assert plan_sentence("pena per omicidio volontario", definitions).calculator_id == CALC_ID
    assert plan_sentence("pena per furto", definitions).calculator_id == "legal_it.furto_pena_draft"
    assert plan_sentence("che pena rischia un rapinatore", definitions).calculator_id == "legal_it.rapina_pena_draft"


# ---------------------------------------------------------------------------
# Multa (fine) — reported as its base statutory frame, never adjusted
# ---------------------------------------------------------------------------

def test_furto_reports_base_multa_frame():
    result = engine.calculate(CalculationRequest(
        calculator_id="legal_it.furto_pena_draft",
        inputs={"aggravanti_comuni": 0, "attenuanti_comuni": 0},
    ))
    assert result.result["multa_specie"] == "multa"
    assert result.result["multa_base_minima_eur"] == "154.00"
    assert result.result["multa_base_massima_eur"] == "516.00"
    assert any(s["type"] == "multa_edittale_base" for s in result.steps)


def test_multa_is_not_adjusted_by_circumstances():
    # Aggravanti move the reclusione range but must leave the fine at its
    # base frame — this draft does not model the fine-specific rules.
    result = engine.calculate(CalculationRequest(
        calculator_id="legal_it.furto_pena_draft",
        inputs={"aggravanti_comuni": 2, "attenuanti_comuni": 0},
    ))
    assert result.result["pena_massima"] != "3 anni"  # reclusione did move
    assert result.result["multa_base_massima_eur"] == "516.00"  # fine did not
    assert any("art. 66 n. 3" in w.message for w in result.warnings)


def test_omicidio_has_no_multa():
    result = _calculate(aggravanti_comuni=0, attenuanti_comuni=0)
    assert "multa_specie" not in result.result


# ---------------------------------------------------------------------------
# Aggravated-offence drafts — routing and statutory frames
# ---------------------------------------------------------------------------

def test_furto_aggravato_routes_and_returns_its_frame():
    definitions = engine.registry.definitions()
    assert plan_sentence("furto in casa", definitions).calculator_id == "legal_it.furto_aggravato_draft"
    assert plan_sentence("furto in abitazione", definitions).calculator_id == "legal_it.furto_aggravato_draft"
    result = engine.calculate(CalculationRequest(
        calculator_id="legal_it.furto_aggravato_draft",
        inputs={"aggravanti_comuni": 0, "attenuanti_comuni": 0},
    ))
    assert result.result["pena_minima"] == "2 anni"
    assert result.result["pena_massima"] == "6 anni"
    assert result.result["multa_base_massima_eur"] == "1500.00"


def test_rapina_aggravata_routes_and_returns_its_frame():
    definitions = engine.registry.definitions()
    assert plan_sentence("rapina a mano armata", definitions).calculator_id == "legal_it.rapina_aggravata_draft"
    result = engine.calculate(CalculationRequest(
        calculator_id="legal_it.rapina_aggravata_draft",
        inputs={"aggravanti_comuni": 0, "attenuanti_comuni": 0},
    ))
    assert result.result["pena_minima"] == "7 anni"
    assert result.result["pena_massima"] == "20 anni"


def test_aggravated_forms_never_fall_through_to_the_simple_draft():
    definitions = engine.registry.definitions()
    assert plan_sentence("furto in casa", definitions).calculator_id != "legal_it.furto_pena_draft"
    assert plan_sentence("rapina a mano armata pena", definitions).calculator_id != "legal_it.rapina_pena_draft"


# ---------------------------------------------------------------------------
# Draft caveat is emitted by code, keyed off the "-draft" version
# ---------------------------------------------------------------------------

def test_draft_calculators_carry_a_machine_readable_caveat():
    result = _calculate(aggravanti_comuni=0, attenuanti_comuni=0)
    codes = [w.code for w in result.warnings]
    assert "draft_not_validated" in codes
    # It leads the warning list so a renderer can surface it first.
    assert result.warnings[0].code == "draft_not_validated"
