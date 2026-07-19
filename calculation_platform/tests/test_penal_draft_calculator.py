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
