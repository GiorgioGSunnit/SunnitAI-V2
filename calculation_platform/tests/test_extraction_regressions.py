"""Regression tests for the extraction/routing failures reported from the
/simulate/chat testing pass. Each test pins the fixed behavior so the old
defect (a confident wrong number, a dropped input, a mis-route) cannot return.

The simulation layer is a dev fixture, not production — but it must never
produce a confident wrong legal number; when it cannot confidently associate a
value with a field it asks instead.
"""

from app.core.matcher import match_query
from app.main import engine
from app.schemas.calculation_request import CalculationRequest
from simulation.conversation import SimulatedConversation
from simulation.planner import plan_sentence


def _defs():
    return engine.registry.definitions()


def _conv():
    return SimulatedConversation(engine)


# --- #1 Positional binding no longer maps unlabeled numbers by position ------

def test_preavviso_does_not_bind_unlabeled_numbers_positionally():
    reply = _conv().send(
        "Indennità di mancato preavviso: ho lavorato 11 anni e 7 mesi, "
        "retribuzione mensile 2500 euro"
    )
    # The labeled value binds; the unlabeled 11/7 (seniority, not preavviso)
    # do NOT — so instead of computing 11 x 7 = 77 the fixture asks.
    assert reply.kind == "question"
    assert reply.tool_call.inputs.get("retribuzione_mensile_globale") == "2500"
    assert "mesi_preavviso" not in reply.tool_call.inputs


# --- #2 IMU: per-mille, cadastral code, named association --------------------

def test_imu_per_mille_and_cadastral_code_are_handled():
    reply = _conv().send(
        "Quanto pago di IMU categoria A/3 con rendita 850, moltiplicatore 160, "
        "aliquota 10,6 per mille?"
    )
    assert reply.kind == "answer"
    inputs = reply.tool_call.inputs
    assert inputs["rendita_catastale"] == "850"
    assert inputs["moltiplicatore"] == "160"
    assert inputs["aliquota"] == "0.0106"           # per-mille, not per-cent
    assert "3" not in inputs.values()                # A/3 is not a value
    assert reply.calculation.result["imposta_dovuta"] == "1513.68"


# --- #3 Optional numeric inputs are captured when supplied -------------------

def test_tfr_optional_inputs_are_extracted():
    reply = _conv().send(
        "Calcolo TFR: retribuzione lorda annua 36000, fondo precedente 20000, "
        "inflazione ISTAT 1,2%"
    )
    assert reply.kind == "answer"
    inputs = reply.tool_call.inputs
    assert inputs["retribuzione_lorda_annua"] == "36000"
    assert inputs["fondo_precedente"] == "20000"
    assert inputs["tasso_inflazione_istat"] == "0.012"


# --- #4 string_list (fasi) is extracted from a follow-up --------------------

def test_dm55_string_list_fasi_binds_and_valore_causa_is_not_the_decree_number():
    conv = _conv()
    first = conv.send("Compenso avvocato DM 55 per una causa da 30000 euro")
    assert first.tool_call.inputs["valore_causa"] == "30000"   # not 55
    second = conv.send("studio, introduttiva, istruttoria, decisionale")
    assert second.kind == "answer"
    assert second.tool_call.inputs["fasi"] == [
        "studio", "introduttiva", "istruttoria", "decisionale",
    ]


# --- #5 Natural-language dates and explicit booleans ------------------------

def test_procedural_deadline_natural_date_and_booleans():
    reply = _conv().send(
        "Il termine è di 30 giorni dalla notifica del 28 luglio 2025, non sono "
        "giorni liberi, si applica la sospensione feriale e non è un termine a ritroso."
    )
    assert reply.kind == "answer"
    inputs = reply.tool_call.inputs
    assert inputs["data_decorrenza"] == "2025-07-28"
    assert inputs["giorni"] == 30
    assert inputs["giorni_liberi"] is False
    assert inputs["sospensione_feriale"] is True
    assert inputs["termine_a_ritroso"] is False
    assert reply.calculation.result["scadenza"] == "2025-09-29"


# --- #6 Unsupported topics must not route to a calculator -------------------

def test_succession_question_does_not_route():
    assert match_query("come si divide l'eredità tra i figli", _defs()).status == "no_match"


def test_personal_injury_question_does_not_route():
    assert match_query("risarcimento del danno biologico permanente", _defs()).status == "no_match"


def test_legittima_question_does_not_route():
    assert match_query("quanto spetta di legittima al coniuge superstite", _defs()).status == "no_match"


# --- #7 Combined revaluation + interest routes to the combined calculator ----

def test_combined_revaluation_and_interest_prefers_the_combined_calculator():
    response = match_query(
        "rivaluta 1000 euro dal 2020 al 2024 e calcola gli interessi legali", _defs()
    )
    assert response.status == "matched"
    assert response.candidates[0].calculator_id == "legal_it.rivalutazione_interessi_1712"


def test_revaluation_only_and_interest_only_still_route_to_their_own_calculators():
    assert match_query("calcola solo la rivalutazione monetaria di 5000 euro", _defs()) \
        .candidates[0].calculator_id == "legal_it.rivalutazione_istat"
    assert match_query("calcola gli interessi legali su 5000 euro", _defs()) \
        .candidates[0].calculator_id == "legal_it.legal_interest"


# --- #8 Legal interest coverage now spans 2021-2026 -------------------------

def test_legal_interest_covers_2021_through_2026():
    expected = {
        2021: "0.0001", 2022: "0.0125", 2023: "0.05",
        2024: "0.025", 2025: "0.02", 2026: "0.016",
    }
    for year, value in expected.items():
        pv = engine.parameter_store.resolve_by_tax_year("legal_it.legal_interest_rate", year)
        assert str(pv.value) == value, f"{year}: {pv.value} != {value}"


def test_legal_interest_pre_2021_is_refused_not_guessed():
    result = engine.calculate(CalculationRequest(
        calculator_id="legal_it.legal_interest",
        inputs={"capital": 10000},
        period={"start_date": "2019-01-01", "end_date": "2019-12-31"},
    ))
    assert result.status == "error"
    assert result.errors[0].code == "parameter_unresolved"
