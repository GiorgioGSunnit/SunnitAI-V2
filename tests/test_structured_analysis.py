"""STRUCTURED_ANALYSIS_ENABLED (handover 19 Oct, 6.3): the case analysis in two
short calls, the facts sentence by sentence and then the area's grid, returned
in _analyse_case_v2's shape. Offline: the model's replies are mocked; the cases
are synthetic, never the held-out ones."""

import json
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402

CASE = ("Tizio, durante una lite, colpisce Caio con un pugno. Caio cade e batte la testa. "
        "Tizio chiama i soccorsi; Caio muore in ospedale. Il PM contesta l'art. 584 c.p. a Tizio.")

FACTS_REPLY = {
    "parte_assistita": "Tizio", "posizione": "indagato", "area": "penale",
    "richiesta": "strategia difensiva", "ricerca_fatti": ["lite", "pugno", "caduta", "trauma cranico"],
    "fatti": [
        {"frase": 1, "fatto": "Tizio colpisce Caio con un pugno durante una lite", "tipo": "condotta",
         "effetto": "sfavorevole", "perche": "condotta violenta"},
        {"frase": 2, "fatto": "Caio cade e batte la testa", "tipo": "evento", "effetto": "sfavorevole",
         "perche": "evento lesivo"},
        {"frase": 3, "fatto": "Tizio chiama i soccorsi", "tipo": "condotta_successiva", "effetto": "favorevole",
         "perche": "soccorso"},
        {"frase": 3, "fatto": "Caio muore in ospedale", "tipo": "evento", "effetto": "sfavorevole",
         "perche": "morte"},
    ],
}
GRID_REPLY = [
    {"voce": "qualificazione", "pertinente": True, "valutazione": "pugno seguito da morte non voluta",
     "istituti": [{"nome": "omicidio preterintenzionale", "codice": "c.p.", "numero_indicativo": "584"}],
     "parole_chiave": ["percosse", "lesioni", "evento morte"]},
    {"voce": "elemento_soggettivo", "pertinente": True, "valutazione": "dolo di percosse",
     "istituti": [{"nome": "elemento psicologico del reato", "codice": "c.p."}],
     "parole_chiave": ["dolo", "preterintenzione"]},
    {"voce": "nesso_causale", "pertinente": True, "valutazione": "caduta dopo il pugno",
     "istituti": [{"nome": "concorso di cause", "codice": "c.p."}], "parole_chiave": ["causa sopravvenuta"]},
    {"voce": "forma_del_reato", "pertinente": False, "valutazione": "", "istituti": [], "parole_chiave": []},
    {"voce": "cause_di_giustificazione", "pertinente": False,
     "istituti": [{"nome": "legittima difesa", "codice": "c.p."}]},           # not pertinent: ignored anyway
    {"voce": "qualificazioni_alternative", "pertinente": True, "valutazione": "morte per colpa",
     "istituti": [{"nome": "omicidio colposo", "codice": "c.p."}, {"nome": "  "}],
     "parole_chiave": ["colpa"]},
    {"voce": "voce_inventata", "pertinente": True, "istituti": [{"nome": "rapina", "codice": "c.p."}]},
]


@pytest.fixture
def model(monkeypatch):
    calls = []

    def fake(messages, max_tokens=None, stop=None, temperature=None):
        system, human = messages[0].content, messages[-1].content
        calls.append((system, human))
        if "frasi numerate" in system:
            return json.dumps(FACTS_REPLY)
        if "griglia" in system:
            return json.dumps(GRID_REPLY)
        if "impostazione di un parere" in system:                           # _analyse_case_v2, the fallback
            return json.dumps({"area": "penale", "questioni": [{"titolo": "v2", "istituti": []}], "fatti": []})
        raise AssertionError("unexpected call")

    monkeypatch.setattr(dg, "_call_chat", fake)
    return calls


def test_sentences_are_numbered_without_splitting_article_references():
    sentences = dg._numbered_sentences(CASE)
    assert sentences[-1] == "Il PM contesta l'art. 584 c.p. a Tizio."
    assert len(sentences) == 5                                             # "; Caio" starts a new one
    assert dg._numbered_sentences("Primo fatto.\nSecondo fatto; terzo") == ["Primo fatto.", "Secondo fatto; terzo"]


def test_the_analysis_has_the_v2_shape_from_the_pertinent_slots_only(model):
    analysis = dg._analyse_case_structured(CASE, "it")
    assert [q["voce"] for q in analysis["questioni"]] == [
        "qualificazione", "elemento_soggettivo", "nesso_causale", "qualificazioni_alternative"]
    first = analysis["questioni"][0]
    assert first["titolo"] == "Qualificazione del fatto: pugno seguito da morte non voluta"
    assert first["categoria"] == "qualificazione"
    assert first["istituti"] == [{"nome": "omicidio preterintenzionale", "codice": "c.p.", "numero_indicativo": ""}]
    assert analysis["questioni"][3]["categoria"] == "qualificazione_alternativa"
    assert analysis["questioni"][3]["istituti"] == [{"nome": "omicidio colposo", "codice": "c.p.",
                                                     "numero_indicativo": ""}]
    assert analysis["parte_assistita"] == "Tizio" and analysis["area"] == "penale"
    assert analysis["ricerca_fatti"] == ["lite", "pugno", "caduta", "trauma cranico"]
    assert analysis["fatti"][2] == {"fatto": "Tizio chiama i soccorsi", "effetto": "favorevole",
                                    "perche": "soccorso"}
    assert analysis["fatti_strutturati"]["condotta_successiva"] == "Tizio chiama i soccorsi"


def test_two_calls_numbered_sentences_then_the_penal_grid(model):
    dg._analyse_case_structured(CASE, "it")
    (facts_system, facts_human), (grid_system, grid_human) = model
    assert facts_human.startswith("[1] Tizio, durante una lite") and "[5] Il PM contesta" in facts_human
    assert "- Tizio chiama i soccorsi" in grid_human and "- nesso_causale:" in grid_human
    assert "- rapporto:" not in grid_human                                  # the civil grid is not asked
    assert "Mai un numero di articolo" in grid_system and "nel dubbio è false" in grid_system


@pytest.mark.parametrize("area, grid", [("penale", "penale"), ("lavoro", "civile"), ("tributario", "amministrativo"),
                                        ("qualcosa", "civile")])
def test_the_grid_follows_the_area(monkeypatch, model, area, grid):
    monkeypatch.setitem(FACTS_REPLY, "area", area)
    dg._analyse_case_structured("Un fatto qualunque.", "it")
    first_slot = dg._STRUCTURED_GRIDS[grid][0][0]
    assert f"- {first_slot}:" in model[1][1]


@pytest.mark.parametrize("facts_reply", ["not json at all", json.dumps({"area": "penale", "fatti": []})])
def test_falls_back_to_the_v2_analysis_without_facts(monkeypatch, facts_reply):
    def fake(messages, **kwargs):
        system = messages[0].content
        if "frasi numerate" in system:
            return facts_reply
        if "impostazione di un parere" in system:
            return json.dumps({"area": "penale", "questioni": [{"titolo": "v2", "istituti": []}], "fatti": []})
        raise AssertionError("the grid must not be asked without facts")

    monkeypatch.setattr(dg, "_call_chat", fake)
    assert [q["titolo"] for q in dg._analyse_case_structured(CASE, "it")["questioni"]] == ["v2"]


def test_falls_back_to_the_v2_analysis_when_nothing_is_pertinent(monkeypatch):
    nothing = [{"voce": slot, "pertinente": False} for slot, *_ in dg._STRUCTURED_GRIDS["penale"]]

    def fake(messages, **kwargs):
        system = messages[0].content
        if "frasi numerate" in system:
            return json.dumps(FACTS_REPLY)
        if "griglia" in system:
            return json.dumps(nothing)
        return json.dumps({"area": "penale", "questioni": [{"titolo": "v2", "istituti": []}], "fatti": []})

    monkeypatch.setattr(dg, "_call_chat", fake)
    assert [q["titolo"] for q in dg._analyse_case_structured(CASE, "it")["questioni"]] == ["v2"]


def test_the_pipeline_chooses_the_analysis_by_flag(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "_analyse_case", lambda *a, **k: seen.append("v1") or {})
    monkeypatch.setattr(dg, "_analyse_case_v2", lambda *a, **k: seen.append("v2") or {})
    monkeypatch.setattr(dg, "_analyse_case_structured", lambda *a, **k: seen.append("structured") or {})
    for case_analysis, structured in (("false", "true"), ("true", "false"), ("true", "true")):
        monkeypatch.setenv("CASE_ANALYSIS_ENABLED", case_analysis)
        monkeypatch.setenv("STRUCTURED_ANALYSIS_ENABLED", structured)
        dg._analyse_case_for_pipeline(CASE, "it")
    assert seen == ["v1", "v2", "structured"]


def test_the_penal_grid_states_the_rules_found_on_production():
    """20 Oct, Sempronio (prod): "reato consumato" filled the form slot and pulled
    in 158/270-ter; "Circostanze attenuanti" matched art. 114 (concorso) by its
    exact title; the alternative was "lesioni personali aggravate" after a death,
    so 589 was missing."""
    grid = {slot: question for slot, _, question, _ in dg._STRUCTURED_GRIDS["penale"]}
    assert "una sola persona non pone questa voce" in grid["forma_del_reato"]
    assert "non della" in grid["circostanze"] and "categoria" in grid["circostanze"]
    assert "mai una lesione personale" in grid["qualificazioni_alternative"]
    assert "nome specifico dell'istituto" in dg._STRUCTURED_GRID_SYSTEM
