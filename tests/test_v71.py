"""v71: the thesis checkable against the record; no self-reported legal content -- synthetic records and texts.

1. each thesis declares the elements it contests and relies on (ids) -- none declared: unverifiable, never main;
   contesting an «integrato» one needs the minority label (realismo basso), unlabelled it is barred; a minority thesis
   leads only without a sustainable non-minority line, and with the charge fully integrato the posture table's
   «ammettere la condotta … contenere la pena» always is one; the coherence check reads the ids, never the words
2. the charge from intake («contesta / imputa / chiede la condanna ai sensi di art. N»), the outcome from its record
3. orientamenti rendered by the code: a retrieved ruling with its id and massima, or «orientamento da verificare»
4. the facts from the narrative only; a missing fact names the element it would change
5. no remissione / 162-ter for a d'ufficio charge; a subordinate line equal to the main one omitted
6. the scorer: «non risolto» by default, a quoted and verified sentence to resolve a point, the thesis points from
   the soluzioni table
7. the thesis call logged (logs/thesis_<case>.json) and replayable
"""

import inspect
import json
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT / "scripts"))

from src.rag import intake, lawyer_method as M, posture as posture_mod, render_computed  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

ENTRY = posture_mod.entry_for("penale", "imputato", "primo_grado")
IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado", "entry": ENTRY}
SENTS = [{"id": "S1", "text": "Sempronio a seguito di un acceso diverbio con Caio gli sferra un violento schiaffo al volto.",
          "source": "caso", "certainty": "accertato"},
         {"id": "S2", "text": "Caio cade a terra battendo il capo e muore in ospedale.", "source": "caso",
          "certainty": "accertato"},
         {"id": "S3", "text": "Il candidato assunte le vesti del legale di Sempronio delinea una strategia difensiva.",
          "source": "caso", "certainty": "accertato"},
         {"id": "S4", "text": "Chiunque, con atti diretti a commettere uno dei delitti preveduti dagli articoli 581 e 582, "
                              "cagiona la morte di un uomo, è punito con la reclusione da dieci a diciotto anni.",
          "source": "caso", "certainty": "accertato"},
         {"id": "S5", "text": "Forse Caio aveva una patologia pregressa.", "source": "caso", "certainty": "ipotizzato"}]


def _records(*statuses, alternative=None):
    texts = ["atti diretti a percuotere o ledere", "la morte come conseguenza degli atti"]
    recs = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "charged": 1, "requisites": [
        {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": texts[i - 1], "status": st,
         "supporting": ["S1"] if st == "provato" else [], "question": "q1" if i == 2 else ""}
        for i, st in enumerate(statuses, 1)]}}
    if alternative:
        recs["589 c.p."] = {"rubrica": "Omicidio colposo", "side": "weapon", "requisites": [
            {"id": "R9", "comma": 1, "kind": "requisito", "text": "condotta colposa", "status": alternative}]}
    return recs


def _ctx(recs, rows=(), questions=(), posture=IMPUTATO, rulings=()):
    return M.prepare({"records": recs, "sentences": SENTS}, posture, list(rows), "Nel 2025 Sempronio colpisce Caio.",
                     "Sempronio", "", list(recs), list(rulings), [], [], [], link_call=lambda *a: {},
                     questions=list(questions), case_id_="sempronio_test")


# --- 1. the thesis by its declared ids --------------------------------------------------------------------------------------

def test_a_thesis_contesting_an_integrated_element_is_barred_and_the_next_sustainable_one_is_main():
    recs = _records("provato", "provato", alternative="da provare")
    a = M.assessment(recs, IMPUTATO)
    plan = M.clean_method({"soluzioni": [
        {"tesi": "Manca il nesso causale tra lo schiaffo e la morte", "contesta": ["R2"], "realismo": "alto"},
        {"tesi": "Riqualificazione in omicidio colposo", "contesta": [], "poggia_su": ["R9"], "realismo": "basso",
         "minoritaria": True},
        {"tesi": "Pena vicina al minimo", "contesta": [], "poggia_su": ["R1"], "realismo": "medio"}],
        "tesi_principale": 1})
    log = []
    sols, main = M.coherent_solutions(plan, recs, a, (), log, IMPUTATO)
    assert sols[0]["barred"] == ["R2"] and sols[0]["realismo"] == "basso"
    assert sols[main]["tesi"] == "Pena vicina al minimo"          # sustainable, non-minority first
    table = M.solutions_section(sols, main, plan, [], "x")
    assert f"Manca il nesso causale tra lo schiaffo e la morte ({M.BARRED_LABEL}: R2)" in table
    assert log[0]["barred"] == ["R2"]
    # words never decide: a thesis saying the opposite of an integrated element but declaring nothing is not barred
    plan = M.clean_method({"soluzioni": [{"tesi": "Gli atti non erano diretti a percuotere", "realismo": "medio"}]})
    assert not M.coherent_solutions(plan, recs, a, (), None, IMPUTATO)[0][0].get("barred")


def test_every_thesis_barred_falls_back_to_the_posture_tables_line():
    recs = _records("provato", "provato")
    a = M.assessment(recs, IMPUTATO)
    plan = M.clean_method({"soluzioni": [{"tesi": "Assoluzione", "contesta": ["R1"]},
                                         {"tesi": "Nesso interrotto", "contesta": ["R2"]}]})
    sols, main = M.coherent_solutions(plan, recs, a, ["Giudizio abbreviato: x"], None, IMPUTATO)
    assert sols[main]["tesi"] == "Ammettere la condotta, contestare l'inquadramento / contenere la pena"
    assert sols[main]["fallback"] and ENTRY["fallback_main"] == "ammettere la condotta, contestare l'inquadramento / " \
                                                                "contenere la pena"


def test_the_prompt_lists_the_element_ids_and_asks_for_them():
    ctx = _ctx(_records("provato", "da provare"))
    ctx["candidates"] = M.thesis_candidates(ctx)
    material = M.method_material(ctx)
    assert "· [R1] atti diretti a percuotere o ledere: integrato" in material
    assert "- [T1] (sostenibile) L'elemento «la morte come conseguenza degli atti» (R2) dell'art. 584 c.p. non è " \
           "provato — contesta R2" in material                   # v72: the code's candidates, with their ids
    assert "Non aggiungere tesi e non cambiarle" in M.METHOD_SYSTEM and "la sezione la scrive il codice" in M.METHOD_SYSTEM


def test_undeclared_or_unknown_ids_are_dropped_and_logged(tmp_path, monkeypatch):
    monkeypatch.setenv(M.THESIS_LOG_ENV, str(tmp_path))
    ctx = _ctx(_records("provato", "provato"))
    plan = M.method_plan(ctx, call=lambda messages: {"soluzioni": [{"tesi": "T", "contesta": ["R2"]}],
                                                     "valutazioni": {"T9": {"realismo": "alto"}}})
    # v72: a thesis of the model's own -- a new key, or a «soluzioni» list -- never reaches the table
    assert all(t.get("candidate") for t in plan["soluzioni"]) and "T" not in [t["tesi"] for t in plan["soluzioni"]]
    log = json.loads((tmp_path / "thesis_sempronio_test.json").read_text(encoding="utf-8"))
    assert log["samples"][0]["why"] and "added ['T9']" in log["samples"][0]["why"]


def test_a_thesis_declaring_no_element_is_unverifiable_and_never_main():
    recs = _records("provato", "da provare")
    plan = M.clean_method({"soluzioni": [{"tesi": "Tesi senza elementi", "realismo": "alto"},
                                         {"tesi": "Tesi dichiarata", "poggia_su": ["R1"], "realismo": "basso"}]})
    log = []
    sols, main = M.coherent_solutions(plan, recs, M.assessment(recs, IMPUTATO), (), log, IMPUTATO)
    assert sols[main]["tesi"] == "Tesi dichiarata" and log == [{"tesi": "Tesi senza elementi", "unverifiable": True}]
    assert f"Tesi senza elementi ({M.UNVERIFIABLE_LABEL})" in M.solutions_section(sols, main, plan, [], "x")


def test_contesting_an_integrated_element_needs_the_minority_label():
    recs = _records("provato", "da provare")
    a = M.assessment(recs, IMPUTATO)
    plan = M.clean_method({"soluzioni": [
        {"tesi": "Manca il dolo di percosse", "contesta": ["R1"], "realismo": "alto"},
        {"tesi": "Riqualificazione in omicidio colposo", "contesta": ["R1"], "realismo": "medio", "minoritaria": True}]})
    sols, main = M.coherent_solutions(plan, recs, a, (), None, IMPUTATO)
    assert sols[0]["barred"] == ["R1"] and not sols[1].get("barred")
    assert sols[1]["contests_integrated"] == ["R1"] and sols[1]["realismo"] == "basso"
    assert main == 1                    # the charge is not fully integrato: no containment line is due first
    table = M.solutions_section(sols, main, plan, [], "x")
    assert f"(tesi principale) (tesi minoritaria) ({M.INTEGRATED_NOTE}: R1)" in table


def test_the_lawyers_case_containment_leads_and_his_minority_thesis_stays_labelled():
    # the charge fully integrato: «ammettere la condotta, contenere la pena» is a sustainable non-minority line, so
    # the riqualificazione 589 -- declared minoritaria, contesting an integrated element -- is not the main one
    recs = _records("provato", "provato")
    a = M.assessment(recs, IMPUTATO)
    plan = M.clean_method({"soluzioni": [
        {"tesi": "Riqualificazione in omicidio colposo (art. 589)", "contesta": ["R1"], "realismo": "basso",
         "minoritaria": True},
        {"tesi": "Assenza di nesso causale", "contesta": ["R2"], "realismo": "medio"}], "tesi_principale": 1})
    sols, main = M.coherent_solutions(plan, recs, a, ["Giudizio abbreviato: x"], None, IMPUTATO)
    assert sols[main]["tesi"].startswith("Ammettere la condotta") and main == 0
    assert sols[1]["contests_integrated"] == ["R1"] and sols[2]["barred"] == ["R2"]
    assert M.settle_objective(a, sols, recs)["obiettivo"] == M.CONTENIMENTO     # a minority thesis is no objective
    import rubric_score as R
    crit = json.loads((ROOT / "docs" / "lawyer" / "critiques" / "sempronio_no_hint.critique.json").read_text("utf-8"))
    rows = R.solutions_rows("\n".join(["**6. Soluzioni a confronto**", ""] +
                                       M.solutions_section(sols, main, plan, [], "x").split("\n")))
    exp = crit["tesi_attesa"]
    assert R.thesis_outcome("principale", exp["principale"], rows)[0] == "risolto"
    assert R.thesis_outcome("esclusa", exp["esclusa"], rows)[0] == "risolto"


# --- 2. the charge from intake, the outcome from its record ---------------------------------------------------------------

@pytest.mark.parametrize("text, articles", [
    ("Il PM contesta a Sempronio il reato di cui all'art. 584 c.p.", ["584 c.p."]),
    ("Sempronio è imputato di lesione personale aggravata, a norma degli articoli 582 - 583, comma 1, numero 1 – 585 "
     "del codice penale.", ["582 c.p.", "583 c.p.", "585 c.p."]),
    ("Il pubblico ministero chiede la condanna di Tizio ai sensi dell'art. 572 c.p.", ["572 c.p."]),
])
def test_a_charge_sentence_is_a_contestazione_with_its_articles(text, articles):
    found = intake.prior_decisions(text)
    assert found[0]["outcome"] == posture_mod.CHARGE_OUTCOME and found[0]["articles"] == articles
    assert posture_mod.intake_charges({"prior_decisions": found}) == articles
    assert intake.prior_decisions("Sempronio rischia una contestazione.") == []          # no article: no charge


def test_a_charge_is_no_decision_for_the_appeals_date():
    posture = {"rito": "penale", "role": "imputato", "stage": "impugnazione", "entry": None, "prior_decisions": [
        {"court": "", "outcome": "contestazione", "date": "3 marzo 2024", "articles": ["584 c.p."]}]}
    text = "\n".join(b for _, b in render_computed.stage_blocks(posture))
    assert "3 marzo 2024" not in text


def test_the_outcome_follows_the_charge_and_full_only_with_a_non_integrated_element():
    all_in = M.assessment(_records("provato", "provato"), IMPUTATO)
    assert (all_in["esito"], all_in["obiettivo"], all_in["all_integrated"]) == (M.SFAVOREVOLE, M.CONTENIMENTO, True)
    open_ = M.assessment(_records("provato", "non risulta"), IMPUTATO)
    assert open_["obiettivo"] == M.PIENO
    none = M.assessment({}, IMPUTATO)
    assert none["obiettivo"] != M.PIENO
    recs = _records("provato", "provato", alternative="da provare")
    sols = [{"tesi": "Riqualificazione in 589", "contesta": [], "poggia_su": ["R9"]}]
    assert M.settle_objective(all_in, sols, recs)["obiettivo"] == M.PARZIALE
    assert M.settle_objective(all_in, [{**sols[0], "barred": ["R1"]}], recs)["obiettivo"] == M.CONTENIMENTO


# --- 3. orientamenti by the code ------------------------------------------------------------------------------------------

def test_the_orientations_are_the_codes_and_the_writers_are_never_printed():
    recs = _records("provato", "provato")
    rulings = [{"id": "doc-22676", "label": "Cass. pen., Sez. Un., n. 22676/2009",
                "massima": "Nella preterintenzione la morte deve essere conseguenza degli atti diretti a percuotere."}]
    ctx = _ctx(recs, rulings=rulings)
    plan = M.clean_method({
        "orientamenti": [{"punto": "prevedibilità dell'evento", "pronuncia": "doc-22676",
                          "orientamento": "la giurisprudenza prevalente esclude la prevedibilità"},
                         {"punto": "causa sopravvenuta", "pronuncia": ""}],
        "conclusioni": "Secondo la giurisprudenza prevalente il nesso regge. La pena va contenuta.",
        "obiezioni": [{"obiezione": "o", "risposta": "La Cassazione ammette la tesi. Si risponde sui fatti."}]})
    log = {}
    answer = M.compose([("Norme applicate ai fatti", "Le Sezioni Unite hanno chiarito il punto. La norma vale.")], ctx,
                       plan, log=log)
    by = dict(M.answer_sections(answer))
    orient = by["Orientamenti rilevanti"]
    assert orient.split("\n") == [
        "- prevedibilità dell'evento: Cass. pen., Sez. Un., n. 22676/2009 (documento doc-22676): «Nella preterintenzione "
        "la morte deve essere conseguenza degli atti diretti a percuotere.»",
        "- causa sopravvenuta: orientamento da verificare"]
    assert "prevalente" not in answer and "Sezioni Unite hanno" not in answer and "Cassazione ammette" not in answer
    assert "Si risponde sui fatti." in answer
    assert "causa sopravvenuta" not in by["Punti da verificare"]       # v75b: §10 only the open questions; it stays in §5
    checks = {c["check"]: c["ok"] for c in M.structure_checks(answer, IMPUTATO)}
    assert checks["orientations_code_rendered"] and checks["markers_collected"]
    assert len(log["orientation_claims_out"]) == 2       # v72: the conclusions are the code's, never the model's


# --- 4. the facts from the narrative --------------------------------------------------------------------------------------

def test_the_facts_are_the_narratives_and_the_missing_name_their_element():
    recs = _records("provato", "da provare")
    recs["584 c.p."]["requisites"][1]["missing"] = "la causa della caduta"
    questions = [{"id": "q1", "text": "Caio aveva patologie pregresse?"},
                 {"id": "q9", "text": "Sempronio ha precedenti penali?"}]
    rows = [{"_article": ("584", "c.p."), "s": {"plain_text": SENTS[3]["text"]}}]
    log = {}
    split = M.facts_split(SENTS, recs, True, questions, [r["s"]["plain_text"] for r in rows], log)
    assert split["certi"] == [SENTS[0]["text"], SENTS[1]["text"]]          # not the ask, not the article's text
    assert split["ipotizzabili"] == [{"text": SENTS[4]["text"], "how": "ipotesi"}]
    assert [m["elemento"] for m in split["mancanti"]] == ["R2"]
    assert "l'elemento R2 dell'art. 584 c.p." in split["mancanti"][0]["cambierebbe"]
    assert log["questions_without_element"] == ["Sempronio ha precedenti penali?"]
    assert any("Il candidato" in x for x in log["excluded_sentences"])
    recs["584 c.p."]["requisites"][1]["status"] = "provato"
    split = M.facts_split(SENTS, recs, True, questions[:1])
    assert [m["fatto"] for m in split["mancanti"]] == ["Caio aveva patologie pregresse?"]
    assert split["mancanti"][0]["elemento"] == "R2"


# --- 5. procedibilità, the subordinate line -------------------------------------------------------------------------------

def test_no_remissione_or_162_ter_for_a_dufficio_charge():
    rows = [{"_article": ("584", "c.p."), "_proc": {"querela": False}},
            {"_article": ("590", "c.p."), "_proc": {"querela": True}}]
    recs = _records("provato", "provato")
    recs["590 c.p."] = {"requisites": [], "side": "weapon"}
    proc = M.procedibility(recs, rows)
    assert proc == {"584 c.p.": "ufficio", "590 c.p.": "querela"}
    line = M.stragiudiziale_line(IMPUTATO, "ufficio")
    assert "162-ter" not in line and "remission" not in line.lower()
    assert "162-ter" in M.stragiudiziale_line(IMPUTATO, "querela")
    out = []
    text = ("La remissione della querela estingue il reato. Per l'art. 590 c.p. la remissione della querela è ammessa. "
            "Le condotte riparatorie (art. 162-ter c.p.) estinguono il reato. La pena va contenuta.")
    kept = M.procedibility_out(text, proc, "584 c.p.", out)
    assert kept == "Per l'art. 590 c.p. la remissione della querela è ammessa. La pena va contenuta."
    assert len(out) == 2
    ctx = _ctx(recs, rows=rows)
    assert ctx["procedibilita"]["584 c.p."] == "ufficio" and "162-ter" not in ctx["stragiudiziale"]


def test_a_subordinate_line_equal_to_the_main_one_is_omitted():
    assert M.same_line("Contenere la pena con le attenuanti generiche", "contenere la pena: attenuanti generiche")
    assert not M.same_line("Riqualificazione in omicidio colposo", "Contenere la pena con le attenuanti generiche")
    sols = [{"tesi": "Contenere la pena con le attenuanti generiche", "realismo": "alto"}]
    log = {}
    text = M.solutions_section(sols, 0, {"subordinata": "contenere la pena: attenuanti generiche"}, [], "x", log)
    assert "Linea subordinata" not in text and log["subordinate_omitted"]
    fallback = [{**M.fallback_thesis({}, ["Giudizio abbreviato: riduzione"], IMPUTATO)}]
    text = M.solutions_section(fallback, 0, {"subordinata": ""}, ["Giudizio abbreviato: riduzione"], "x")
    assert "- Linea principale: Ammettere la condotta" in text and "  - Giudizio abbreviato" in text
    assert "Linea subordinata" not in text


# --- 6. the scorer --------------------------------------------------------------------------------------------------------

ANSWER = """**6. Soluzioni a confronto**

| Tesi | Pro | Contro | Realismo |
|---|---|---|---|
| Riqualificazione in omicidio colposo (art. 589 c.p.) (tesi principale) (tesi minoritaria) | a | b | basso |
| Assenza di nesso causale (tesi che nega un elemento integrato (non sostenibile): R2) | a | b | basso |

L'attenuante della provocazione richiede un fatto ingiusto altrui, che il caso non descrive.
"""


def test_a_point_is_resolved_only_with_a_verified_quote():
    import rubric_score as R
    given = [{"id": "3.5", "categoria": "coerenza", "punto": "62 n. 2"},
             {"id": "3.4", "categoria": "coerenza", "punto": "soccorso"},
             {"id": "6.2", "categoria": "pregio", "punto": "prove"}]
    raw = {"punti": {"3.5": {"esito": "risolto", "citazione": "L’attenuante della **provocazione** richiede un fatto "
                                                             "ingiusto altrui"},
                     "3.4": {"esito": "risolto", "citazione": "Il soccorso è valorizzato come attenuante generica."},
                     "6.2": {"esito": "mantenuto"}},
           "coerenza": {"score": 1}}
    got = R.clean_critique(raw, given, ANSWER)
    by = {p["id"]: p for p in got["points"]}
    assert by["3.5"]["esito"] == "risolto" and by["3.5"]["come"] == "citazione verificata"
    assert by["3.4"]["esito"] == "non risolto" and by["3.4"]["modello"]["esito"] == "risolto"
    assert by["6.2"]["esito"] == "non risolto"
    assert got["scores"]["coerenza"]["score"] == 0.5 and got["model_scores"]["coerenza"]["score"] == 1.0


def test_the_thesis_points_are_scored_from_the_soluzioni_table():
    import rubric_score as R
    crit = json.loads((ROOT / "docs" / "lawyer" / "critiques" / "sempronio_no_hint.critique.json").read_text("utf-8"))
    exp = crit["tesi_attesa"]
    rows = R.solutions_rows(ANSWER)
    assert rows[0]["principale"] and rows[0]["minoritaria"] and rows[1]["barred"]
    assert R.thesis_outcome("principale", exp["principale"], rows)[0] == "risolto"
    assert R.thesis_outcome("esclusa", exp["esclusa"], rows)[0] == "risolto"
    swapped = ANSWER.replace(" (tesi principale) (tesi minoritaria)", "").replace(
        "| Assenza di nesso causale (tesi che nega un elemento integrato (non sostenibile): R2)",
        "| Assenza di nesso causale (tesi principale)")
    rows = R.solutions_rows(swapped)
    assert R.thesis_outcome("esclusa", exp["esclusa"], rows)[0] == "non risolto"
    assert R.thesis_outcome("principale", exp["principale"], rows)[0] == "parziale"      # present, not main
    unlabelled = ANSWER.replace(" (tesi minoritaria)", "")
    assert R.thesis_outcome("principale", exp["principale"], R.solutions_rows(unlabelled))[0] == "parziale"
    bare = ANSWER.replace(" (tesi che nega un elemento integrato (non sostenibile): R2)", "")
    assert R.thesis_outcome("esclusa", exp["esclusa"], R.solutions_rows(bare))[0] == "non risolto"   # unlabelled
    labelled = ANSWER.replace(" (tesi che nega un elemento integrato (non sostenibile): R2)", " (tesi minoritaria)")
    assert R.thesis_outcome("esclusa", exp["esclusa"], R.solutions_rows(labelled))[0] == "risolto"
    got = R.critique_score(crit, "", ANSWER, caller=lambda s, h: {"punti": {"1.1": {"esito": "ripetuto"}},
                                                                 "tesi": {"score": 0}})
    by = {p["id"]: p for p in got["points"]}
    assert by["1.1"]["esito"] == "risolto" and by["7.2"]["esito"] == "risolto"
    assert got["scores"]["tesi"]["score"] == 1.0 and got["model_scores"]["tesi"]["score"] == 0.0


# --- 7. the thesis call, logged and replayable ----------------------------------------------------------------------------

def test_the_thesis_call_is_logged_and_replays_to_the_same_check(tmp_path, monkeypatch):
    import thesis_replay
    monkeypatch.setenv(M.THESIS_LOG_ENV, str(tmp_path))
    ctx = _ctx(_records("provato", "provato"))
    reply = {"soluzioni": [{"tesi": "Nesso interrotto", "contesta": ["R2"], "realismo": "alto"},
                           {"tesi": "Pena al minimo", "contesta": [], "poggia_su": ["R1"], "realismo": "medio"}]}
    M.method_plan(ctx, call=lambda messages: reply)
    path = tmp_path / "thesis_sempronio_test.json"
    assert ctx["thesis_log"] == str(path)
    log = json.loads(path.read_text(encoding="utf-8"))
    assert [m["role"] for m in log["messages"]] == ["system", "user"] and "FASE 6" in log["messages"][0]["content"]
    assert log["elements"]["R2"]["status"] == "provato" and log["parsed"] == reply
    checked = thesis_replay.check(log["parsed"], log)
    # v72: the log carries the code's candidates -- the reply's own theses are dropped, the charge fully integrato
    assert checked["main"].startswith("Ammettere la condotta") and checked["dropped"] == ["soluzioni"]
    assert M.case_id({"case": "x"}).startswith("h") and M.case_id({"case_id": "a b/c"}) == "a_b_c"
    assert "case_id" in inspect.signature(flow.single_shot).parameters
