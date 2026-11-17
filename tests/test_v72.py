"""v72: the theses enumerated by the code, argued by the model -- synthetic records and texts.

1. thesis_candidates: per-element contests, riqualificazione per sound alternative, scriminanti and attenuanti with
   facts, the containment line when the charge is fully integrato -- ids and labels the code's; the call argues them,
   a thesis it adds is dropped and logged
2. best-of-3 on the thesis call, scored by the code, the three logged
3. one tolerant citation parser for intake, the charge, the named articles and the rubrica check
4. the summary, the conclusions and the next steps from the selection; three objections always
5. the lawyer's notes di inquadramento: the file, the store, the record, the writer and the thesis call
"""

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

from src.rag import ai_chat, article_notes, citations, intake, lawyer_method as M, posture as posture_mod  # noqa: E402
from src.rag import record_check  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

ENTRY = posture_mod.entry_for("penale", "imputato", "primo_grado")
IMPUTATO = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado", "entry": ENTRY}
ATTORE = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing"}
SENTS = [{"id": "S1", "text": "Sempronio sferra uno schiaffo a Caio, che cade e muore.", "source": "caso",
          "certainty": "accertato"}]


def _req(i, text, status, **extra):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status, **extra}


def _records(*statuses, alt=None, scriminante=None):
    texts = ["atti diretti a percuotere o ledere", "la morte come conseguenza degli atti", "il nesso causale"]
    recs = {"584 c.p.": {"rubrica": "Omicidio preterintenzionale", "side": "threat", "charged": 1,
                         "requisites": [_req(i, texts[i - 1], st, distinctive=(i == 1))
                                        for i, st in enumerate(statuses, 1)]}}
    if alt:
        recs["589 c.p."] = {"rubrica": "Omicidio colposo", "side": "weapon", "alternative": True,
                            "requisites": [_req(9, "condotta colposa", alt)]}
    if scriminante:
        recs["52 c.p."] = {"rubrica": "Difesa legittima", "side": "weapon",
                           "requisites": [_req(20, "necessità di difendersi", scriminante)]}
    return recs


def _ctx(recs, posture=IMPUTATO, circ_link=None, rows=()):
    return M.prepare({"records": recs, "sentences": SENTS}, posture, list(rows), "Nel 2025 Sempronio colpisce Caio, in "
                     "stato d'ira per un fatto ingiusto.", "Sempronio", "", list(recs), [], [], [], [],
                     link_call=circ_link or (lambda *a: {}), case_id_="t72")


# --- 1. the candidates --------------------------------------------------------------------------------------------------

def test_the_code_enumerates_the_candidates_with_their_ids_and_labels():
    recs = _records("provato", "da provare", "non risulta", alt="provato", scriminante="provato")

    def link(topic, reqs, sents):          # the provocazione's state of anger proven
        return {r["id"]: {"sostiene": ["S1"], "contraddice": [], "manca": ""} for _, r in reqs
                if r["text"].startswith("il reato è commesso in stato d'ira")}
    ctx = _ctx(recs, circ_link=link)
    cands = M.thesis_candidates(ctx)
    by_kind = {c["kind"]: c for c in cands}
    assert [c["id"] for c in cands] == [f"T{i}" for i in range(1, len(cands) + 1)]
    assert by_kind["contest_excluded"]["contesta"] == ["R3"] and by_kind["contest_excluded"]["label"] == M.SUSTAINABLE
    assert by_kind["contest_open"]["contesta"] == ["R2"] and by_kind["contest_open"]["label"] == M.SUSTAINABLE
    assert by_kind["contest_integrated"]["contesta"] == ["R1"] and by_kind["contest_integrated"]["label"] == M.MINORITY
    requalify = by_kind["requalify"]
    assert requalify["tesi"] == "Riqualificazione nell'art. 589 c.p. (Omicidio colposo)"
    assert requalify["contesta"] == ["R1"] and requalify["poggia_su"] == ["R9"] and requalify["label"] == M.MINORITY
    assert by_kind["scriminante"]["poggia_su"] == ["R20"]
    assert by_kind["attenuante"]["tesi"].startswith("Attenuante dell'art. 62 n. 2 c.p.")
    # v79.7 (item 2): the imputato's order -- contests by element, the riqualificazione, the scriminante and the
    # attenuante, the containment line last (offered on every charge, resting on no element when not all integrato)
    assert [c["kind"] for c in cands] == ["contest_integrated", "contest_open", "contest_excluded", "requalify",
                                          "scriminante", "attenuante", "contenimento"]
    assert by_kind["contenimento"]["poggia_su"] == []


def test_the_containment_line_when_the_charge_is_fully_integrated_and_always_kept():
    recs = _records("provato", "provato", "provato")
    cands = M.thesis_candidates(_ctx(recs))
    kinds = [c["kind"] for c in cands]
    assert kinds[-1] == "contenimento" and cands[-1]["fallback"] and cands[-1]["poggia_su"] == ["R1", "R2", "R3"]
    assert kinds.count("contest_integrated") == 3
    many = _records("provato", "provato", "provato")
    many["584 c.p."]["requisites"] += [_req(i, f"elemento {i}", "provato") for i in range(4, 12)]
    capped = M.thesis_candidates(_ctx(many))
    assert len(capped) == M.CANDIDATES_MAX and capped[-1]["kind"] == "contenimento"


def test_a_claimant_gets_his_articles_with_proven_elements():
    recs = {"1051 c.c.": {"rubrica": "Passaggio coattivo", "side": "weapon",
                          "requisites": [_req(1, "fondo intercluso", "provato"), _req(2, "accesso", "da provare")]},
            "1032 c.c.": {"rubrica": "Servitù coattiva", "side": "weapon",
                          "requisites": [_req(3, "titolo", "non risulta")]}}
    cands = M.thesis_candidates(_ctx(recs, ATTORE))
    assert [c["tesi"] for c in cands] == ["Fondare la domanda sull'art. 1051 c.c. (Passaggio coattivo)",
                                          "Chiedere l'azione confessoria (art. 1079 c.c.)"]   # v75: the matter's remedy


def test_the_call_argues_the_candidates_and_its_own_theses_are_dropped():
    recs = _records("provato", "da provare")
    ctx = _ctx(recs)
    cands = M.thesis_candidates(ctx)
    log = {}
    sols = M.apply_valuations({"valutazioni": {"T1": {"pro": "p", "contro": "c", "realismo": "alto", "argomento": "a"},
                                               "T2": {"realismo": "alto"}, "T7": {"realismo": "alto"}},
                               "soluzioni": [{"tesi": "mia"}]}, cands, log)
    assert [s["candidate"] for s in sols] == [c["id"] for c in cands] and log["dropped"] == ["T7", "soluzioni"]
    assert sols[0]["argomento"] == "a" and sols[0]["realismo"] == "basso"     # T1: the minority contest of R1
    assert sols[1]["kind"] == "contest_open" and sols[1]["realismo"] == "alto"
    minority = next(s for s in sols if s["minoritaria"])
    assert minority["realismo"] == M.BASSO                   # a minority candidate: basso whatever the model says


# --- 2. best-of-3 -----------------------------------------------------------------------------------------------------

def test_best_of_three_is_scored_by_the_code_and_logged(tmp_path, monkeypatch):
    monkeypatch.setenv(M.THESIS_LOG_ENV, str(tmp_path))
    recs = _records("provato", "da provare")
    ctx = _ctx(recs)
    cands = M.thesis_candidates(ctx)
    ids = [c["id"] for c in cands]
    minority = next(c["id"] for c in cands if c["label"] == M.MINORITY)
    objections = [{"obiezione": f"o{i}", "risposta": "r"} for i in range(3)]
    full = {k: {"realismo": "basso" if k == minority else "medio", "pro": "p"} for k in ids}
    replies = [{"valutazioni": {**full, minority: {"realismo": "alto"}}, "obiezioni": objections},   # minority alto
               {"valutazioni": {**full, "T9": {"realismo": "alto"}}, "obiezioni": objections},       # adds T9
               {"valutazioni": full, "obiezioni": objections}]                                       # clean
    plan = M.method_plan(ctx, call=lambda messages: replies)
    log = json.loads((tmp_path / "thesis_t72.json").read_text(encoding="utf-8"))
    assert log["best"] == 2 and len(log["samples"]) == 3 and log["n"] == M.SAMPLES
    assert log["samples"][0]["score"] < log["samples"][2]["score"] and "added ['T9']" in log["samples"][1]["why"]
    assert len(plan["obiezioni"]) == 3
    score, why = M.score_sample("not json", cands, {})
    assert score == float("-inf")


def test_n_replies_in_one_request_or_one_at_a_time(monkeypatch):
    class Gen:
        def __init__(self, text):
            self.message = type("M", (), {"content": text, "response_metadata": {"finish_reason": "stop"},
                                           "usage_metadata": {}})()

    class Model:
        def __init__(self, **kw):
            Model.kw = kw

        def generate(self, batches):
            return type("R", (), {"generations": [[Gen("a"), Gen("b")]]})()
    monkeypatch.setattr(ai_chat, "ChatOpenAI", Model)
    monkeypatch.setattr(ai_chat, "_call_chat", lambda messages, **kw: "c")
    assert ai_chat.call_chat_n([], 3, max_tokens=100) == ["a", "b", "c"]        # the server gave two: one more asked
    assert Model.kw["n"] == 3 and Model.kw["temperature"] == 0.7


# --- 3. one citation parser -----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, refs", [
    ("art 584 cp", [("584", "c.p.")]), ("art.584 c.p.", [("584", "c.p.")]),
    ("articolo 584 del codice penale", [("584", "c.p.")]), ("risponde ex 584 c.p.", [("584", "c.p.")]),
    ("l'attenuante dell'art. 62 n. 6 c.p.", [("62", "c.p.")]), ("comma 2 c.p.", []),
    ("artt. 582 e 583 c.p.", [("582", "c.p."), ("583", "c.p.")]),
])
def test_the_shared_parser_reads_every_form(text, refs):
    assert citations.article_refs(text) == refs


def test_every_consumer_uses_the_shared_parser():
    named = intake.named_articles("Il PM contesta a Sempronio il 584 c.p.")
    assert [a["article"] for a in named] == ["584 c.p."]
    assert posture_mod.sentence_articles("Il PM contesta l'art 584 cp e 589 c.p.") == ["584 c.p.", "589 c.p."]
    assert intake.prior_decisions("Il pubblico ministero contesta il 584 c.p.")[0]["articles"] == ["584 c.p."]
    state = {"intake": {}, "text": "Si applica l'articolo 2043 del codice civile e il 1218 c.c."}
    assert flow.user_named(state) == [("2043", "c.c."), ("1218", "c.c.")]
    assert record_check._BEFORE_RE.search("la legittima difesa (art 52 cp)").group("num") == "52"
    assert record_check._LOOSE_BEFORE_RE.search("lesioni ai sensi dell'articolo 582").group("num") == "582"


# --- 4. what follows from the selection -----------------------------------------------------------------------------------

def test_the_conclusions_and_steps_come_from_the_selection_not_the_plan():
    recs = _records("provato", "da provare")
    recs["584 c.p."]["requisites"][1]["missing"] = "la causa della morte"
    ctx = _ctx(recs)
    cands = M.thesis_candidates(ctx)
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "medio", "argomento": f"arg {c['id']}"}
                                           for c in cands},
                           "conclusioni": "Assoluzione certa.", "passi": [{"passo": "Passo del modello"}],
                           "sintesi": "Sintesi del modello."}, cands)
    answer = M.compose([("Norme applicate ai fatti", "regola")], ctx, plan)
    by = dict(M.answer_sections(answer))
    con = by["Conclusioni e valutazione del rischio"]
    assert "Assoluzione certa" not in answer and "Passo del modello" not in answer and "Sintesi del modello" not in answer
    # v79 (E2): outside the tables an element is its words, never its code
    el = "«la morte come conseguenza degli atti»"
    assert f"**Linea principale**: L'elemento {el} dell'art. 584 c.p. non è provato — arg T2." in con
    assert f"dipende da: la prova degli elementi {el}" in con and f"che l'accusa provi gli elementi {el}" in con
    # v73: the main line contests R2 -- its step is the accusa's proof of it, from the line's own element
    assert f"- Verificare quale prova l'accusa ha di: la causa della morte — scopo: decide l'elemento {el} dell'art. " \
           "584 c.p., che la linea principale contesta e il caso non prova." in con
    assert "R2" not in con.split("|")[0]
    assert by["Sintesi della soluzione"].startswith("Linea principale: L'elemento")


def test_three_objections_always_even_with_no_model_and_no_record_fallback():
    recs = {"1051 c.c.": {"rubrica": "Passaggio coattivo", "side": "weapon",
                          "requisites": [_req(1, "fondo intercluso", "provato")]}}
    ctx = _ctx(recs, ATTORE)
    plan = M.clean_method({}, M.thesis_candidates(ctx))
    by = dict(M.answer_sections(M.compose([], ctx, plan)))
    assert by["Obiezioni e risposte"].count("**Obiezione") == 3


def test_the_parenthesis_marker_never_spans_lines():
    sections = [("A", "**Obiezione 1.** x (vedi\n**Risposta.** y (da verificare: z).")]
    body, points = M.collect_points(sections)
    assert "**Risposta.**" in body[0][1] and points == ["Y (z)."]          # v74: the sentence, not its label


# --- 5. the notes ---------------------------------------------------------------------------------------------------------

def test_the_notes_file_is_parsed_per_article_and_numero():
    notes = article_notes.parse((ROOT / "docs" / "lawyer" / "note_inquadramento.md").read_text(encoding="utf-8"))
    assert "Nella preterintenzione il dolo riguarda gli atti base (581/582)" in notes["584 c.p."]["nota"]
    assert notes["584 c.p."]["nota"].startswith("Errori da evitare:")
    assert set(notes["62 c.p."]["numeri"]) == {"2", "6"} and "41 c.p." in notes and "589 c.p." in notes
    store = article_notes.store_items(notes)
    assert {n for n, _ in store["c.p."]} >= {"584", "41", "43", "62", "589", "590"}


def test_a_store_note_wins_over_the_file_and_reaches_the_record_and_the_writers():
    rows = [{"_article": ("584", "c.p."), "_note": {"nota": "Nota dello store.", "numeri": {}}}]
    assert article_notes.for_label("584 c.p.", rows) == ("Nota dello store.", "store")
    assert article_notes.for_label("41 c.p.")[1] == "file"
    assert article_notes.for_label("999 c.p.") == ("", "")
    recs = _records("provato", "da provare")
    ctx = _ctx(recs, rows=rows)
    assert recs["584 c.p."]["note_di_inquadramento"] == "Nota dello store."
    assert f"{M.NOTES_TITLE}:\n- art. 584 c.p.: Nota dello store." in M.method_material(ctx)
    assert "62-bis c.p." not in ctx["notes"]                # no note for it in the file
    state = {"mode": "single_shot", "records": recs}
    assert flow.method_notes(state) == ""                    # the method off: no notes block
    os.environ["LAWYER_METHOD"] = "true"
    try:
        assert flow.method_notes(state).startswith(M.NOTES_TITLE)
    finally:
        del os.environ["LAWYER_METHOD"]


def test_the_replay_checks_a_reply_against_the_logged_candidates(tmp_path, monkeypatch):
    import thesis_replay
    monkeypatch.setenv(M.THESIS_LOG_ENV, str(tmp_path))
    ctx = _ctx(_records("provato", "provato", "provato"))
    M.method_plan(ctx, call=lambda messages: {"valutazioni": {}})
    log = json.loads((tmp_path / "thesis_t72.json").read_text(encoding="utf-8"))
    checked = thesis_replay.check({"valutazioni": {"T9": {"realismo": "alto"}}}, log)
    assert checked["main"].startswith("Ammettere la condotta") and checked["dropped"] == ["T9"]
