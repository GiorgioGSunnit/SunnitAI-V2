"""v79.19.8 (branch v79.19) -- the objections argue §1's line; the relabel narrowed; the questions answered, not
repeated; a fit-confirmed hypothesis is a candidate, not the charge; one retry on a malformed step; the general-part
articles graded for an accused. Synthetic fixtures only (invented names and facts, statutory article texts).

1. row 95: the verified analysis reaches the objections call -- through the flow (the end-to-end harness of
   tests/_fixture_79_15_0.py, every model faked): lawyer_method_objections thesis_source «step», its thesis the step's
   first line. No ctx["verified_analysis"] set by hand.
2. row 96: the relabel fires on a penal frame whose first line is choose_charge's article only; a civil defendant and a
   penal page whose first line is not the charge keep the usual labels; the fixed sentence appears nowhere; the Q&A's
   strategy sentence is the relabelled line.
3. row 98: a three-point request with only the first point interrogative -> three questions; an inserted clause is no
   point; an item that repeats §1's main line is not printed («v. §1» with nothing else); a procedure item is a whole
   sentence, never cut inside «d.lgs.».
4. row 100: a hypothesis the fit confirms is a candidate line -- the frame's charge stays unset, choose_charge's rule
   is not case_frame, logged frame_hypothesis_confirmed {as: candidate}.
5. row 101: a malformed first reply is asked again once inside the remaining budget (analysis_step_retry); none left:
   no retry.
6. row 97: 55 c.p. in the records of an accused is graded (interactive_flow.general_part_grading), printed in the
   general-part section, never a line candidate; a claimant is untouched; a claimant in contract gets the remedy
   records (1224 c.c. too); a non-contractual claimant's empty section says why.
"""

import json
import os
import re
import sys
import types
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import _fixture_79_15_0 as H  # noqa: E402
import test_v79_15_0_analysis as T  # noqa: E402
import test_v79_15_0_fixture_penal as P  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import intake  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import lawyer_method as M  # noqa: E402
from src.rag import posture  # noqa: E402
from src.rag import verbose_logger  # noqa: E402
from test_v79_14_1_b import _ctx, _penal_records, _plan  # noqa: E402
from test_v79_15_0_analysis import _Model, _el, _line, _reply, _sealed  # noqa: E402,F401
from test_v79_19_5 import _defence_reply, _penal_input  # noqa: E402

ROOT = HERE.parent
FIXED_SENTENCE = "Nessun elemento della contestazione"


def _events(monkeypatch):
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda e, d=None, ms=None: events.append((e, d)))
    return events


def _case(**over):
    keys = ("TEXT", "ARTICLES", "FRAME_REPLY", "fit_reply", "link", "analysis_reply", "TITLE", "ANCHOR", "RUBRICA",
            "DOCUMENTS", "CLIENT", "writer_line", "CASE_ID")
    return types.SimpleNamespace(**{**{k: getattr(P, k) for k in keys}, **over})


# --- 1. row 95: the verified analysis reaches the objections call, through the flow ---------------------------------------

def test_through_the_flow_the_objections_argue_the_steps_first_line(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, _case(CASE_ID="v79198_flow"))
    state, by = run["state"], run["by"]
    assert [x for x in run["failed"] if x.split(":")[0] in ("analysis_step", "lawyer_method_plan",
                                                             "lawyer_method_objections", "lawyer_method_compose")] == []
    step = by["analysis_step"][0][0]
    assert step["status"] == "ok"
    first = state["analysis_step"]["analysis"]["linea_principale"]
    assert first["articolo"] == "646 c.p." and first["stato"] == A.VERIFIED
    # the objections call saw the verified step: thesis_source «step», the thesis the step's first line
    logged = by["lawyer_method_objections"]
    assert logged, "the objections call was not logged"
    entry = logged[0][0]
    assert entry["thesis_source"] == "step"
    assert entry["main"].startswith("art. 646 c.p. — "), entry["main"]
    # and nothing set it by hand: the context the flow built carries the view the step returned
    assert "legacy" not in [d["thesis_source"] for d, _ in logged]


def test_method_inputs_sets_the_verified_analysis_before_the_objections(monkeypatch):
    """The objections are asked inside method_inputs: the context must hold the verified analysis by then."""
    seen = {}

    def objections(ctx, mplan):
        seen["verified"] = ctx.get("verified_analysis")
        return mplan
    monkeypatch.setattr(flow, "method_objections", objections)
    records = _penal_records("provato", "da provare", "da provare")
    state = {"records": records, "sentences": [], "text": "Mevio spinge Sempronio, che cade.", "analysis": {},
             "block_counters": {}}
    posture_ = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "primo_grado",
                "entry": posture.entry_for("penale", "imputato", "primo_grado")}
    view = {"status": A.OK, "analysis": {}, "marker": "the step's view"}
    ctx, _ = flow.method_inputs(state, posture_, {"parte_assistita": "Mevio"}, [], [], [],
                                "Mevio spinge Sempronio, che cade.", verified=view)
    assert seen["verified"] is view and ctx["verified_analysis"] is view


# --- 2. row 96: the relabel narrowed, the fixed sentence gone -------------------------------------------------------------

def _view(monkeypatch, inp, reply):
    events = _events(monkeypatch)
    res = A.analyse(inp, call=_Model(reply))
    return A.view(res, inp), events


def test_the_relabel_fires_on_a_penal_frame_whose_first_line_is_the_settled_charge(monkeypatch, _sealed):
    inp = _penal_input()
    inp["settled_charge"] = "640 c.p."                   # choose_charge's article (state["charge"])
    v, events = _view(monkeypatch, inp, _defence_reply("provato", "da_provare"))
    assert v["relabel"]["applied"] is True and v["charge"] == "640 c.p." and v["penal"] is True
    assert A.summary_text(v).startswith("Tesi dell'accusa: art. 640 c.p.")
    logged = next(d for e, d in events if e == "analysis_defence_relabel")
    assert logged["charge"] == "640 c.p." and logged["penal"] is True
    # the Q&A's strategy sentence is the relabelled line, never «La linea principale è: … risulta provato»
    strategy = A.answer_strategy(v)
    assert strategy[0].startswith("Tesi dell'accusa: art. 640 c.p. — l'elemento «con artifizi o raggiri» risulta provato")
    assert "La linea principale è" not in strategy[0]


def test_a_penal_page_whose_first_line_is_not_the_charge_keeps_the_usual_labels(monkeypatch, _sealed):
    inp = _penal_input()
    inp["settled_charge"] = "641 c.p."                   # the settled charge is another article (user_article)
    v, events = _view(monkeypatch, inp, _defence_reply("provato", "da_provare"))
    assert v["relabel"] == {"applied": False, "why": "prima linea non è l'addebito"}
    assert A.summary_text(v).startswith("Linea principale: art. 640 c.p.") and "Tesi dell'accusa" not in A.summary_text(v)
    assert not [e for e, _ in events if e == "analysis_defence_relabel"]
    assert A.answer_strategy(v)[0].startswith("La linea principale è: art. 640 c.p.")


def test_a_civil_defendant_never_gets_the_penal_labels(monkeypatch, _sealed):
    frame = json.loads(json.dumps(T.CIVIL_FRAME))
    frame["ruolo"] = {"valore": "convenuto", "intake": "convenuto", "perche": "«Assisto Lucia Ferri, convenuta»"}
    frame["parti"][0]["ruolo_nel_fatto"] = "promittente venditrice"
    inp = A.make_input(frame, T.CIVIL_NORMS, T._sents(T.CIVIL_SENTENCES), frame.get("quesiti") or [],
                       client="Lucia Ferri", client_roles=["convenuto"])
    inp["settled_charge"] = "1385 c.c."
    assert inp["side"] == A.SIDE_DEFENCE
    reply = _reply(_line("1385 c.c.", T.E1385[0], "x", ["S2"], ["A1385cc.1"]),
                   elementi=[_el("1385 c.c.", T.E1385[0], "provato", ["S2"])])
    v, _ = _view(monkeypatch, inp, reply)
    assert v["analysis"]["linea_principale"]["esito_in_gioco"] == "provato" and v["penal"] is False
    assert v["relabel"] == {"applied": False, "why": "non penale"}
    summary = A.summary_text(v)
    assert summary.startswith("Linea principale: art. 1385 c.c.") and "Tesi dell'accusa" not in summary
    assert "qualificazione e sulla pena" not in summary


def test_every_element_proven_prints_the_accusa_then_the_next_line_nothing_in_between(monkeypatch, _sealed):
    inp = _penal_input()
    inp["settled_charge"] = "640 c.p."
    v, _ = _view(monkeypatch, inp, _defence_reply("provato", "provato"))
    lines = A.summary_text(v).split("\n\n")
    assert [x.split(":")[0] for x in lines] == ["Tesi dell'accusa", "Linea subordinata"]
    assert "Linea difensiva" not in A.summary_text(v) and FIXED_SENTENCE not in A.solutions_text(v)
    assert FIXED_SENTENCE not in A.conclusions_text(v) and all(FIXED_SENTENCE not in s for s in A.answer_strategy(v))
    alone, _ = _view(monkeypatch, inp, _defence_reply("provato"))
    assert A.summary_text(alone).count("\n\n") == 0 and A.summary_text(alone).startswith("Tesi dell'accusa")


def test_the_fixed_sentence_is_nowhere():
    assert not hasattr(A, "FIXED_LINE")
    data = json.loads((ROOT / "data" / "analysis_step.json").read_text(encoding="utf-8"))
    assert "nessun_elemento" not in data["rendering"]["difesa"]
    assert FIXED_SENTENCE not in json.dumps(data, ensure_ascii=False)
    assert FIXED_SENTENCE not in (ROOT / "docs" / "copy_review.md").read_text(encoding="utf-8")
    for name in ("analysis_step.py", "lawyer_method.py", "interactive_flow.py"):
        assert FIXED_SENTENCE not in (ROOT / "src" / "rag" / name).read_text(encoding="utf-8")


# --- 3. row 98: the questions answered, not repeated ---------------------------------------------------------------------

def test_a_request_whose_first_point_alone_is_interrogative_yields_one_question_per_point():
    text = "Assisto Tizio. Il cliente chiede se il recesso sia valido, i danni risarcibili e i tempi dell'azione."
    assert intake.questions(text) == ["se il recesso sia valido", "i danni risarcibili", "i tempi dell'azione"]
    assert intake.split_points("Vorrei sapere quali rimedi ha, le conseguenze e in che tempi deve agire") == \
        ["quali rimedi ha", "le conseguenze", "in che tempi deve agire"]
    # the «e» inside a point and the one-word remainder stay with their point
    assert intake.split_points("Chiedo se il contratto sia valido e efficace, quali danni può chiedere") == \
        ["se il contratto sia valido e efficace", "quali danni può chiedere"]
    # an inserted clause is no point: one question
    one = "Vorrei sapere se il contratto, stipulato nel 2020, sia valido."
    assert intake.split_points(one) == [one]
    # the 79.19.5 form still splits, and a first point that is no interrogative does not
    assert intake.split_points("Vorrei sapere a chi rivolgermi, in che tempi e con quali costi.") == \
        ["a chi rivolgermi", "in che tempi", "con quali costi"]
    assert intake.split_points("Vorrei sapere la procedura, i tempi e i costi.") == \
        ["Vorrei sapere la procedura, i tempi e i costi."]


PROCEDURE = ("Il termine per l'appello è di quindici giorni dal deposito della sentenza (d.lgs. 31 ottobre 2024, n. 164). "
             "L'atto si deposita nella cancelleria del giudice che ha emesso la sentenza (art. 582 c.p.p.).")


def test_the_sentence_split_never_breaks_on_an_act_abbreviation():
    got = M._first_sentences(PROCEDURE, 60)
    assert len(got) == 2 and got[0].endswith("(d.lgs. 31 ottobre 2024, n. 164).")
    assert all(not re.search(r"\b(?:d\.lgs|d\.P\.R|d\.l|L|art|artt|co|n)\.$", x) for x in got)
    pieces = M._citation_rejoin(["Ai sensi del d.lgs.", "28/2010 la mediazione è obbligatoria.", "Il d.P.R.",
                                 "115/2002 regola le spese.", "La L.", "604/1966 e il d.l.", "132/2014 pure.",
                                 "Fine."])
    assert pieces == ["Ai sensi del d.lgs. 28/2010 la mediazione è obbligatoria.", "Il d.P.R. 115/2002 regola le spese.",
                      "La L. 604/1966 e il d.l. 132/2014 pure.", "Fine."]
    for abbr in ("d.lgs", "d.P.R", "d.l", "art", "artt", "co", "n", "L"):
        assert abbr in M._ABBREV
    assert not M._whole_sentence("31 ottobre 2024, n. 164), dal 26.11.2024: il rito è quello ordinario.")
    assert not M._whole_sentence("Il licenziamento è impugnabile (d.lgs.")
    assert M._whole_sentence("Il termine è di quindici giorni (d.lgs. 31 ottobre 2024, n. 164).")


def test_an_item_that_repeats_section_1_is_not_printed_and_the_procedure_item_is_a_whole_sentence(monkeypatch):
    events = _events(monkeypatch)
    ctx = _ctx(_penal_records("provato", "da provare", "da provare"), "penale")
    sols, main, _ = M.selection(ctx, _plan(ctx))
    repeat = f"La linea principale è: {sols[main]['tesi'].rstrip('.')}."
    ctx["asked"] = ["Quale strategia difensiva?", "Entro quando si può proporre l'appello?", "Con quale procedura?"]
    block, strategy = M.answers_block(ctx, [("Quale strategia difensiva?", repeat)], PROCEDURE, sols, main,
                                      ctx.get("assessment") or {}, "Mevio", lambda t: t, log := {})
    items = dict(M.answers_of(block))
    # the writer's item equal to §1's main line: not printed -- «v. §1» in its place
    assert items["Quale strategia difensiva?"] == M.SEE_SUMMARY_ANSWER and log["answers"][0] == "v. §1"
    assert repeat not in block
    # the procedure item: the whole §8 sentence, never cut on «d.lgs.», never padded with §1's line
    assert items["Entro quando si può proporre l'appello?"] == \
        "Il termine per l'appello è di quindici giorni dal deposito della sentenza (d.lgs. 31 ottobre 2024, n. 164)."
    assert "(d.lgs." not in items["Entro quando si può proporre l'appello?"].split("n. 164")[0][-8:]
    assert log["answers"][1] == "procedura"
    # a procedural question §8 does not answer: no source, not §1's line
    assert items["Con quale procedura?"] == M.NO_SOURCE_ANSWER and log["answers"][2] == "nessuna fonte"
    assert "La linea principale è" not in block
    logged = next(d for e, d in events if e == "lawyer_method_answers_repeat_of_summary")
    assert logged["items"][0]["question"] == "Quale strategia difensiva?" and logged["items"][0]["left"] == 0


def test_the_selection_item_keeps_the_forecast_without_the_repeat():
    ctx = _ctx(_penal_records("provato", "da provare", "da provare"), "penale")
    sols, main, _ = M.selection(ctx, _plan(ctx))
    ctx["asked"] = ["Quali rischi corre Mevio?"]
    block, _ = M.answers_block(ctx, [], "", sols, main, ctx.get("assessment") or {}, "Mevio", lambda t: t, {})
    answer = dict(M.answers_of(block))["Quali rischi corre Mevio?"]
    assert answer.startswith("L'esito probabile è") and "La linea principale è" not in answer


# --- 4. row 100: a fit-confirmed hypothesis is a candidate line, not the charge ------------------------------------------

LITE_FRAME = {
    "parti": [{"nome": "Remo Valle", "ruolo_nel_fatto": "", "lato": "cliente"},
              {"nome": "Ugo Brenta", "ruolo_nel_fatto": "", "lato": "controparte"}],
    "cliente": {"nome": "Remo Valle", "perche": "«Assisto Remo Valle»"},
    "ruolo": {"valore": "indagato", "intake": "indagato", "perche": "«indagato»"},
    "rito": {"valore": "penale", "intake": "penale", "perche": "«la Procura procede»"},
    "stadio": None, "istituto": None, "quesiti": [],
    "addebito_o_domanda": {"articolo": None, "numero": "", "codice": "", "descrizione": "percosse seguite da morte",
                           "frase": "«Remo Valle colpisce Ugo Brenta, che cade e muore»", "ipotesi": "584 c.p.",
                           "da_fatti": True}}


def test_a_hypothesis_the_fit_confirms_stays_a_candidate_and_choose_charge_ignores_it(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    events = _events(monkeypatch)
    state = {"case_frame": {"frame": json.loads(json.dumps(LITE_FRAME)), "corrections": [], "counters": {}}}
    cf.settle(state)
    assert cf.alternative_charges(state) == ["584 c.p."] and cf.hypotheses(state) == [{"label": "584 c.p.",
                                                                                         "stato": "aperta"}]
    made = cf.apply_fit(state, [{"article": "584 c.p.", "fit": "si", "verified": True}])
    assert made == [] and cf.bound(state, "addebito_o_domanda") is None and cf.charges(state) == []
    assert cf.alternative_charges(state) == ["584 c.p."]                   # a candidate line still
    assert cf.hypotheses(state) == [{"label": "584 c.p.", "stato": "confermata"}]
    c = state["frame"]["counters"]
    assert c["frame_hypothesis_confirmed"] == 1 and c.get("frame_hypothesis_dropped", 0) == 0
    assert c.get("frame_corrections_applied", 0) == 0
    logged = next(d for e, d in events if e == "frame_hypothesis_confirmed")
    assert logged["article"] == "584 c.p." and logged["as"] == "candidate" and logged["fit"] == "si"
    assert not [d for e, d in events if e == "frame_correction"]
    # choose_charge: the frame gives no charge -- the rule stays what the other sources give (none here)
    records = {"584 c.p.": {"article": "584 c.p.", "rubrica": "Omicidio preterintenzionale", "requisites": []}}
    out = posture.choose_charge(state, {"rito": "penale", "family": "penale", "role": "imputato"}, records,
                                "Remo Valle colpisce Ugo Brenta, che cade e muore.")
    assert out["rule"] != posture.FRAME_CHARGE_RULE and out["article"] != "584 c.p."
    assert state["charge"]["rule"] == posture.NO_CHARGE_RULE
    # the step's input: no named charge, no charge hypothesis forced to lead
    inp = A.make_input(cf._frame(cf.held(state)), [], [], [])
    assert inp["charge"] == "" and inp["charge_hypothesis"] == ""


def test_a_hypothesis_the_text_confirms_is_unchanged(monkeypatch):
    """A charge the text names (articolo set, no «da_fatti») is the charge as before: no hypothesis at all."""
    monkeypatch.setenv("CASE_FRAME", "true")
    frame = json.loads(json.dumps(LITE_FRAME))
    frame["addebito_o_domanda"] = {"articolo": "584 c.p.", "numero": "584", "codice": "c.p.",
                                   "descrizione": "omicidio preterintenzionale",
                                   "frase": "«indagato per il reato di cui all'art. 584 c.p.»"}
    state = {"case_frame": {"frame": frame, "corrections": [], "counters": {}}}
    cf.settle(state)
    assert cf.bound(state, "addebito_o_domanda") == "584 c.p." and cf.hypotheses(state) == []
    assert cf.apply_fit(state, [{"article": "584 c.p.", "fit": "si", "verified": True}]) == []
    assert cf.bound(state, "addebito_o_domanda") == "584 c.p."


# --- 5. row 101: one retry on a malformed step ---------------------------------------------------------------------------

def test_a_malformed_reply_is_asked_again_once_then_the_lines_stand(monkeypatch, _sealed):
    events = _events(monkeypatch)
    model = _Model("niente di utile", T.GOOD_PENAL)
    res = A.analyse(_penal_input(), call=model)
    assert res["status"] == A.OK and res["analysis"]["linea_principale"]["articolo"] == "640 c.p."
    assert len(model.seen) == 2 and res["counters"]["analysis_calls"] == 2 and res["counters"]["analysis_retried"] == 1
    assert [c["status"] for c in res["calls"]] == ["malformed", "ok"]
    assert "analysis_retried" in A.COUNTERS
    retries = [d for e, d in events if e == "analysis_step_retry"]
    assert retries == [{"attempt": 2, "status": "ok", "first_status": "malformed", "budget_s": retries[0]["budget_s"],
                        "error": ""}] and retries[0]["budget_s"] > 0
    # the same input: both calls carry the same messages
    assert [m.content for m in model.seen[0]["messages"]] == [m.content for m in model.seen[1]["messages"]]
    assert res["counters"][A.SKIPPED_REASON] != A.SKIP_STEP_FAILED


def test_malformed_twice_fails_as_before_and_no_budget_means_no_retry(monkeypatch, _sealed):
    model = _Model("niente", "[1, 2]")
    res = A.analyse(_penal_input(), call=model)
    assert res["status"] == A.MALFORMED and res["analysis"] is None and len(model.seen) == 2
    assert res["counters"]["analysis_failed"] == 1 and res["counters"][A.SKIPPED_REASON] == A.SKIP_STEP_FAILED
    assert res["counters"]["analysis_retried"] == 1
    # no budget left after the first call: no retry, logged skipped
    events = _events(monkeypatch)
    calls = []

    def call_once(messages, call, budget, attempt=0):
        calls.append(attempt)
        return {"status": A.MALFORMED, "reply": "niente", "data": None, "seconds": 0.0, "error": "no JSON"}
    monkeypatch.setattr(A, "call_once", call_once)
    monkeypatch.setattr(A, "deadline_for", lambda inp: 0.0)
    res = A.analyse(_penal_input(), call=lambda m, k: "niente")
    assert calls == [1] and res["status"] == A.MALFORMED and res["counters"]["analysis_retried"] == 0
    assert [d for e, d in events if e == "analysis_step_retry"] == [
        {"attempt": 2, "status": "skipped", "first_status": "malformed", "budget_s": 0.0, "error": "nessun tempo residuo"}]


def test_a_retry_then_a_reask_number_the_calls_in_order(_sealed):
    """A retry followed by the verification's re-ask: three calls, the re-ask the third."""
    bad = _reply(_line("640 c.p.", T.E640[0], "x", ["S99"], ["A640cp.1"]),            # S99: no such fact -> re-ask
                 elementi=[_el("640 c.p.", T.E640[0], "da_provare")])
    model = _Model("niente", bad, T.GOOD_PENAL)
    res = A.analyse(_penal_input(), call=model)
    assert len(model.seen) == 3 and [c["attempt"] for c in res["calls"]] == [1, 2, 3]
    assert res["counters"]["analysis_retried"] == 1 and res["counters"]["analysis_reasked"] == 1


# --- 6. row 97: the general-part articles graded for an accused -------------------------------------------------------------

ART_55 = ("Quando, nel commettere alcuno dei fatti preveduti dagli articoli 51, 52, 53 e 54, si eccedono colposamente i "
          "limiti stabiliti dalla legge o dall'ordine dell'Autorità ovvero imposti dalla necessità, si applicano le "
          "disposizioni concernenti i delitti colposi, se il fatto è preveduto dalla legge come delitto colposo.")


def _rows_55():
    return [{"_article": ["55", "c.p."], "s": {"name": "55.0.0", "plain_text": ART_55, "titolo_articolo": "Eccesso colposo"},
             "d": {"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"}}]


def _records(grade_55=None, dropped_55=None):
    return {"640 c.p.": {"article": "640 c.p.", "rubrica": "Truffa", "grade": 3, "dropped": None, "kept": ["tema"],
                         "requisites": [{"id": "R1", "kind": "requisito", "text": T.E640[0], "status": "da provare"}]},
            "55 c.p.": {"article": "55 c.p.", "rubrica": "Eccesso colposo", "grade": grade_55, "dropped": dropped_55,
                        "kept": [], "requisites": [{"id": "R7", "kind": "requisito",
                                                    "text": "l'eccesso colposo dai limiti della difesa",
                                                    "status": "da provare"}]},
            "1218 c.c.": {"article": "1218 c.c.", "rubrica": "Responsabilità del debitore", "grade": None,
                          "dropped": None, "kept": [], "requisites": []}}


def _graded_3(facts, request, rows, analysis):
    for r in rows:
        r["_grade"] = 3
    return list(rows)


def test_55_cp_in_the_records_of_an_accused_is_graded_a_claimant_untouched(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    monkeypatch.setattr(dg, "_filter_applicable_articles", _graded_3)
    events = _events(monkeypatch)
    state = {"case_frame": {"frame": json.loads(json.dumps(T.PENAL_FRAME))}, "records": _records(dropped_55="prefiltro"),
             "rows": [], "analysis": {}, "text": "Paolo Sarti, indagato per truffa, …", "charge": {"article": "640 c.p.",
                                                                                               "rule": "user_article"}}
    got = flow.general_part_grading(state, {"rito": "penale", "family": "penale", "role": "imputato"}, _rows_55())
    assert got == ["55 c.p."]
    assert state["records"]["55 c.p."]["grade"] == 3 and state["records"]["55 c.p."]["graded_for_defence"] is True
    assert state["records"]["55 c.p."]["dropped"] == "prefiltro"              # grading only: the gate's drop stands
    assert state["records"]["1218 c.c."]["grade"] is None and state["records"]["640 c.p."]["grade"] == 3
    logged = next(d for e, d in events if e == "interactive_general_part_gate")
    assert logged["graded_for_defence"]["articles"] == ["55 c.p."] and logged["graded_for_defence"]["grades"] == {
        "55 c.p.": 3}
    assert state["block_counters"]["general_part_graded_for_defence"] == 1
    # a claimant: untouched, nothing logged
    events.clear()
    civil = {"case_frame": {"frame": json.loads(json.dumps(T.CIVIL_FRAME))}, "records": _records(), "rows": [],
             "analysis": {}, "text": "Lucia Ferri …"}
    assert flow.general_part_grading(civil, {"rito": "civile", "family": "civile", "role": "attore"}, _rows_55()) == []
    assert civil["records"]["55 c.p."]["grade"] is None
    assert not [e for e, _ in events if e == "interactive_general_part_gate"]
    # a record of the set already graded, or an article outside the set: nothing to grade
    done = {**state, "records": _records(grade_55=2)}
    assert flow.general_part_grading(done, {"rito": "penale", "family": "penale", "role": "imputato"}, _rows_55()) == []
    assert A.defence_graded_set() == {(A._in_num(x), "cp") for x in ("41", "52", "55", "56", "59", "60", "61", "62",
                                                                      "62-bis")}
    assert A.in_defence_set("62-bis c.p.") and A.in_defence_set("55 c.p.") and not A.in_defence_set("640 c.p.")


def test_the_graded_55_cp_is_printed_in_the_section_and_is_no_line_candidate(monkeypatch, _sealed):
    monkeypatch.setenv("CASE_FRAME", "true")
    state = {"case_frame": {"frame": json.loads(json.dumps(T.PENAL_FRAME))}, "records": _records(grade_55=3,
                                                                                                 dropped_55="prefiltro"),
             "rows": [], "sentences": T._sents(T.PENAL_SENTENCES, sistema=(9,)), "analysis": {}, "intake": {},
             "charge": {"article": "640 c.p.", "rule": "user_article"}}
    inp = A.inputs(state)
    # the dropped record is no line candidate (not among the norms) and is listed for the section, flagged dropped
    assert "55 c.p." not in [n["article"] for n in inp["norms"]]
    graded = {g["article"]: g for g in inp["graded"]}
    assert graded["55 c.p."]["grade"] == 3 and graded["55 c.p."]["dropped"] is True and "dropped" not in graded["640 c.p."]
    assert inp["settled_charge"] == "640 c.p."
    # the section prints it after the lines; the accounting skips it
    full = _penal_input()
    full["graded"], full["settled_charge"] = inp["graded"], "640 c.p."
    events = _events(monkeypatch)
    res = A.analyse(full, call=_Model(T.GOOD_PENAL))
    v = A.view(res, full)
    assert [r["article"] for r in v["general_part"]["records"]] == ["55 c.p."]
    assert v["general_part"]["title"] == "Difese di parte generale che i fatti sollevano"
    assert "- art. 55 c.p. — Eccesso colposo: «l'eccesso colposo dai limiti della difesa» (da provare)." in \
        A.solutions_text(v)
    assert next(d for e, d in events if e == "analysis_general_part_section")["articles"] == ["55 c.p."]
    assert not [r for r in res["rejected"] if r.get("kind") == A.RECORD_KIND and r["articolo"] == "55 c.p."]
    assert "55 c.p." not in [x["articolo"] for x in (v["analysis"]["linea_principale"], v["analysis"]["linea_subordinata"])
                             if x]
    # ungraded (the flow did not grade it): nothing printed, the reason said
    full["graded"] = [dict(g, grade=None) if g["article"] == "55 c.p." else g for g in inp["graded"]]
    v2 = A.view(A.analyse(full, call=_Model(T.GOOD_PENAL)), full)
    assert v2["general_part"]["records"] == [] and v2["general_part"]["why"] == "nessun record del set graduato 2 o più"


def test_a_claimant_in_contract_gets_the_remedy_records_and_a_non_contractual_one_says_why(monkeypatch, _sealed):
    inp = T._civil()
    assert inp["contractual"] is True and inp["side"] == A.SIDE_CLAIMANT
    inp["graded"] = [{"article": "1218 c.c.", "grade": 3, "rubrica": "Responsabilità del debitore", "elements": []},
                     {"article": "1223 c.c.", "grade": 2, "rubrica": "Risarcimento del danno", "elements": []},
                     {"article": "1224 c.c.", "grade": 2, "rubrica": "Danni nelle obbligazioni pecuniarie", "elements": []},
                     {"article": "1453 c.c.", "grade": 3, "rubrica": "Risolubilità del contratto", "elements": []},
                     {"article": "1385 c.c.", "grade": 3, "rubrica": "Caparra confirmatoria", "elements": []}]
    events = _events(monkeypatch)
    res = A.analyse(inp, call=_Model("niente", "niente"))          # the step malformed twice: the fallback view
    assert res["status"] == A.MALFORMED
    v = A.view(res, inp)
    assert v["analysis"]["side"] == A.SIDE_CLAIMANT and v["analysis"]["linea_principale"]["stato"] != A.VERIFIED
    assert [r["article"] for r in v["general_part"]["records"]] == ["1218 c.c.", "1223 c.c.", "1224 c.c.", "1453 c.c."]
    assert v["general_part"]["title"] == "Rimedi applicabili"
    assert next(d for e, d in events if e == "analysis_general_part_section")["count"] == 4
    # a claimant whose frame is not contractual: nothing printed, why logged
    frame = json.loads(json.dumps(T.CIVIL_FRAME))
    frame["addebito_o_domanda"] = {"articolo": "2043 c.c.", "descrizione": "danno da fatto illecito",
                                   "frase": "«ha subito un danno»"}
    frame.pop("istituto", None)
    other = A.make_input(frame, T.CIVIL_NORMS, T._sents(T.CIVIL_SENTENCES), [], client="Lucia Ferri",
                         client_roles=["attore"])
    assert other["contractual"] is False
    other["graded"] = inp["graded"]
    events.clear()
    v = A.view(A.analyse(other, call=_Model("niente", "niente")), other)
    assert v["general_part"]["records"] == [] and v["general_part"]["why"] == "quadro non contrattuale"
    skipped = next(d for e, d in events if e == "analysis_general_part_skipped")
    assert skipped["side"] == A.SIDE_CLAIMANT and skipped["contractual"] is False
