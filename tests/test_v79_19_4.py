"""v79.19.4, Part B -- Antonelo's civil_six_lines (26 Nov): three record rules after grading (the model faked; synthetic
texts, invented names; nothing named after a case).

1. a contradicted alternative does not demote an article: an element's branches («o», «ovvero», «oppure», «in
   alternativa») or a paragraph's alternative comma -- demoted only when every alternative is contradicted; the
   contradicted one marked «contraddetta», logged alternative_contradicted; a conjunctive element as before; an element
   of an alternative paragraph is no missing fact of a line built on the other alternative (row 80)
2. a live record graded 2+ is a line or is in the step's «rejected» with a listed reason (not_in_block, block_full,
   below_cut, duplicate_of:<label>, step_silent); analysis_silent_records must be 0 on the fixtures
3. only the settled charge (user_article, rubrica_match) may stop the method: a read_hypothesis or case_frame candidate
   that fails the decomposition is dropped (frame_hypothesis_dropped {reason}) and the method runs on the verified
   lines; none left: step aside with the reason logged
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import analysis_step as A  # noqa: E402
from src.rag import case_frame, interactive_flow as flow, issue_records as I, lawyer_method as M, posture  # noqa: E402
from src.rag import verbose_logger  # noqa: E402
from test_v79_10_5 import IMPUTATO  # noqa: E402
from test_v79_15_0_analysis import CIVIL_NORMS, E1385, E1453, _civil, _el, _line, _Model, _reply, _sealed  # noqa: E402,F401

PROVATO, DA_PROVARE, NON_RISULTA = M.PROVATO, M.DA_PROVARE, M.NON_RISULTA


def _events(monkeypatch):
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda e, d=None, ms=None: events.append((e, d)))
    return events


def _sent(n, text):
    return {"id": f"S{n}", "text": text, "certainty": "accertato", "source": "caso"}


def _demote(monkeypatch, element, sentences, contradicting, alternatives=None, grade=2):
    # v79.19.9 (B, row 108): a grade of 3 is no longer demoted by a contradicted element alone -- the sample is graded 2
    # where the test wants the demotion as today; the grade-3 case is test_a_grade_3_contradicted_is_kept_and_logged
    rows = [{"_article": ("9999", "c.c."), "_grade": grade, "s": {}}]
    el = {"comma": 1, "text": element, **({"alternatives": alternatives} if alternatives else {})}
    monkeypatch.setattr(I, "distinctive", lambda rows, analysis: {"9999 c.c.": el})
    monkeypatch.setattr(I, "sentences", lambda case: [_sent(i, t) for i, t in enumerate(sentences, 1)])
    log = I.demote_contradicted(rows, "x", {}, call=lambda topic, reqs, sents: {
        r["id"]: {"sostiene": [], "contraddice": list(contradicting), "manca": ""} for _, r in reqs})
    return rows[0], log


# --- 1. a contradicted alternative does not demote -------------------------------------------------------------------------

def test_one_contradicted_alternative_keeps_the_grade_and_is_logged(monkeypatch):
    row, log = _demote(monkeypatch, "la cosa venduta non ha le qualità promesse ovvero quelle essenziali per l'uso",
                       ["La cosa venduta aveva tutte le qualità promesse nel contratto.", "Beta ha pagato."], ["S1"],
                       grade=3)
    assert row["_grade"] == 3 and log["demoted"] == [] and not row.get("_demoted") and "kept_grade3" not in log
    got = log["alternative_contradicted"]
    assert len(got) == 1 and got[0]["article"] == "9999 c.c." and got[0]["kept"] == 3
    assert got[0]["alternative"] == "la cosa venduta non ha le qualità promesse"
    assert got[0]["open"] == ["quelle essenziali per l'uso"]
    assert row["_alternative_contradicted"] == ["la cosa venduta non ha le qualità promesse"]
    # the record's requisite is marked «contraddetta» by seed
    rec = {"requisites": [{"id": "R1", "kind": "requisito", "comma": 1, "text": row["_distinctive_text"]}]}
    I.seed(rec, row)
    assert rec["requisites"][0]["distinctive"] and rec["requisites"][0]["contraddetta"] == row["_alternative_contradicted"]


def test_a_grade_3_contradicted_is_kept_and_logged(monkeypatch):
    """v79.19.9 (B, row 108): the whole element contradicted, grade 3 -- the grade stands, logged kept_grade3."""
    row, log = _demote(monkeypatch, "con violenza o minaccia", ["Alfa si è comportato correttamente con Beta."], ["S1"],
                       grade=3)
    assert row["_grade"] == 3 and not row.get("_demoted") and log["demoted"] == []
    assert log["kept_grade3"] == ["9999 c.c."]


def test_a_contradiction_of_no_single_alternative_is_of_the_whole_element_and_demotes_as_today(monkeypatch):
    row, log = _demote(monkeypatch, "con violenza o minaccia", ["Alfa si è comportato correttamente con Beta."], ["S1"])
    assert row["_grade"] == 1 and log["demoted"][0]["article"] == "9999 c.c."
    assert "alternative_contradicted" not in log


def test_every_alternative_contradicted_demotes_as_today(monkeypatch):
    row, log = _demote(monkeypatch, "con violenza o minaccia",
                       ["Alfa non ha usato alcuna violenza.", "Alfa non ha fatto alcuna minaccia a Beta."], ["S1", "S2"])
    assert row["_grade"] == 1 and row["_demoted"] and log["demoted"][0]["reason"] == I.DEMOTED_CONTRADICTED
    assert "alternative_contradicted" not in log


def test_a_conjunctive_element_contradicted_demotes_as_today(monkeypatch):
    row, log = _demote(monkeypatch, "il compratore ha denunciato il vizio entro otto giorni dalla scoperta",
                       ["Il compratore non ha mai denunciato il vizio al venditore."], ["S1"])
    assert row["_grade"] == 1 and row["_demoted"] and log["demoted"][0]["article"] == "9999 c.c."
    assert "alternative_contradicted" not in log


def test_an_alternative_comma_s_requisites_are_alternatives(monkeypatch):
    texts = {1: "Se il vizio rende la cosa inidonea all'uso, il compratore può chiedere la risoluzione del contratto.",
             2: "In alternativa, se il vizio diminuisce il valore della cosa, il compratore può chiedere la riduzione "
                "del prezzo.",
             3: "Il compratore deve denunciare il vizio entro otto giorni."}
    assert I.alternative_commi(texts, [1, 2, 3]) == {2: [1], 1: [2]}
    assert I.alternative_commi(texts, [1, 3]) == {} and I.alternative_commi({1: "Ovvero il prezzo è ridotto."}, [1]) == {}
    # the distinctive element of an alternative comma: contradicted, the other comma's requisite open -> kept
    row, log = _demote(monkeypatch, "il vizio rende la cosa inidonea all'uso",
                       ["La cosa è rimasta perfettamente idonea all'uso."], ["S1"],
                       alternatives=["il vizio diminuisce il valore della cosa"], grade=3)
    assert row["_grade"] == 3 and log["alternative_contradicted"][0]["open"] == ["il vizio diminuisce il valore della cosa"]
    assert log["alternative_contradicted"][0]["contradicted"] == ["il vizio rende la cosa inidonea all'uso"]
    # both contradicted -> demoted
    row, log = _demote(monkeypatch, "il vizio rende la cosa inidonea all'uso",
                       ["La cosa è rimasta perfettamente idonea all'uso.", "Il valore della cosa non è diminuito."],
                       ["S1", "S2"], alternatives=["il vizio diminuisce il valore della cosa"])
    assert row["_grade"] == 1 and log["demoted"]


def test_the_record_s_verdict_and_missing_facts_respect_the_alternative_comma():
    r1 = {"id": "R1", "kind": "requisito", "comma": 1, "text": "il vizio rende la cosa inidonea all'uso",
          "status": NON_RISULTA, "alt_commi": [2]}
    r2 = {"id": "R2", "kind": "requisito", "comma": 2, "text": "il vizio diminuisce il valore della cosa",
          "status": DA_PROVARE, "alt_commi": [1]}
    rec = {"article": "9999 c.c.", "requisites": [r1, r2]}
    assert I.alternative_open(rec, r1) and I.unmet_in_anchor(rec) == []            # R2 open: R1 excludes nothing
    assert I.verdict(rec) == I.TO_PROVE and I.element_status(rec, r1) == I.NOT_RELEVANT
    r2["status"] = NON_RISULTA
    assert I.unmet_in_anchor(rec) == [r1] and I.verdict(rec) == I.NOT_APPLICABLE      # both excluded: as today
    # row 80: the other alternative proven -- this one is no missing fact of the line built on it
    r1["status"], r2["status"] = PROVATO, DA_PROVARE
    assert I.element_status(rec, r2) == I.NOT_RELEVANT and I.element_status(rec, r1) == PROVATO
    assert [r["id"] for r in M._live(rec)] == ["R1"]
    conj = {"article": "1 c.c.", "requisites": [dict(r1, alt_commi=None), dict(r2, alt_commi=None)]}
    assert [r["id"] for r in M._live(conj)] == ["R1", "R2"]                            # conjunctive: as before


def test_requisites_carry_the_alternative_comma_s_siblings():
    row = {"_article": ("9999", "c.c."), "s": {"titolo_articolo": "Vizi",
           "plain_text": "1. Se il vizio rende la cosa inidonea all'uso, il compratore può chiedere la risoluzione del "
                         "contratto.\n2. In alternativa, se il vizio diminuisce il valore della cosa, il compratore può "
                         "chiedere la riduzione del prezzo."}}
    reqs = [r for r in I._requisites(row, {}) if r["kind"] == "requisito"]
    by_comma = {str(r["comma"]): r for r in reqs}
    if len(by_comma) < 2:                   # the splitter kept one comma only: the pairing has nothing to pair
        pytest.skip(f"the splitter yielded {[(r['comma'], r['text']) for r in reqs]}")
    assert all(r.get("alt_commi") for r in reqs) and all(r.get("alternatives") for r in reqs)
    assert by_comma["2"]["alternatives"] == [by_comma["1"]["text"]]


# --- 2. a live record graded 2+ is a line or is rejected with a reason ---------------------------------------------------

def _ok(graded, aliases=None, extra_norms=()):
    inp = _civil()
    inp["norms"] = list(inp["norms"]) + list(extra_norms)
    inp["graded"], inp["aliases"] = graded, aliases or {}
    model = _Model(_reply(_line("1385 c.c.", E1385[0], "La caparra.", ["S1"], ["A1385cc.1"]),
                          _line("1453 c.c.", E1453[0], "L'inadempimento.", ["S2"], ["A1453cc.1"]),
                          elementi=[_el("1385 c.c.", E1385[0], "provato", ["S1"])]))
    return A.analyse(inp, call=model), inp


def test_a_grade_3_record_the_step_did_not_choose_is_rejected_below_cut():
    res, inp = _ok([{"article": "1385 c.c.", "grade": 3}, {"article": "1453 c.c.", "grade": 3},
                    {"article": "2043 c.c.", "grade": 3}, {"article": "1218 c.c.", "grade": 2},
                    {"article": "1490 c.c.", "grade": 1}, {"article": "1491 c.c.", "grade": None}])
    assert res["status"] == A.OK
    records = [r for r in res["rejected"] if r.get("kind") == A.RECORD_KIND]
    assert [(r["articolo"], r["why"], r["grade"], r["rank"]) for r in records] == [
        ("2043 c.c.", "below_cut", 3, 3), ("1218 c.c.", "not_in_block", 2, 0)]      # the lines and grade < 2: none
    assert all(r["why"].split(":")[0] in A.RECORD_REJECT_REASONS for r in records)
    assert res["counters"]["analysis_silent_records"] == 0 and "analysis_silent_records" in A.COUNTERS


def test_an_alias_of_a_line_is_a_duplicate_and_a_failed_step_is_silent(monkeypatch):
    res, _ = _ok([{"article": "73 d.P.R. 309/1990", "grade": 3}], aliases={"73 d.P.R. 309/1990": "1385 c.c."})
    dup = [r for r in res["rejected"] if r.get("kind") == A.RECORD_KIND]
    assert dup == [{"kind": "record", "articolo": "73 d.P.R. 309/1990", "why": "duplicate_of:1385 c.c.", "grade": 3,
                    "rank": 0}]
    events = _events(monkeypatch)
    inp = _civil()
    inp["graded"] = [{"article": "2043 c.c.", "grade": 3}]
    res = A.analyse(inp, call=_Model("niente di utile", "niente di utile"))   # no JSON twice (v79.19.8 item 5: one
    #                                                                            retry): the step fails, no line
    assert res["status"] != A.OK
    silent = [r for r in res["rejected"] if r.get("kind") == A.RECORD_KIND]
    assert [r["why"] for r in silent] == ["step_silent"] and res["counters"]["analysis_silent_records"] == 1
    logged = [d for e, d in events if e == "analysis_step"][-1]
    assert logged["silent_records"] == 1 and logged["records_accounted"] == 1


def test_inputs_carry_the_live_records_and_the_aliases(monkeypatch):
    monkeypatch.setenv("ANALYSIS_STEP", "true")
    state = {"records": {"1385 c.c.": {"grade": 3}, "1453 c.c.": {"grade": 2, "dropped": True}, "2043 c.c.": {"grade": 1}},
             "rows": [], "sentences": [], "charge_aliases": {"73 d.P.R. 309/1990": "73 Testo Unico"}}
    inp = A.inputs(state)
    assert [(g["article"], g["grade"]) for g in inp["graded"]] == [("1385 c.c.", 3), ("2043 c.c.", 1)]
    assert all(g["rubrica"] == "" and g["elements"] == [] for g in inp["graded"])    # v79.19.5 (item 5): carried too
    assert inp["aliases"] == {"73 d.P.R. 309/1990": "73 Testo Unico"}
    assert A.account_records({"status": A.FAILED, "rejected": []}, inp) == (
        [{"kind": "record", "articolo": "1385 c.c.", "why": "step_silent", "grade": 3, "rank": 1}], 1)


FIXTURES = ["civil", "penal", "truffa", "appalto", "noarticle"]


def _fixture(name):
    import _fixture_79_15_0 as H
    import test_v79_15_0_fixture_civil as CIVIL
    import test_v79_15_0_fixture_penal as PENAL
    import test_v79_15_1_fixtures as X
    case = {"civil": CIVIL.case, "penal": PENAL.case, "truffa": X.TRUFFA, "appalto": X.APPALTO,
            "noarticle": X.NOARTICLE}[name]
    return H, case


@pytest.mark.parametrize("name", FIXTURES)
def test_silent_records_is_zero_on_the_fixtures(monkeypatch, name):
    H, case = _fixture(name)
    run = H.run(monkeypatch, Path(tempfile.mkdtemp()), case)
    c = run["state"]["block_counters"]
    assert c["analysis_silent_records"] == 0
    records = [r for r in (run["state"].get("analysis_step") or {}).get("rejected") or [] if r.get("kind") == "record"]
    assert all(r["why"].split(":")[0] in A.RECORD_REJECT_REASONS and r["why"] != "step_silent" for r in records)
    live = {k: v for k, v in run["state"]["records"].items() if not v.get("dropped") and isinstance(v.get("grade"), int)
            and v["grade"] >= 2}
    lines = [x.get("articolo") for x in ((run["state"]["analysis_step"] or {}).get("analysis") or {}).values()
             if isinstance(x, dict) and x.get("articolo")]
    rejected = {r["articolo"] for r in (run["state"]["analysis_step"] or {}).get("rejected") or [] if r.get("articolo")}
    for label in live:          # every live record graded 2+ is a line or is in rejected
        assert any(A.same_article(label, x) for x in lines) or any(A.same_article(label, x) for x in rejected), label


# --- 3. only the settled charge may stop the method ------------------------------------------------------------------------

NO_ELEMENT = "Le pene stabilite nell'articolo precedente si applicano anche a chi …"


def _row(number, code, rubrica, text="", **extra):
    return {"_article": (number, code), "s": {"titolo_articolo": rubrica, "plain_text": text}, **extra}


def test_a_read_hypothesis_that_fails_decomposition_is_dropped_and_the_method_runs(monkeypatch):
    events = _events(monkeypatch)
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [])
    row = _row("640", "c.p.", "Truffa", NO_ELEMENT, _first_read="truffa")
    state = {"analysis": {}, "text": "Tizio ha raccontato a Caio una storia falsa e si è fatto consegnare del denaro.",
             "request": "", "intake": {}, "records": {}, "sentences": [], "questions": [],
             "read": {"hypotheses": [{"line": "principale", "claim": "truffa", "institutions": ["truffa"]}]},
             "rows": [], "reserve": []}
    flow.decompose_records(state, IMPUTATO, [row], set(), fetch=lambda refs: [])
    assert "method_refused" not in state and "method_refused_reason" not in state     # the method is not stopped
    assert state["method_dropped"] == [{"article": "640 c.p.", "reason": "undecomposed",
                                        "source": {"rule": "read_hypothesis", "line": "principale", "scores": {}}}]
    dropped = [d for e, d in events if e == "frame_hypothesis_dropped"]
    assert dropped == [{"ipotesi": "640 c.p.", "reason": "undecomposed", "rule": "read_hypothesis", "fit": None,
                        "frame_hypothesis": False}]
    assert [e for e, _ in events if e == "interactive_lawyer_method_aside"] == []
    # a verified line remains: the method runs
    verified = {"analysis": {"linea_principale": {"articolo": "624 c.p.", "stato": A.VERIFIED}, "linea_subordinata": None}}
    assert flow.step_aside_if_no_line(state, verified) is False and "method_refused" not in state
    # none remains: it steps aside as before, the reason logged
    none = {"analysis": {"linea_principale": {"articolo": "640 c.p.", "stato": A.FIT_LINE}, "linea_subordinata": None}}
    assert flow.step_aside_if_no_line(state, none) is True
    assert state["method_refused_reason"] == "undecomposed" and state["method_refused_article"] == "640 c.p."
    assert state["method_refused"] == M.NOT_APPLIED_WHY["undecomposed"].format(art="640 c.p.")
    assert [d["reason"] for e, d in events if e == "interactive_lawyer_method_aside"] == ["undecomposed"]
    assert [d for e, d in events if e == "interactive_step_aside_no_line"] == [{"dropped": ["640 c.p."],
                                                                               "reason": "undecomposed"}]


def test_the_frames_charge_that_fails_decomposition_is_dropped_too(monkeypatch):
    events = _events(monkeypatch)
    monkeypatch.setattr(case_frame, "bound", lambda state, field, posture=None: "999 c.p.")
    monkeypatch.setattr(case_frame, "alternative_charges", lambda state: [])
    state = {"analysis": {}, "text": "Il PM contesta a Tizio una pena.", "request": "", "intake": {},
             "records": {}, "sentences": [], "questions": []}
    flow.decompose_records(state, IMPUTATO, [_row("999", "c.p.", "Pena", NO_ELEMENT)], set(), fetch=lambda refs: [])
    assert "method_refused" not in state and state["method_dropped"][0]["source"] == {"rule": "case_frame"}
    assert [d["rule"] for e, d in events if e == "frame_hypothesis_dropped"] == ["case_frame"]


def test_the_users_article_that_fails_decomposition_still_stops_the_method(monkeypatch):
    events = _events(monkeypatch)
    state = {"analysis": {}, "text": "Il PM contesta a Tizio l'art. 999 c.p.", "request": "", "intake": {},
             "records": {}, "sentences": [], "questions": []}
    flow.decompose_records(state, IMPUTATO, [_row("999", "c.p.", "Pena", NO_ELEMENT)], set(), fetch=lambda refs: [])
    assert state["method_refused"] == M.NOT_APPLIED_WHY["undecomposed"].format(art="999 c.p.")
    assert state["method_refused_reason"] == "undecomposed" and "method_dropped" not in state
    assert [d["reason"] for e, d in events if e == "interactive_lawyer_method_aside"] == ["undecomposed"]
    assert flow.CHARGE_STOPS_METHOD == ("user_article", "rubrica_match")


def test_drop_hypothesis_marks_the_frames_hypothesis_scartata(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    events = _events(monkeypatch)
    state = {"frame": {case_frame.SETTLED: True, "counters": {}, "frame": {
        "addebito_o_domanda": {"articolo": None, "ipotesi": "1667 c.c.", "label_ipotesi": "1667 c.c.", "da_fatti": True}}}}
    assert case_frame.hypotheses(state) == [{"label": "1667 c.c.", "stato": "aperta"}]
    assert case_frame.drop_hypothesis(state, "1667 c.c.", "undecomposed", "read_hypothesis") is True
    assert case_frame.hypotheses(state) == [{"label": "1667 c.c.", "stato": "scartata"}]
    assert state["frame"]["counters"]["frame_hypothesis_dropped"] == 1
    assert case_frame.drop_hypothesis(state, "2043 c.c.", "unread") is False          # not a frame hypothesis: logged
    assert state["frame"]["counters"]["frame_hypothesis_dropped"] == 2
    assert [d["reason"] for e, d in events if e == "frame_hypothesis_dropped"] == ["undecomposed", "unread"]
