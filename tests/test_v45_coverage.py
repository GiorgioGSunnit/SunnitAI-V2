"""v45 coverage flags, each off by default: PLAN_RULE_INCLUDE, GRADE_ALL_PROMPT, GRADER_FIXED_ORDER, REGIME_CANDIDATES,
PARTIES_ENFORCED; and the Leonardo case's draft expectations. Invented rows only."""
import json
import os
import re

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import fact_signals as fs  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
import src.rag.plan_then_write as ptw  # noqa: E402

FLAGS = ("PLAN_RULE_INCLUDE", "GRADE_ALL_PROMPT", "GRADER_FIXED_ORDER", "REGIME_CANDIDATES", "PARTIES_ENFORCED",
         "CHECK_GRADED_ENABLED", "GRADER_EVIDENCE_ENABLED", "CHECK_PRIORITY_ENABLED", "CHECKLIST_GATED_ENABLED",
         "GRADED_MAX_CHECKED", "COMMA_MATERIAL_ENABLED", "WRITER_RULES_ENABLED", "REVIEW_FIXES_ENABLED",
         "GROUNDED_REVIEW_ENABLED")


@pytest.fixture(autouse=True)
def flags_off(monkeypatch):
    for name in FLAGS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def events(monkeypatch):
    seen = []
    monkeypatch.setattr(dg, "vlog", lambda name, data=None, **k: seen.append((name, data)))
    return seen


def row(number, code="c.c.", grade=None, heading="", issue="", **extra):
    tags = {"_grade": grade} if grade is not None else {}
    if issue:
        tags["_grade_issue"] = issue
    return {"d": {"id": f"doc-{code}", "name": code}, "_article": (number, code), **tags, **extra,
            "s": {"name": f"{number}.0.0", "plain_text": f"Testo dell'articolo {number}.", "titolo_articolo": heading}}


# --- A. PLAN_RULE_INCLUDE --------------------------------------------------------------------------------------------

def _plan():
    return {"parte_assistita": "Leonardo", "questioni": [
        {"titolo": "Servitù di passaggio", "argomento": "La servitù di passaggio sul fondo di Domitilla.",
         "norme": [("1062", "c.c.")], "pronunce": [], "obiezione": "", "replica": ""},
        {"titolo": "Tutela possessoria", "argomento": "Lo spoglio con il cancello.", "norme": [("1168", "c.c.")],
         "pronunce": [], "obiezione": "", "replica": ""}]}


def test_every_graded_prompt_row_absent_from_the_plan_goes_in_by_rule(monkeypatch):
    monkeypatch.delenv("EDGE_EXPAND", raising=False)        # its signal would join the reasons (test_edge_expand.py)
    rows = [row("1062", grade=3), row("1079", grade=2, issue="tutela della servitù di passaggio",
                                          _signals=["current", "frame", "distill"]),
            row("1170", grade=3, heading="Azione di manutenzione"),        # grade 3: whatever found it
            row("1061", grade=2, heading="Servitù non apparenti", _signals=["distill"], _institution=True),
            row("2043", grade=1),                                          # grade 1 without two signals
            row("1027", grade=3),                                          # not in the prompt's material
            row("1052", grade=3)]                                          # excluded by a fired rule
    articles = " ".join(f"[art. {n} c.c.]" for n in ("1062", "1079", "1170", "1061", "2043", "1052"))
    headings = ptw._headings(rows)
    issue_of = {r["_article"]: r.get("_grade_issue") or "" for r in rows}
    plan, inserted, skipped = ptw.include_by_rule(_plan(), rows, articles, headings, issue_of, [("1052", "c.c.")])
    assert [(e["article"], e["reason"]) for e in skipped] == [
        ("2043 c.c.", "grade 1 without two of frame/distill/institution/ruling_applied")]
    assert {e["article"]: e["signals"] for e in inserted}["1061 c.c."] == ["distill", "institution"]
    assert ptw.rule_inserted(plan) == {("1079", "c.c."), ("1170", "c.c."), ("1061", "c.c.")}
    where = {e["article"]: e["issue"] for e in inserted}
    assert where == {"1079 c.c.": "Servitù di passaggio", "1170 c.c.": ptw.RULE_INCLUDE_TITLE,
                     "1061 c.c.": "Servitù di passaggio"}
    assert [q["titolo"] for q in plan["questioni"]][-1] == ptw.RULE_INCLUDE_TITLE
    held = {ref for q in plan["questioni"] for ref in q["norme"]}
    assert {("1079", "c.c."), ("1170", "c.c."), ("1061", "c.c.")} <= held
    assert not held & {("2043", "c.c."), ("1027", "c.c."), ("1052", "c.c.")}
    assert all(e["reason"].startswith("grade ") for e in inserted)
    assert ptw.include_by_rule(plan, rows, articles, headings, issue_of, [("1052", "c.c.")])[1] == []    # idempotent


def test_an_article_check_plan_removed_is_never_added_back():
    rows = [row("1062", grade=3), row("1079", grade=2, issue="tutela della servitù di passaggio",
                                          _signals=["frame"], _institution=True),
            row("1061", grade=3, heading="Servitù non apparenti")]
    articles = "[art. 1062 c.c.] [art. 1079 c.c.] [art. 1061 c.c.]"
    issue_of = {r["_article"]: r.get("_grade_issue") or "" for r in rows}
    plan, inserted, skipped = ptw.include_by_rule(_plan(), rows, articles, ptw._headings(rows), issue_of, (),
                                                  [("1061", "c.c.")])
    assert [e["article"] for e in inserted] == ["1079 c.c."]
    assert skipped == [{"article": "1061 c.c.", "grade": "3", "signals": [], "reason": "check_plan_removed"}]
    assert ("1061", "c.c.") not in {ref for q in plan["questioni"] for ref in q["norme"]}


def test_make_plan_passes_what_check_plan_removed(monkeypatch, events):
    monkeypatch.setenv("PLAN_RULE_INCLUDE", "true")
    raw = {"questioni": [{"titolo": "Servitù di passaggio", "argomento": "La servitù di passaggio.",
                          "norme": ["art. 1062 c.c.", "art. 1061 c.c."]}]}
    monkeypatch.setattr(ptw, "_plan_json", lambda messages: raw)
    monkeypatch.setattr(dg, "_opposing_arguments", lambda *a, **k: [])
    monkeypatch.setattr(ptw, "_issue_check", lambda issue, rows, headings: [
        {"articolo": "art. 1062 c.c.", "pertinente": True, "contraddice": False},
        {"articolo": "art. 1061 c.c.", "pertinente": False, "contraddice": False}])
    rows = [row("1062", grade=3), row("1061", grade=3, heading="Distanze nelle costruzioni"),
            row("1079", grade=2, issue="servitù di passaggio", _signals=["frame", "distill"])]
    articles = "[art. 1062 c.c.] [art. 1061 c.c.] [art. 1079 c.c.]"
    plan = ptw.make_plan("Fatti.", "Domanda.", {}, articles, "", "", rows, {})
    held = {ref for q in plan["questioni"] for ref in q["norme"]}
    assert ("1061", "c.c.") not in held and ("1079", "c.c.") in held
    log = next(d for n, d in events if n == "case_analysis_plan_rule_include")
    assert log["skipped"] == [{"article": "1061 c.c.", "grade": "3", "signals": [], "reason": "check_plan_removed"}]


def test_a_grade_2_needs_one_of_frame_distill_institution(monkeypatch):
    monkeypatch.delenv("EDGE_EXPAND", raising=False)
    rows = [row("1062", grade=3),
            row("1079", grade=2, _signals=["current", "baseline"]),                    # other searches only
            row("2653", grade=2, _regime_candidate="azione_reale_immobiliare"),        # a regime candidate alone
            row("1061", grade=2, _signals=["frame"], _regime_candidate="x"),           # one strong signal: enough
            row("1158", grade=2, _signals=["frame", "distill"]),
            row("1051", grade=3, _regime_candidate="azione_reale_immobiliare")]        # grade 3: always
    articles = " ".join(f"[art. {n} c.c.]" for n in ("1062", "1079", "2653", "1061", "1158", "1051"))
    plan, inserted, skipped = ptw.include_by_rule(_plan(), rows, articles, ptw._headings(rows), {}, ())
    assert [e["article"] for e in inserted] == ["1061 c.c.", "1158 c.c.", "1051 c.c."]
    assert {e["article"]: (e["grade"], e["signals"]) for e in skipped} == {
        "1079 c.c.": ("2", ["current", "baseline"]), "2653 c.c.": ("2", ["regime_candidate"])}
    assert all(e["reason"] == "grade 2 without any of frame/distill/institution" for e in skipped)


def test_an_exempt_heading_or_crossref_row_counts_as_grade_2():
    from src.rag.nodes.cypher import CROSSREF_SOURCE
    rows = [row("1062", grade=3),
            row("1170", heading="Azione di manutenzione", _heading_match=True, _institution=True),   # exempt: heading
            row("1063", _source=CROSSREF_SOURCE, _signals=["distill"]),                             # exempt: crossref
            row("444", code="c.p.p.", _source=CROSSREF_SOURCE),                                      # crossref alone
            row("1169", _heading_match=True, _signals=["current"]),                                 # other search alone
            row("1171", _heading_match=True, _institution=True),                                    # not in the prompt
            row("1172", _signals=["frame"]),                                                        # ungraded, not exempt
            row("1173", grade=1, _heading_match=True, _institution=True)]                           # graded: its grade
    articles = ("[art. 1062 c.c.] [art. 1170 c.c.] [art. 1063 c.c.] [art. 444 c.p.p.] [art. 1169 c.c.] "
                "[art. 1172 c.c.] [art. 1173 c.c.]")
    plan, inserted, skipped = ptw.include_by_rule(_plan(), rows, articles, ptw._headings(rows), {}, ())
    assert {e["article"]: (e["grade"], e["exempt"]) for e in inserted} == {
        "1170 c.c.": ("2", "heading"), "1063 c.c.": ("2", "crossref")}
    assert {e["article"]: e.get("exempt") for e in skipped} == {"444 c.p.p.": "crossref", "1169 c.c.": "heading",
                                                               "1173 c.c.": None}     # graded 1: its own grade's rule
    assert next(e for e in skipped if e["article"] == "1173 c.c.")["grade"] == "1"
    assert ptw.rule_inserted(plan) == {("1170", "c.c."), ("1063", "c.c.")}
    missing = ptw._omitted(plan, [("A", "Testo senza citazioni.")], rows, [])      # unforced
    assert [m[0] for m in missing] == [("1062", "c.c.")]


def test_an_exempt_row_check_plan_removed_stays_out():
    rows = [row("1062", grade=3), row("1170", _heading_match=True, _institution=True)]
    plan, inserted, skipped = ptw.include_by_rule(_plan(), rows, "[art. 1062 c.c.] [art. 1170 c.c.]", {}, {}, (),
                                                  [("1170", "c.c.")])
    assert inserted == [] and [(e["article"], e["reason"]) for e in skipped] == [("1170 c.c.", "check_plan_removed")]


def test_a_grade_1_needs_two_distinct_signals_and_a_grade_0_never_goes_in(events, monkeypatch):
    monkeypatch.delenv("EDGE_EXPAND", raising=False)
    rows = [row("1062", grade=3),
            row("1079", grade=1, _signals=["frame", "distill"]),                         # two searches
            row("1061", grade=1, _signals=["frame"], _vote=True),                         # a search, voted by rulings
            row("1158", grade=1, _institution=True, _vote_stats={"rulings": 3}),          # applied by 3 similar rulings
            row("1051", grade=1, _signals=["distill"], _map=True),                        # a confirmed map hypothesis
            row("1027", grade=1, _signals=["frame", "frame", "current"]),                 # one distinct strong signal
            row("2653", grade=1, _signals=["frame"], _regime_candidate="x"),              # a regime candidate: not one
            row("1102", grade=1, _institution=True, _vote_stats={"rulings": 2}),          # 2 rulings: not applied
            row("1170", grade=0, _signals=["frame", "distill"], _institution=True)]       # grade 0: never
    articles = " ".join(f"[art. {n} c.c.]" for n in ("1062", "1079", "1061", "1158", "1051", "1027", "2653", "1102", "1170"))
    plan, inserted, skipped = ptw.include_by_rule(_plan(), rows, articles, ptw._headings(rows), {}, ())
    assert {e["article"]: (e["grade"], e["signals"]) for e in inserted} == {
        "1079 c.c.": ("1", ["frame", "distill"]), "1061 c.c.": ("1", ["frame", "ruling_applied"]),
        "1158 c.c.": ("1", ["institution", "ruling_applied"]), "1051 c.c.": ("1", ["distill", "ruling_applied"])}
    assert {e["article"] for e in skipped} == {"1027 c.c.", "2653 c.c.", "1102 c.c."}
    assert all(e["reason"] == "grade 1 without two of frame/distill/institution/ruling_applied" for e in skipped)
    assert "1170 c.c." not in {e["article"] for e in inserted + skipped}
    assert ptw.rule_inserted(plan) == {("1079", "c.c."), ("1061", "c.c."), ("1158", "c.c."), ("1051", "c.c.")}
    assert [m[0] for m in ptw._omitted(plan, [("A", "Testo senza citazioni.")], rows, [])] == [("1062", "c.c.")]


def test_a_rule_inserted_article_is_never_forced_into_a_paragraph():
    rows = [row("1062", grade=3), row("1158", grade=3)]
    plan, inserted, _ = ptw.include_by_rule(_plan(), rows, "[art. 1062 c.c.] [art. 1158 c.c.]", {}, {}, ())
    assert inserted and ptw.rule_inserted(plan) == {("1158", "c.c.")}
    missing = ptw._omitted(plan, [("A", "Testo senza citazioni.")], rows, [])
    assert [m[0] for m in missing] == [("1062", "c.c.")]
    plan, _ = ptw.one_issue_per_article(plan)                   # the tag survives the later steps
    assert ptw.rule_inserted(plan) == {("1158", "c.c.")}


def test_the_rule_is_off_by_default():
    assert not ptw.rule_include()


# --- B. GRADE_ALL_PROMPT ---------------------------------------------------------------------------------------------

def _budget(n):
    """A prompt with room for the first n articles of its rows."""
    def fit(system, facts, analysis, rows, rulings, request="", principles="", budget=0, left=None):
        refs = list(dict.fromkeys(r["_article"] for r in rows if r.get("_article")))
        if left is not None:
            left.extend(refs[n:])
        return "", "", ""
    return fit


def test_only_the_rows_reaching_the_prompt_are_graded_in_rounds(monkeypatch, events):
    monkeypatch.setattr(dg, "_fit_parere_prompt", _budget(4))
    rows = [row(str(n)) for n in range(1, 11)]
    texts = {f"{n} c.c.": "t" for n in range(1, 11)}
    by_label = {f"{n} c.c.": (str(n), "c.c.") for n in range(1, 11)}
    asked = []

    def grade_these(chosen):
        asked.append(list(chosen))
        return {dg._applicability_key(label): {"grade": 0 if label == "2 c.c." else 2} for label in chosen}, []
    graded, verdicts, ungraded = dg._grade_prompt_rows(texts, by_label, rows, grade_these, "f", "r", {}, "it")
    # 2 graded 0 leaves, its room goes to 5, graded in a second round; 6-10 never reach the prompt
    assert asked == [["1 c.c.", "2 c.c.", "3 c.c.", "4 c.c."], ["5 c.c."]]
    assert list(graded) == ["1 c.c.", "2 c.c.", "3 c.c.", "4 c.c.", "5 c.c."] and ungraded == []
    assert next(d for n, d in events if n == "case_analysis_grade_prompt")["rounds"] == [4, 1]


def test_the_rounds_stop_at_the_cap(monkeypatch, events):
    monkeypatch.setattr(dg, "_fit_parere_prompt", _budget(10))
    monkeypatch.setattr(dg, "_GRADE_ALL_PROMPT_CAP", 3)
    rows = [row(str(n)) for n in range(1, 11)]
    texts = {f"{n} c.c.": "t" for n in range(1, 11)}
    by_label = {f"{n} c.c.": (str(n), "c.c.") for n in range(1, 11)}
    graded, _, _ = dg._grade_prompt_rows(texts, by_label, rows, lambda c: ({}, list(c)), "f", "r", {}, "it")
    assert list(graded) == ["1 c.c.", "2 c.c.", "3 c.c."]


def test_a_final_round_grades_what_the_last_estimate_admits(monkeypatch, events):
    # h1's shape: every round's grade 0 frees room, so the four rounds end with articles in reach ungraded
    monkeypatch.setattr(dg, "_fit_parere_prompt", _budget(2))
    rows = [row(str(n), _fused_rank=100 + n) for n in range(1, 13)]
    texts = {f"{n} c.c.": "t" for n in range(1, 13)}
    by_label = {f"{n} c.c.": (str(n), "c.c.") for n in range(1, 13)}
    asked = []

    def grade_these(chosen):
        asked.append(list(chosen))
        return {dg._applicability_key(label): {"grade": 0} for label in chosen}, []
    graded, _, _ = dg._grade_prompt_rows(texts, by_label, rows, grade_these, "f", "r", {}, "it")
    log = next(d for n, d in events if n == "case_analysis_grade_prompt")
    assert log["rounds"] == [2, 2, 2, 2] and log["final_round"] == 2      # 9 and 10, which round 4's grade 0 let in
    assert asked[-1] == ["9 c.c.", "10 c.c."] and {"9 c.c.", "10 c.c."} <= set(graded)
    # their grade 0 frees room again: 11 and 12 are in reach, left ungraded, logged with place and fusion rank
    assert log["left_in_reach"] == [{"article": "11 c.c.", "check_order": 11, "fused_rank": 111},
                                    {"article": "12 c.c.", "check_order": 12, "fused_rank": 112}]
    assert log["left_because"] == "room freed by the final round"


def test_rows_left_in_reach_at_the_cap_are_logged_with_their_ranks(monkeypatch, events):
    monkeypatch.setattr(dg, "_fit_parere_prompt", _budget(10))
    monkeypatch.setattr(dg, "_GRADE_ALL_PROMPT_CAP", 3)
    rows = [row(str(n), _fused_rank=n) for n in range(1, 6)]
    texts = {f"{n} c.c.": "t" for n in range(1, 6)}
    by_label = {f"{n} c.c.": (str(n), "c.c.") for n in range(1, 6)}
    dg._grade_prompt_rows(texts, by_label, rows, lambda c: ({dg._applicability_key(k): {"grade": 2} for k in c}, []),
                          "f", "r", {}, "it")
    log = next(d for n, d in events if n == "case_analysis_grade_prompt")
    assert log["final_round"] == 0 and log["left_because"] == "cap"
    assert [(e["article"], e["check_order"], e["fused_rank"]) for e in log["left_in_reach"]] == [
        ("4 c.c.", 4, 4), ("5 c.c.", 5, 5)]


def test_nothing_left_when_the_rounds_converge(monkeypatch, events):
    monkeypatch.setattr(dg, "_fit_parere_prompt", _budget(4))
    rows = [row(str(n)) for n in range(1, 11)]
    texts = {f"{n} c.c.": "t" for n in range(1, 11)}
    by_label = {f"{n} c.c.": (str(n), "c.c.") for n in range(1, 11)}
    dg._grade_prompt_rows(texts, by_label, rows, lambda c: ({dg._applicability_key(k): {"grade": 2} for k in c}, []),
                          "f", "r", {}, "it")
    log = next(d for n, d in events if n == "case_analysis_grade_prompt")
    assert log["rounds"] == [4] and log["final_round"] == 0 and log["left_in_reach"] == [] and "left_because" not in log


def _grading_model(zero=()):
    def chat(messages, **kwargs):
        human = messages[-1].content
        ids = re.findall(r"^\[(A\d+)\] (.+)$", human, re.MULTILINE)
        return json.dumps([{"a": key, "g": 0 if label.strip() in zero else 2, "q": "q", "r": "r"} for key, label in ids])
    return chat


def test_the_check_grades_the_prompt_rows_and_leaves_the_rest_unchecked(monkeypatch, events):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    monkeypatch.setenv("GRADE_ALL_PROMPT", "true")
    monkeypatch.setattr(dg, "_fit_parere_prompt", _budget(3))
    monkeypatch.setattr(dg, "_chat", _grading_model(zero={"1 c.c."}))
    rows = [row(str(n)) for n in range(1, 8)]
    out = dg._filter_applicable_articles("Fatti.", "Domanda.", rows, {})
    graded = {r["_article"][0]: r.get("_grade") for r in out}
    assert "1" not in graded and graded["2"] == graded["3"] == graded["4"] == 2
    assert graded["5"] is None and graded["7"] is None             # never reached the prompt: unchecked, kept
    log = next(d for n, d in events if n == "case_analysis_applicability" and "unchecked_labels" in (d or {}))
    assert log["unchecked_labels"] == ["5 c.c.", "6 c.c.", "7 c.c."] and log["max_checked"].startswith("prompt")


def test_without_the_flag_the_slots_are_unchanged(monkeypatch, events):
    monkeypatch.setenv("CHECK_GRADED_ENABLED", "true")
    monkeypatch.setenv("GRADED_MAX_CHECKED", "2")
    monkeypatch.setattr(dg, "_fit_parere_prompt", lambda *a, **k: pytest.fail("no estimate without the flag"))
    monkeypatch.setattr(dg, "_chat", _grading_model())
    out = dg._filter_applicable_articles("Fatti.", "Domanda.", [row(str(n)) for n in range(1, 5)], {})
    assert [r.get("_grade") for r in out] == [2, 2, None, None]


def test_the_estimate_drops_grade_zero_and_tags_the_others(monkeypatch):
    seen = {}

    def fit(system, facts, analysis, rows, rulings, request="", principles="", budget=0, left=None):
        seen["rows"] = rows
        return "", "", ""
    monkeypatch.setattr(dg, "_fit_parere_prompt", fit)
    reach = dg._prompt_reach([row("1"), row("2"), row("3")], {("1", "c.c."): 0, ("2", "c.c."): 3}, "f", "r", {}, "it")
    assert reach == {("2", "c.c."), ("3", "c.c.")}
    assert seen["rows"][0]["_applicability"] == "applicabile" and "_grade" not in seen["rows"][1]


# --- C. GRADER_FIXED_ORDER -------------------------------------------------------------------------------------------

def _batches(monkeypatch, labels):
    sent = []

    def batch(case, texts, retry=False, system="", issues=None):
        sent.append(list(texts))
        return {label: {"article": label, "grade": 2} for label in texts}, ""
    monkeypatch.setattr(dg, "_graded_batch", batch)
    monkeypatch.setattr(dg, "_APPLICABILITY_BATCH", 3)
    dg._graded_verdicts("caso", {label: "t" for label in labels})
    return sorted(sent)


def test_the_batches_do_not_depend_on_the_retrieval_order(monkeypatch, events):
    monkeypatch.setenv("GRADER_FIXED_ORDER", "true")
    labels = ["2043 c.c.", "614-bis c.p.c.", "1218 c.c.", "614 c.p.c.", "21-nonies L241", "3 L241", "1176 c.c."]
    first, second = _batches(monkeypatch, labels), _batches(monkeypatch, list(reversed(labels)))
    assert first == second
    assert first == sorted([["3 L241", "21-nonies L241", "1176 c.c."], ["1218 c.c.", "2043 c.c.", "614 c.p.c."],
                            ["614-bis c.p.c."]])            # by code, then number, then suffix


def test_without_the_flag_the_order_is_retrieval_s(monkeypatch, events):
    assert _batches(monkeypatch, ["2043 c.c.", "1218 c.c.", "1176 c.c."]) == [["2043 c.c.", "1218 c.c.", "1176 c.c."]]


# --- D. REGIME_CANDIDATES --------------------------------------------------------------------------------------------

def _servitude_case(requests=()):
    """v48: with requests, the client's typed facts (client_roles reads them)."""
    case = {"frame": {"areas": ["civile"], "tasks": ["Come tutelare il passaggio"],
                      "central_issue_candidates": [{"nome": "Servitù di passaggio per destinazione del padre di famiglia"}]},
            "questioni": [{"titolo": "Acquisto per usucapione", "istituti": [{"nome": "azione confessoria"}]}]}
    if requests:
        case["fact_signals"] = {"facts": {"parties": [{"name": "Sempronio", "role": "cliente, proprietario"}],
                                          "client_side": "Sempronio", "requests": list(requests)}, "fired": []}
    return case


def test_the_regimes_the_case_issues_call_for():
    excluded = []
    found = {r["id"]: r for r in fs.regime_candidates(_servitude_case(), ["civile"], log=excluded)}
    assert set(found) == {"azione_reale_immobiliare"}
    terms = {t for t, _ in found["azione_reale_immobiliare"]["matched"]}
    assert {"servitù", "usucapione", "azione confessoria"} <= terms
    # v48: no facts showing the client's position: neither the negatoria nor the rivendicazione
    assert found["azione_reale_immobiliare"]["refs"] == [("1079", "c.c.")]
    assert {e.get("article") for e in excluded} >= {"948 c.c.", "949 c.c.", "1146 c.c.", "1167 c.c."}
    assert all(r["draft_pending_lawyer"] for r in fs.REGIME_CANDIDATES)
    assert all(r.get("client_role") for r in fs.REGIME_CANDIDATES) and all(t.get("client_role") for t in fs.PROCEDURAL_TOOLS)


def test_the_negatoria_only_for_the_owner_resisting_and_the_rivendicazione_only_for_who_claims_ownership():
    resists = fs.regime_candidates(_servitude_case(["Il vicino vanta una servitù di passaggio: Sempronio la contesta"]),
                                   ["civile"])
    assert ("949", "c.c.") in resists[0]["refs"] and ("948", "c.c.") not in resists[0]["refs"]
    owner = fs.regime_candidates(_servitude_case(["Ottenere la restituzione della stanza posseduta da Tizio"]), ["civile"])
    assert ("948", "c.c.") in owner[0]["refs"] and ("949", "c.c.") not in owner[0]["refs"]
    claims = _servitude_case(["far valere il diritto di passaggio"])
    claims["fact_signals"]["fired"] = [{"acquired_right": True, "institutions": ["servitù per destinazione del padre di famiglia"]}]
    refs = fs.regime_candidates(claims, ["civile"])[0]["refs"]
    assert ("949", "c.c.") not in refs and ("948", "c.c.") not in refs and ("1079", "c.c.") in refs


def test_a_regime_of_another_family_is_not_called():
    admin = {"questioni": [{"titolo": "Annullamento del provvedimento di diniego"}]}
    assert [r["id"] for r in fs.regime_candidates(admin, ["amministrativo"])] == ["procedimento_amministrativo"]
    assert fs.regime_candidates(admin, ["civile"]) == []
    late = {"questioni": [{"titolo": "Inadempimento del venditore e impossibilità sopravvenuta"}]}
    assert [r["id"] for r in fs.regime_candidates(late, ["civile"])] == ["responsabilita_contrattuale",
                                                                         "impossibilita_sopravvenuta"]
    assert [r["id"] for r in fs.regime_candidates({"questioni": [{"titolo": "Giudizio di ottemperanza"}]},
                                                  ["amministrativo"])] == ["ottemperanza"]


def test_the_candidates_enter_the_pool_once_and_are_graded(monkeypatch, events):
    monkeypatch.setenv("REGIME_CANDIDATES", "true")
    from src.rag import doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])
    asked = []

    def fetch(session, documents, refs, store=None):
        asked.extend(refs)
        return [row(n, c) for n, c in refs if n != "949"]
    monkeypatch.setattr(dg, "_fetch_any", fetch)
    picked = {"article_rows": [row("1079")], "selected": [("1079", "c.c.")]}
    case = _servitude_case(["Sempronio contesta la servitù e chiede la restituzione del fondo"])     # both roles (v48)
    out = dg._add_regime_candidates(None, None, "db", case, picked)
    assert ("1079", "c.c.") not in asked                      # already in the pool
    added = [r for r in out["article_rows"] if r.get("_source") == dg.REGIME_CANDIDATE_SOURCE]
    assert {r["_article"] for r in added} == {("948", "c.c."), ("1167", "c.c.")}      # 1167: usucapione's (v48)
    assert all(r["_regime_candidate"] in ("azione_reale_immobiliare", "usucapione") and r["_draft_pending_lawyer"]
               for r in added)
    log = next(d for n, d in events if n == "case_analysis_regime_candidates")
    assert log["not_found"] == ["949 c.c."] and log["already_in_pool"] == ["1079 c.c."]
    to_check, _ = dg._applicability_candidates("Fatti.", "", out["article_rows"])
    assert ("948", "c.c.") in to_check                        # never exempt from the check


def test_without_the_flag_nothing_is_added(monkeypatch):
    monkeypatch.setattr(dg, "_fetch_any", lambda *a, **k: pytest.fail("fetched"))
    picked = {"article_rows": [], "selected": []}
    assert dg._add_regime_candidates(None, None, "db", _servitude_case(), picked) is picked


# --- E. PARTIES_ENFORCED ---------------------------------------------------------------------------------------------

SIGNALS = {"defendants": [{"name": "Domitilla", "why": "autore dell'atto: appone un cancello"},
                          {"name": "Tizio", "why": "proprietario del fondo attraversato (terreno)"},
                          {"name": "Caio", "why": "proprietario del fondo attraversato (terreno)"}]}


def test_the_parties_line_names_them_all_with_their_basis(monkeypatch):
    assert fs.conclusions_block(SIGNALS) == ""
    monkeypatch.setenv("PARTIES_ENFORCED", "true")
    line = fs.conclusions_block(SIGNALS)
    assert line.startswith("- Parti da convenire in giudizio: Domitilla, Tizio, Caio (litisconsorzio necessario ex "
                           "art. 102 c.p.c. se ")
    assert "Tizio: proprietario del fondo attraversato" in line
    assert fs.conclusions_block({"defendants": SIGNALS["defendants"][:1]}) == ""      # one party: no line


def test_a_conclusions_line_s_norm_is_covered_like_a_plan_article(monkeypatch):
    monkeypatch.setenv("PARTIES_ENFORCED", "true")
    rows = [row("102", "c.p.c."), row("1079", grade=2)]
    plan = {"questioni": [{"titolo": "Passaggio", "argomento": "", "norme": [("1079", "c.c.")]}]}
    lines = ptw.conclusion_lines({"fact_signals": SIGNALS})
    assert len(lines) == 1 and lines[0].startswith("Parti da convenire")
    missing = ptw._omitted(plan, [("Applicazione", "Testo senza citazioni.")], rows, [], (), lines)
    assert [(m[0], m[1].get("conclusione")) for m in missing] == [(("1079", "c.c."), None), (("102", "c.p.c."), True)]
    cited = ptw._omitted(plan, [("Applicazione", "Vanno convenuti tutti (art. 102 c.p.c.).")], rows, [], (), lines)
    assert [m[0] for m in cited] == [("1079", "c.c.")]
    assert ptw._omitted(plan, [("A", "x")], rows, [], ()) == ptw._omitted(plan, [("A", "x")], rows, [], (), [])[:1]


def test_without_a_row_the_line_is_refused_for_no_text(monkeypatch):
    monkeypatch.setenv("PARTIES_ENFORCED", "true")
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: pytest.fail("called"))
    missing = ptw._omitted({"questioni": []}, [("A", "x")], [], [], (), ptw.conclusion_lines({"fact_signals": SIGNALS}))
    assert missing and missing[0][2] == {}
    entry = ptw._cover_one(*missing[0], {"questioni": []}, [], [], "Il caso.", None)
    assert entry["how"] == "refused" and entry["why"] == "no text"


def test_without_the_flag_there_are_no_conclusion_lines():
    assert ptw.conclusion_lines({"fact_signals": SIGNALS}) == []


# --- the rubrica stamp against the article's text -------------------------------------------------------------------

A1061 = ("Le servitù non apparenti non possono acquistarsi per usucapione o per destinazione del padre di famiglia. Non "
         "apparenti sono le servitù quando non si hanno opere visibili e permanenti destinate al loro esercizio.")


def _stamped(monkeypatch, draft):
    monkeypatch.setattr(dg, "_chat_json", lambda *a, **k: [{"articolo": "1061 c.c.", "compatibile": False,
                                                             "tipo": "rubrica"}])
    rows = [{"_article": ("1061", "c.c."), "d": {"name": "Codice Civile"},
             "s": {"name": "1061", "plain_text": A1061, "titolo_articolo": "Servitù non apparenti"}}]
    return dg._check_qualifications(draft, {}, rows, None, [], "Leonardo passa sul fondo.")


@pytest.mark.parametrize("draft", [
    "La servitù di Leonardo è apparente (art. 1061 c.c.).",
    "La servitù è apparente, perché esercitata con opere visibili e permanenti (art. 1061 c.c.)."])
def test_an_apparent_servitude_cited_to_1061_is_not_stamped(monkeypatch, events, draft):
    assert "[DA VERIFICARE" in _stamped(monkeypatch, draft)            # the heading alone: stamped, as before
    monkeypatch.setenv("PLAN_RULE_INCLUDE", "true")                    # not its flag
    assert "[DA VERIFICARE" in _stamped(monkeypatch, draft)
    monkeypatch.setenv("RUBRICA_TEXT_CHECK", "true")
    assert _stamped(monkeypatch, draft) == draft
    log = next(d for n, d in events[::-1] if n == "case_analysis_qualification")
    assert "appar" in log["not_stamped"]["1061 c.c."]


def test_a_clause_the_text_does_not_hold_is_still_stamped(monkeypatch, events):
    monkeypatch.setenv("RUBRICA_TEXT_CHECK", "true")
    out = _stamped(monkeypatch, "Il passaggio coattivo spetta al fondo intercluso (art. 1061 c.c.).")
    assert "[DA VERIFICARE: l'art. 1061 c.c. riguarda «Servitù non apparenti»]" in out


# --- E. regime-based defendants -------------------------------------------------------------------------------------

# The Leonardo case's facts as the extraction gives them: Mevio held Leonardo's and Sempronio's land in 1900, Sempronio
# sold to Domitilla in 2003, Annibale's land lies on the dirt road between two of Leonardo's.
LEONARDO = {"parties": [{"name": "Leonardo", "role": "cliente, proprietario"}, {"name": "Tizio", "role": "proprietario"},
                        {"name": "Caio", "role": "proprietario"}, {"name": "Sempronio", "role": "proprietario"},
                        {"name": "Domitilla", "role": "proprietaria"}, {"name": "Annibale", "role": "proprietario"},
                        {"name": "Mevio", "role": "proprietario"}],
            "property_history": [{"owner": "Mevio", "asset": "terreni di Leonardo e di Sempronio", "period": "inizio 1900"},
                                 {"owner": "Leonardo", "asset": "terreni in comune di Manciano"},
                                 {"owner": "Sempronio", "asset": "terreno con soprastante fabbricato", "to": "2003",
                                  "ended_by": "compravendita"},
                                 {"owner": "Domitilla", "asset": "terreno con soprastante fabbricato", "from": "2003"},
                                 {"owner": "Annibale", "asset": "terreno tra due fondi di Leonardo"}],
            "access": [{"asset": "fondi di Leonardo", "landlocked": "sì",
                        "how_reached": "strada sterrata che attraversa i terreni di Tizio, Caio e Sempronio"}],
            "events": [{"what": "appone un cancello chiuso a chiave", "who": "Domitilla", "kind": "chiusura/ostruzione accesso"}],
            "works_or_signs": [], "possession": []}
REAL = {"frame": {"areas": ["civile"]}, "questioni": [{"titolo": "Servitù di passaggio"}]}


def test_the_real_action_names_the_author_and_every_current_owner(monkeypatch):
    authors, owners, how = fs.real_action_defendants(LEONARDO)
    # never Mevio (owner in 1900, in the chain of title only), never Sempronio (sold); Annibale between Leonardo's land
    assert (authors, owners, how) == (["Domitilla"], ["Tizio", "Caio", "Annibale"], "current owners")
    signals = {"facts": LEONARDO, "defendants": SIGNALS["defendants"]}
    monkeypatch.setenv("PARTIES_ENFORCED", "true")
    assert fs.parties_line(signals, REAL) == (
        "Parti da convenire: Domitilla; da valutare la chiamata di Tizio, Caio, Annibale quali proprietari dei fondi "
        "interessati "
        "(litisconsorzio necessario ex art. 102 c.p.c. se la decisione non può pronunciarsi che in confronto di tutte)")
    assert fs.conclusions_block(signals, REAL).startswith("- Parti da convenire: Domitilla;")
    assert ptw.conclusion_lines({**REAL, "fact_signals": signals})[0].startswith("Parti da convenire: Domitilla;")


def test_no_owner_on_the_path_every_other_owner_is_named():
    facts = {**LEONARDO, "access": [{"asset": "fondo", "landlocked": "sì", "how_reached": "una strada sterrata"}],
             "property_history": LEONARDO["property_history"][:4]}
    authors, owners, how = fs.real_action_defendants(facts)
    assert authors == ["Domitilla"] and owners == ["Tizio", "Caio", "Annibale"] and how == "current owners"


def test_an_owner_the_facts_place_off_the_path_is_named_too():
    """v51: no filter by where the facts place the land; an owner only in property_history counts as well."""
    facts = {**LEONARDO, "parties": LEONARDO["parties"] + [{"name": "Bruto", "role": "proprietario"}],
             "works_or_signs": [{"what": "un capannone di Bruto", "crosses": ""}],
             "property_history": LEONARDO["property_history"] + [{"owner": "Livia", "asset": "un bosco"}]}
    assert fs.real_action_defendants(facts)[1] == ["Tizio", "Caio", "Annibale", "Bruto", "Livia"]


@pytest.mark.parametrize("mevio", [
    {"parties": LEONARDO["parties"][:6] + [{"name": "Mevio", "role": "proprietario originario"}],
     "property_history": LEONARDO["property_history"][1:]},
    {"property_history": LEONARDO["property_history"][1:],
     "events": LEONARDO["events"] + [{"what": "i terreni appartenevano a Mevio", "who": "Mevio", "kind": ""}]}])
def test_a_previous_owner_is_never_a_defendant_however_the_facts_say_it(mevio):
    assert "Mevio" not in fs.real_action_defendants({**LEONARDO, **mevio})[1]


def test_another_regime_keeps_the_parties_to_sue_line(monkeypatch):
    monkeypatch.setenv("PARTIES_ENFORCED", "true")
    other = {"frame": {"areas": ["civile"]}, "questioni": [{"titolo": "Inadempimento del contratto"}]}
    signals = {"facts": LEONARDO, "defendants": SIGNALS["defendants"]}
    assert fs.parties_line(signals, other).startswith("Parti da convenire in giudizio: Domitilla, Tizio, Caio (")


# --- the regime's procedural tools ---------------------------------------------------------------------------------

def test_the_real_action_brings_its_tools_and_102_only_with_the_parties_line(monkeypatch):
    signals = {"facts": LEONARDO, "defendants": SIGNALS["defendants"], "areas": ["civile"]}
    assert fs.regime_tools(signals, REAL) == []                                        # REGIME_CANDIDATES off
    monkeypatch.setenv("REGIME_CANDIDATES", "true")
    assert [t["id"] for t in fs.regime_tools(signals, REAL)] == ["trascrizione_domanda", "competenza_territoriale"]
    monkeypatch.setenv("PARTIES_ENFORCED", "true")
    tools = {t["id"]: t for t in fs.regime_tools(signals, REAL)}
    assert tools["litisconsorzio"]["refs"] == [["102", "c.p.c."]]
    assert tools["litisconsorzio"]["matched"] == ["parti da convenire: Domitilla", "da valutare: Tizio, Caio, Annibale"]
    assert tools["trascrizione_domanda"]["matched"] == [
        "trascrizione della domanda giudiziale (azione confessoria/negatoria) ai sensi dell'art. 2653, n. 1, c.c."]
    claimed = {**signals, "fired": [{"acquired_right": True, "institutions": ["servitù di passaggio"]}]}
    assert fs.regime_tools(claimed, REAL)[0]["matched"] == [
        "trascrizione della domanda giudiziale (azione confessoria) ai sensi dell'art. 2653, n. 1, c.c."]
    one = {**LEONARDO, "parties": LEONARDO["parties"][:1] + LEONARDO["parties"][4:5],   # Domitilla alone: one name
           "property_history": [e for e in LEONARDO["property_history"] if e["owner"] in ("Leonardo", "Domitilla")]}
    assert "litisconsorzio" not in {t["id"] for t in fs.regime_tools({**signals, "facts": one}, REAL)}
    criminal = {"frame": {"areas": ["penale"]}, "questioni": [{"titolo": "Servitù di passaggio"}]}
    assert fs.regime_tools(signals, criminal) == []                                    # civil cases only
    assert not {"trascrizione_domanda", "competenza_territoriale", "litisconsorzio"} & {
        t["id"] for t in fs.procedural_tools(LEONARDO, ["civile"])}                     # never by the facts' own rule


def test_the_tools_are_read_and_take_their_rubrica(monkeypatch, events):
    monkeypatch.setenv("REGIME_CANDIDATES", "true")
    monkeypatch.setenv("PARTIES_ENFORCED", "true")
    monkeypatch.setenv("FACT_REGIMES_ENABLED", "true")
    from src.rag import doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])
    monkeypatch.setattr(fs, "attach_term_sources", lambda *a, **k: None)
    rubriche = {"2653": "Altre domande e sentenze soggette a trascrizione", "21": "Foro per le cause relative a diritti "
                "reali e a possesso", "102": "Litisconsorzio necessario"}
    asked = []

    def fetch(session, documents, refs, store=None):
        asked.extend(refs)
        return [row(n, c, heading=rubriche.get(n, "")) for n, c in refs]
    monkeypatch.setattr(dg, "_fetch_any", fetch)
    signals = {"facts": LEONARDO, "defendants": SIGNALS["defendants"], "areas": ["civile"], "tools": []}
    rows = dg._regime_rows(None, None, "db", {**REAL, "fact_signals": signals})
    assert {tuple(r) for r in asked} == {("2653", "c.c."), ("21", "c.p.c."), ("102", "c.p.c.")}
    assert all(r["_procedural_tool"] for r in rows)
    texts = {t["id"]: t["text"] for t in signals["tools"]}
    assert texts["trascrizione_domanda"] == "Altre domande e sentenze soggette a trascrizione (art. 2653 c.c.)."
    assert texts["litisconsorzio"] == "Litisconsorzio necessario (art. 102 c.p.c.)."
    assert "Nel caso: parti da convenire: Domitilla; da valutare: Tizio, Caio, Annibale." in fs.tools_block(signals)
    assert "(art. 2653 c.c.). Nel caso: trascrizione della domanda giudiziale (azione" in fs.tools_block(signals)


# --- TERM_EXPIRY_TEXT --------------------------------------------------------------------------------------------------

def _non_use(present=""):
    facts = {k: ([] if isinstance(v, list) else "") for k, v in fs.SCHEMA.items()}
    facts.update(possession=[{"who": "Leonardo", "asset": "stradella", "until": "marzo 2016", "use": "passaggio"}],
                 present_date=present)
    return next(t for t in fs.compute_terms(facts) if t["id"] == "non_uso_servitu")["text"]


def test_the_non_use_term_says_when_it_ends(monkeypatch):
    before = _non_use("2018")
    assert before.startswith("estinzione per non uso: 2036 (")                       # off: unchanged
    monkeypatch.setenv("TERM_EXPIRY_TEXT", "true")
    assert _non_use("2018") == ("estinzione per non uso: non decorso, scade nel 2036 (20 anni dall'ultimo uso, "
                                "marzo 2016); non è il termine dell'usucapione")
    assert _non_use("2040").startswith("estinzione per non uso: già decorso, scaduto nel 2036 (")
    assert "decorre" not in _non_use("2018")


# --- F. the Leonardo case's draft expectations ------------------------------------------------------------------------

def test_the_leonardo_case_expects_the_draft_articles():
    path = os.path.join(os.path.dirname(__file__), "eval_narrative_cases.json")
    h3 = next(c for c in json.load(open(path, encoding="utf-8"))["cases"] if c["id"] == "h3")
    drafts = {(e["code"], e["article"]) for e in h3["expected"] if e.get("status") == "draft — pending lawyer validation"}
    assert drafts == {("cc", "1061"), ("cpc", "102"), ("cc", "2653")}


# --- the real Leonardo run (v45d): its extracted facts, through the plan step -----------------------------------------

REAL_FACTS = os.path.join(os.path.dirname(__file__), "fixtures", "leonardo_v45d_facts.json")
REAL_AREAS = ["diritti_reali", "contratti", "altro"]


@pytest.fixture
def real_case(monkeypatch, events):
    for name in ("PLAN_RULE_INCLUDE", "PARTIES_ENFORCED", "REGIME_CANDIDATES", "FACT_REGIMES_ENABLED",
                 "FACT_SIGNALS_ENABLED", "RUBRICA_TEXT_CHECK", "TERM_EXPIRY_TEXT", "GRADER_FIXED_ORDER"):
        monkeypatch.setenv(name, "true")
    monkeypatch.setattr(fs, "vlog", lambda name, data=None, **k: events.append((name, data)))
    facts = json.load(open(REAL_FACTS, encoding="utf-8"))
    assert facts["_regimes"] is True                       # the flag the extraction leaves among the facts
    signals = fs.signals_from(facts, "", REAL_AREAS, rules=True)
    return {"frame": {"areas": REAL_AREAS}, "fact_signals": signals,
            "questioni": [{"titolo": "Usucapione di servitù apparente", "istituti": [{"nome": "azione confessoria"}]}]}


def test_the_real_facts_give_the_parties_and_the_tools(monkeypatch, real_case, events):
    authors, owners, _ = fs.real_action_defendants(real_case["fact_signals"]["facts"])
    assert (authors, owners) == (["Domitilla"], ["Tizio", "Caio", "Annibale"])           # never Mevio, never Sempronio
    assert "Annibale" in fs.parties_line(real_case["fact_signals"], real_case)           # v51: on the real fixture
    from src.rag import doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])
    monkeypatch.setattr(fs, "attach_term_sources", lambda *a, **k: None)
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs, store=None: [row(n, c, heading=f"Rubrica {n}")
                                                                                        for n, c in refs])
    rows = dg._regime_rows(None, None, "db", real_case)
    ids = {t["id"] for t in real_case["fact_signals"]["tools"]}
    assert {"trascrizione_domanda", "competenza_territoriale", "litisconsorzio"} <= ids and rows
    line = ptw._conclusions(real_case)
    assert "Parti da convenire: Domitilla; da valutare la chiamata di Tizio, Caio, Annibale" in line
    assert "Mevio" not in line and "Sempronio" not in line
    assert not [d for n, d in events if n == "pipeline_exception"]


def test_the_plan_step_runs_on_the_real_facts(monkeypatch, real_case, events):
    monkeypatch.delenv("ROLE_FIT", raising=False)          # its calls are not stubbed here (tests/test_v48.py has them)
    raw = {"questioni": [{"titolo": "Usucapione di servitù apparente", "argomento": "La servitù di passaggio apparente.",
                          "norme": ["art. 1061 c.c.", "art. 1158 c.c."]}]}
    monkeypatch.setattr(ptw, "_plan_json", lambda messages: raw)
    monkeypatch.setattr(dg, "_opposing_arguments", lambda *a, **k: [])
    monkeypatch.setattr(ptw, "_issue_check", lambda issue, rows, headings: [
        {"articolo": f"art. {n} {c}", "pertinente": True, "contraddice": False} for n, c in issue["norme"]])
    rows = [row("1061", grade=3, heading="Servitù non apparenti"), row("1158", grade=2, _signals=["frame", "distill"]),
            row("1079", grade=2, _signals=["frame"], _institution=True), row("1027", grade=2, _signals=["current"])]
    articles = " ".join(f"[art. {n} c.c.]" for n in ("1061", "1158", "1079", "1027"))
    case = "Leonardo è proprietario di alcuni terreni... Domitilla appone un cancello chiuso a chiave."
    plan = ptw.make_plan(case, "", real_case, articles, "", "", rows, {})
    assert plan is not None and not [d for n, d in events if n == "pipeline_exception"]
    held = {ref for q in plan["questioni"] for ref in q["norme"]}
    assert ("1079", "c.c.") in held and ("1027", "c.c.") not in held
    assert ptw.rule_inserted(plan) == {("1079", "c.c.")}
    assert "Parti da convenire: Domitilla;" in ptw._conclusions(real_case)


# --- a failing v45 step is logged with its traceback and skipped ------------------------------------------------------

def _boom(*a, **k):
    raise TypeError("'bool' object is not iterable")


def test_a_failing_rule_include_leaves_the_plan_as_it_was(monkeypatch, real_case, events):
    monkeypatch.setattr(ptw, "include_by_rule", _boom)
    monkeypatch.setattr(ptw, "_plan_json", lambda messages: {"questioni": [
        {"titolo": "Servitù", "argomento": "La servitù.", "norme": ["art. 1061 c.c."]}]})
    monkeypatch.setattr(dg, "_opposing_arguments", lambda *a, **k: [])
    monkeypatch.setattr(ptw, "_issue_check", lambda *a: [])
    plan = ptw.make_plan("Fatti.", "", real_case, "[art. 1061 c.c.]", "", "", [row("1061", grade=3)], {})
    assert plan["questioni"][0]["norme"] == [("1061", "c.c.")]
    failure = next(d for n, d in events if n == "pipeline_exception")
    assert failure["stage"] == "plan_then_write.make_plan.include_by_rule"
    assert "Traceback (most recent call last)" in failure["traceback"] and "_boom" in failure["traceback"]


def test_a_failing_parties_line_leaves_the_conclusions_block(monkeypatch, real_case, events):
    monkeypatch.setattr(fs, "parties_line", _boom)
    block = ptw._conclusions(real_case)
    assert "Parti da convenire" not in block and "Mediazione" in block
    assert next(d for n, d in events if n == "pipeline_exception")["stage"] == "fact_signals.conclusions_block.parties_line"


def test_failing_regime_tools_leave_the_other_tools(monkeypatch, real_case, events):
    from src.rag import doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])
    monkeypatch.setattr(fs, "attach_term_sources", lambda *a, **k: None)
    monkeypatch.setattr(fs, "regime_tools", _boom)
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs, store=None: [row(n, c) for n, c in refs])
    rows = dg._regime_rows(None, None, "db", real_case)
    assert {r["_article"] for r in rows} >= {("700", "c.p.c.")}
    assert next(d for n, d in events if n == "pipeline_exception")["stage"] == "defensive_generation._regime_rows.regime_tools"


def test_failing_regime_candidates_leave_the_pool(monkeypatch, real_case, events):
    monkeypatch.setattr(fs, "regime_candidates", _boom)
    picked = {"article_rows": [], "selected": []}
    assert dg._add_regime_candidates(None, None, "db", real_case, picked) is picked
    assert next(d for n, d in events if n == "pipeline_exception")["stage"] == "defensive_generation._add_regime_candidates"


def test_an_escaping_exception_is_logged_with_its_traceback_and_raised(monkeypatch, events):
    monkeypatch.setattr(dg, "_case_analysis_pipeline", _boom)
    with pytest.raises(TypeError):
        dg.run_case_analysis_pipeline("Fatti.")
    failure = next(d for n, d in events if n == "pipeline_exception")
    assert failure["stage"] == "defensive_generation.run_case_analysis_pipeline" and "_boom" in failure["traceback"]


def test_the_exception_event_is_printed_whatever_verbose_says(monkeypatch, capsys):
    from src.rag import verbose_logger
    monkeypatch.setattr(verbose_logger, "VERBOSE", False)
    try:
        raise ValueError("x")
    except ValueError as exc:
        verbose_logger.vlog("pipeline_exception", verbose_logger.exception_data("s", exc))
    out = capsys.readouterr().out
    assert "[VERBOSE:pipeline_exception]" in out and "Traceback (most recent call last)" in out


# --- text_tidy: the punctuation closing a copy, when the text goes on with a letter (v45d) -----------------------------

from src.rag import text_tidy  # noqa: E402

PARTIES_TEXT = "art. 102 c.p.c. se la decisione non può pronunciarsi che in confronto di tutte"
TRASCRIZIONE_TEXT = "(art. 2653, n. 1, c.c.)"


@pytest.mark.parametrize("text, end, length", [
    (PARTIES_TEXT, PARTIES_TEXT.index("c.p.c.") + 1, 0),         # ".p.c. se": no space after the dot, never None
    (TRASCRIZIONE_TEXT, TRASCRIZIONE_TEXT.index("c.c.") + 1, 0),  # ".c.)"
    (TRASCRIZIONE_TEXT, TRASCRIZIONE_TEXT.index("2653") + 4, 1),  # ", n. 1": the comma, as before
    ("La domanda è fondata. Altro.", 21, 0), ("La domanda è fondata. Altro.", 20, 1), ("Fine.)", 4, 2)])
def test_the_closing_punctuation_is_always_a_length(text, end, length):
    assert text_tidy._closing_punctuation(text, end) == length


@pytest.mark.parametrize("text", [f"Vanno convenuti tutti ex {PARTIES_TEXT}.",
                                  f"Va chiesta la trascrizione della domanda giudiziale {TRASCRIZIONE_TEXT}."])
def test_a_copy_ending_inside_a_citation_is_dealt_with(text):
    for at in [m.start() + 1 for m in re.finditer(r"c\.", text)]:           # a copy ending after any "c" of a sigla
        out, how = text_tidy._without_copy(text, 0, at, ("102", "c.p.c."))
        assert isinstance(out, str) and how


A102 = ("Se la decisione non può pronunciarsi che in confronto di più parti, queste debbono agire o essere convenute nello "
        "stesso processo. Se questo è promosso da alcune o contro alcune soltanto di esse, il giudice ordina "
        "l'integrazione del contraddittorio in un termine perentorio da lui stabilito.")
A2653 = ("Devono essere trascritte, qualora si riferiscano ai diritti menzionati nell'articolo 2643: 1) le domande dirette a "
         "rivendicare la proprietà o altri diritti reali di godimento su beni immobili e le domande dirette "
         "all'accertamento dei diritti stessi.")


def test_final_checks_and_text_tidy_run_on_the_real_leonardo_material(monkeypatch, real_case, events):
    """The material the writer got on the real run -- the conclusions line and the tools from the v45d facts -- copied
    into the sections as a writer copies it, through final_checks and the pipeline's text_tidy step."""
    from src.rag import doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])
    monkeypatch.setattr(fs, "attach_term_sources", lambda *a, **k: None)
    texts = {"2653": A2653, "102": A102}
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs, store=None: [
        {**row(n, c, heading={"2653": "Altre domande e sentenze soggette a trascrizione", "102": "Litisconsorzio "
                               "necessario", "21": "Foro per le cause relative a diritti reali"}.get(n, "")),
         "s": {"name": n, "plain_text": texts.get(n, f"Testo dell'articolo {n}."), "titolo_articolo": ""}} for n, c in refs])
    rows = dg._regime_rows(None, None, "db", real_case)
    conclusions, tools = ptw._conclusions(real_case), fs.tools_block(real_case["fact_signals"])
    assert "art. 102 c.p.c. se la decisione" in conclusions and "2653, n. 1, c.c." in tools
    sections = [("Questioni", f"{conclusions}\n\nLeonardo deve convenire Domitilla; {A102[:120]} (art. 102 c.p.c.) se la "
                              "decisione riguarda tutti i fondi."),
                ("Strumenti", f"{tools}\n\nVa chiesta la trascrizione della domanda giudiziale (azione confessoria) ai "
                              f"sensi dell'art. 2653, n. 1, c.c. {A2653[:150]} (art. 2653, n. 1, c.c.) e poi la causa.")]
    checked = ptw.final_checks(sections, rows, ptw.statement_context(real_case))
    assert len(checked) == 2 and all(text for _, text in checked)
    draft = "\n\n".join(f"**{i}. {t}**\n\n{x}" for i, (t, x) in enumerate(checked, 1))
    tidied, _ = text_tidy.against_articles(draft, rows)
    tidied, _ = text_tidy.tidy(tidied)
    tidied, _ = text_tidy.headed_norms(tidied, rows)
    assert tidied and not [d for n, d in events if n in ("pipeline_exception", "post_writing_step_failed")]
