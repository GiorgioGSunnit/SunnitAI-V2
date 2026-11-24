"""v79.18.0, items 6-8 -- the questions' rows through the family gate with the frame's verified rito (family_gate
.context(rito=…) / decide / rito_refused, special_laws.rito_families, case_frame.verified_rito, interactive_flow
.gate_quesito_rows / quesiti_counters: item 6); an article with no schema element is context only (analysis_step
.has_elements / listed_norms / no_element_articles / line_tier / _verify_line, counter candidates_no_elements_excluded:
item 7); the replay's seconds per stage (scripts/replay_cases.py stage_seconds: item 8). Synthetic texts only."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import test_v79_10a as G  # noqa: E402
import test_v79_15_0_analysis as T  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import family_gate  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import special_laws  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_cases_18", ROOT / "scripts" / "replay_cases.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

CC = G._row("1453", "c.c.", "Codice Civile 2026")


# --- 6. the frame's rito in the family gate; the questions' rows through it ----------------------------------------------

def test_the_table_lists_every_known_rito_with_families_the_laws_table_knows():
    table = special_laws.rito_families()
    riti = json.load(open(ROOT / "data" / "case_frame.json", encoding="utf-8"))["riti"]
    assert set(table) == {r for r in riti.values() if r != "sconosciuto"}
    known = {law.get("family") for law in special_laws.laws()} | set(special_laws.families())
    for rito, fams in table.items():
        assert fams and all(f in known for f in fams), (rito, fams)
    assert "amministrativo" not in table["civile"] and "tributario" not in table["civile"]
    assert "amministrativo" in table["amministrativo"] and "tributario" in table["tributario"]


def test_the_context_carries_the_rito_and_its_families_or_none():
    ctx = family_gate.context(G.SCIA, rito="civile")
    assert ctx["rito"] == "civile" and "edilizia" in ctx["rito_families"] and "amministrativo" not in ctx["rito_families"]
    assert family_gate.context(G.SCIA)["rito"] == "" and family_gate.context(G.SCIA)["rito_families"] is None
    assert family_gate.context(G.SCIA, rito="sconosciuto")["rito_families"] is None


def test_the_rito_refuses_a_law_the_signal_would_admit_the_codes_and_the_users_naming_pass():
    # the SCIA case: the amministrativo signal admits the L. 241/1990 -- unless the frame's verified rito is civile
    kept, dropped, _ = family_gate.gate([G.L241, CC], family_gate.context(G.SCIA), set())
    assert [r["_article"] for r in kept] == [G.L241["_article"], CC["_article"]] and dropped == []
    kept, dropped, _ = family_gate.gate([G.L241, CC], family_gate.context(G.SCIA, rito="civile"), set())
    assert [r["_article"] for r in kept] == [CC["_article"]]
    assert dropped[0]["article"] == "21-nonies Legge sul Procedimento Amministrativo 2026"
    assert dropped[0]["why"] == "fuori dal rito civile (amministrativo)"
    assert family_gate.rito_dropped(dropped) == [dropped[0]["article"]]
    # the rito amministrativo admits it; the user naming the law wins over the rito
    kept, dropped, _ = family_gate.gate([G.L241], family_gate.context(G.SCIA, rito="amministrativo"), set())
    assert len(kept) == 1 and dropped == []
    named = G.SCIA + " Si chiede se l'annullamento rispetti la legge 241/1990."
    kept, dropped, _ = family_gate.gate([G.L241], family_gate.context(named, rito="civile"), set())
    assert len(kept) == 1 and dropped == []
    # a user-named article passes whatever the rito
    kept, dropped, _ = family_gate.gate([G.L241], family_gate.context(G.SCIA, rito="civile"),
                                        {flow._key(G.L241["_article"])})
    assert len(kept) == 1 and dropped == []


def test_the_rito_never_gates_a_code_or_an_unknown_family():
    ctx = family_gate.context(G.CONTRACT, rito="penale")
    assert family_gate.rito_refused(family_gate.CODE, ctx) == ""
    assert family_gate.rito_refused(family_gate.OTHER, ctx) == ""
    assert family_gate.rito_refused("amministrativo", ctx) == "fuori dal rito penale (amministrativo)"
    assert family_gate.rito_refused("amministrativo", family_gate.context(G.CONTRACT)) == ""


def test_the_questions_rows_pass_the_gate_before_they_join_the_candidates():
    q_l241 = dict(G.L241, _quesito=[1], _origin="quesito:1")
    q_cc = dict(CC, _quesito=[1], _origin="quesito:1")
    overrides = []
    kept, dropped = flow.gate_quesito_rows([q_l241, q_cc], family_gate.context(G.SCIA, rito="civile"), set(), overrides)
    assert [r["_article"] for r in kept] == [CC["_article"]]
    assert dropped[0]["source"] == "quesiti" and dropped[0]["why"].startswith(family_gate.RITO_WHY)
    assert overrides == []
    # without the rito, the signal admits the law for the question too (the gate as 79.17.0 left it)
    kept, dropped = flow.gate_quesito_rows([q_l241, q_cc], family_gate.context(G.SCIA), set())
    assert len(kept) == 2 and dropped == []


def test_the_counters_count_the_gated_question_rows_and_the_ritos_drops():
    state = {"quesiti_retrieval": {"queries": 2, "added": ["1453 c.c."],
                                   "gated": [{"article": "21-nonies L. 241/1990", "family": "amministrativo",
                                              "why": "fuori dal rito civile (amministrativo)"}]},
             "family_gate": {"rito_dropped": ["21-nonies L. 241/1990", "51 d.P.R. 917/1986"]},
             "records": {"1453 c.c.": {}}, "intake": {}}
    flow.quesiti_counters(state)
    c = state["block_counters"]
    assert c["retrieval_quesiti_gated"] == 1 and c["family_rito_dropped"] == 2
    assert c["retrieval_quesiti_queries"] == 2 and c["fit_kept_from_quesiti"] == 1


def test_the_frames_rito_counts_only_when_verified_and_known(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    assert cf.verified_rito({"frame": {"rito": {"valore": "lavoro"}}, "verified": ["rito"]}) == "lavoro"
    assert cf.verified_rito({"frame": {"rito": {"valore": "lavoro"}}, "verified": ["cliente"]}) == ""
    assert cf.verified_rito({"frame": {"rito": {"valore": "sconosciuto"}}, "verified": ["rito"]}) == ""
    assert cf.verified_rito(None) == "" and cf.verified_rito({}) == ""


# --- 7. an article with no schema element is context only ---------------------------------------------------------------

NO_ELEMENTS = T._norm("646 c.p.", "Chiunque, per procurare a sé o ad altri un ingiusto profitto, si appropria il denaro o la cosa "
                      "mobile altrui di cui abbia, a qualsiasi titolo, il possesso, è punito, a querela della persona offesa.",
                      "Appropriazione indebita", [], "X")


def _penal_with_bare():
    return T._penal(norms=list(T.PENAL_NORMS) + [NO_ELEMENTS])


def test_the_article_block_lists_only_articles_with_an_element_and_the_line_lists_leave_it_out():
    inp = _penal_with_bare()
    assert [n["article"] for n in A.listed_norms(inp)] == [n["article"] for n in T.PENAL_NORMS]
    assert A.no_element_articles(inp) == ["646 c.p."]
    human = A.build_messages(inp)[1].content
    assert "ARTICOLO: 640 c.p." in human and "ARTICOLO: 646 c.p." not in human
    for kind in A.LINE_KINDS:
        assert "646 c.p." not in A.line_tier(inp, kind)[1]
    assert "646 c.p." not in A.allowed_text(inp)


def test_a_line_on_an_article_with_no_element_is_rejected_and_the_exclusion_counted():
    inp = _penal_with_bare()
    bad = T._reply(dict(T._line("640 c.p.", T.E640[0], "L'accusa.", ["S5"], ["A640cp.1"])),
                   subordinata={"articolo": "646 c.p.", "elemento_in_gioco": "A646cp:E1", "tesi": "Aggravante.",
                                "cita_fatti": ["S5"], "cita_norma": []})
    model = T._Model(bad, T.GOOD_PENAL)
    res = A.analyse(inp, call=model)
    assert res["status"] == "ok" and res["counters"]["candidates_no_elements_excluded"] == 1
    why = [r["why"] for r in res["rejected"] if r["articolo"] == "646 c.p."]
    assert why and why[0].startswith("art. 646 c.p.: nessun elemento dello schema, solo contesto, mai una linea")
    assert res["analysis"]["linea_subordinata"] is None or res["analysis"]["linea_subordinata"]["articolo"] != "646 c.p."
    # without the bare article nothing is counted
    assert A.analyse(T._penal(), call=T._Model(T.GOOD_PENAL))["counters"]["candidates_no_elements_excluded"] == 0


def test_the_counter_is_in_the_steps_counters_and_in_the_flows_copy():
    assert "candidates_no_elements_excluded" in A.COUNTERS
    assert A.empty_counters()["candidates_no_elements_excluded"] == 0
    assert "candidates_no_elements_excluded" in R.ANALYSIS_COUNTERS


# --- 8. the replay's seconds per stage -----------------------------------------------------------------------------------

def _timed_state():
    return {"timings": {"records_s": 3.4, "analysis_s": 12.3, "retrieval_s": 40.1, "applicability_s": 55.0,
                        "single_shot_s": 240.0},
            "block_counters": {"fit_seconds": 55.0, "analysis_seconds": 30.2},
            "case_frame": {"frame": {}, "counters": {"frame_seconds": 1.25}},
            "analysis_step": {"seconds": 30.2},
            "single_shot_timings": {"answer_s": 165.5, "total_s": 240.0}}


def test_the_stages_are_read_from_what_the_flow_measured():
    got = R.stage_seconds(_timed_state())
    assert got == {"frame": 1.2, "record": 3.4, "extraction": 12.3, "retrieval": 40.1, "fit": 55.0, "analysis": 30.2,
                   "write": 80.3, "total": 240.0}
    assert R.stage_seconds({}) == {}
    assert R.stage_seconds({"timings": {"records_s": "x"}}) == {}


def test_the_row_the_table_and_the_diff_carry_the_stages():
    row = R.measure("case_a", _timed_state(), "answer", 241.0)
    assert row["stages"]["write"] == 80.3
    table = R.case_table(row)
    assert "| stages (s) | frame 1.2, record 3.4, extraction 12.3, retrieval 40.1, fit 55.0, analysis 30.2, write 80.3, " \
           "total 240.0 |" in table
    before = dict(row, stages=dict(row["stages"], fit=70.0), seconds=250.0)
    changes = {c["key"]: c for c in R.diff_rows([row], [before])[0]["changes"]}
    assert changes["seconds.fit"] == {"key": "seconds.fit", "before": 70.0, "now": 55.0}
    assert changes["seconds"]["before"] == 250.0 and "seconds.frame" in changes
    assert "stages" not in R._row_values(row)       # the values compared apart from the seconds
