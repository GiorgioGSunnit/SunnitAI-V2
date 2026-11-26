"""v79.19.3, Part A -- Antonelo's strategy_review_v79192 of 26 Nov, rows 76 and 77 (the model faked; synthetic texts,
invented names; nothing named after a case):

1. row 76: the method stepping aside at the decomposition («unread», «undecomposed») logs the real reason, the
   article and the charge rule that produced it; a missing reason is «unknown»; lawyer_method_aside fires on every
   step-aside path. Logging only.
2. row 77, fix 1: the Capo-scope gate keeps what the similar rulings point to (cap_anchors' labels: _vote, _map,
   _vote_stats at MIN_RULINGS) -- kept, never widening the scope.
3. row 77, fix 2: the map's own first graded hypotheses enter the candidates whether or not a query channel found them
   (confirmed_hypotheses' third way, «map»; map_hypotheses_added logged); grading and the fit decide.
4. row 77, fix 3: code_scope_gate counts the rows it drops when the case has no unit of its own (units_empty).
"""

import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import case_frame, interactive_flow as flow, lawyer_method as M, posture  # noqa: E402
from src.rag import source_tiers as st, topic_groups as T, verbose_logger  # noqa: E402
from test_v79_10_5 import IMPUTATO  # noqa: E402

SRC = HERE.parent / "src" / "rag"


def _events(monkeypatch):
    events = []
    monkeypatch.setattr(verbose_logger, "vlog", lambda e, d=None, ms=None: events.append((e, d)))
    return events


def _row(number, code, rubrica, text="", **extra):
    return {"_article": (number, code), "s": {"titolo_articolo": rubrica, "plain_text": text}, **extra}


def _labels(rows):
    return [f"{r['_article'][0]} {r['_article'][1]}" for r in rows]


NO_ELEMENT = "Le pene stabilite nell'articolo precedente si applicano anche a chi …"     # splits into no element
DRAFT = "**Sintesi**\n\nUn parere.\n\n**Norme applicabili**\n\nL'art. 1218 c.c. regola l'inadempimento."


# --- 1. row 76: the step-aside reason ---------------------------------------------------------------------------------------

def test_the_vocabulary_has_the_decompositions_two_reasons_and_unknown_is_never_the_placeholder():
    assert flow.STEP_ASIDE_REASONS == ("no_action", "placeholder", "unread", "undecomposed")
    assert flow.step_aside_reason("") == flow.step_aside_reason(None) == flow.step_aside_reason("altro") == "unknown"
    assert flow.step_aside_reason("unread") == "unread"
    assert flow.method_aside_record("altro", {}, [])["reason"] == "unknown"
    got = flow.method_aside_record("unread", {"why": "w"}, [], article="999 c.p.", charge={"rule": "user_article"})
    assert got["article"] == "999 c.p." and got["charge_rule"] == {"rule": "user_article"} and got["clients"] == []


def test_an_undecomposable_users_article_logs_undecomposed_the_article_and_the_rule(monkeypatch):
    events = _events(monkeypatch)
    row = _row("999", "c.p.", "Pena", NO_ELEMENT)
    state = {"analysis": {}, "text": "Il PM contesta a Tizio l'art. 999 c.p.", "request": "", "intake": {},
             "records": {}, "sentences": [], "questions": []}
    flow.decompose_records(state, IMPUTATO, [row], set(), fetch=lambda refs: [])
    assert state["method_refused"] == M.NOT_APPLIED_WHY["undecomposed"].format(art="999 c.p.")   # unchanged
    assert state["method_refused_reason"] == "undecomposed" and state["method_refused_article"] == "999 c.p."
    assert state["method_refused_why"] == {"rule": "user_article"}
    aside = [d for e, d in events if e == "interactive_lawyer_method_aside"]
    assert len(aside) == 1 and aside[0]["reason"] == "undecomposed" and aside[0]["article"] == "999 c.p."
    assert aside[0]["charge_rule"] == {"rule": "user_article"} and aside[0]["why"] == state["method_refused"]
    assert state["method_aside"] == aside[0]
    flow.step_aside_answer(state, DRAFT, [], state["text"])
    got = [d for e, d in events if e == "interactive_step_aside_answer"][-1]
    assert got["reason"] == "undecomposed" and got["article"] == "999 c.p." and got["why"] == {"rule": "user_article"}


def test_an_unread_charge_logs_unread(monkeypatch):
    events = _events(monkeypatch)
    state = {"analysis": {}, "text": "Il PM contesta a Tizio l'art. 999 c.p.", "request": "", "intake": {},
             "records": {}, "sentences": [], "questions": []}
    flow.decompose_records(state, IMPUTATO, [], set(), fetch=lambda refs: [])
    assert state["method_refused"] == M.NOT_APPLIED_WHY["unread"].format(art="999 c.p.")
    assert state["method_refused_reason"] == "unread" and state["method_refused_article"] == "999 c.p."
    assert [d["reason"] for e, d in events if e == "interactive_lawyer_method_aside"] == ["unread"]


def test_a_read_hypothesis_carries_its_line_and_scores(monkeypatch):
    events = _events(monkeypatch)
    monkeypatch.setattr(posture, "alternative_rows", lambda s, rows=(): [])
    row = _row("640", "c.p.", "Truffa", NO_ELEMENT, _first_read="truffa")
    state = {"analysis": {}, "text": "Tizio ha raccontato a Caio una storia falsa e si è fatto consegnare del denaro.",
             "request": "", "intake": {}, "records": {}, "sentences": [], "questions": [],
             "read": {"hypotheses": [{"line": "principale", "claim": "truffa", "institutions": ["truffa"]}]},
             "rows": [], "reserve": []}
    flow.decompose_records(state, IMPUTATO, [row], set(), fetch=lambda refs: [])
    assert state["method_refused_reason"] == "undecomposed" and state["method_refused_article"] == "640 c.p."
    assert state["method_refused_why"] == {"rule": "read_hypothesis", "line": "principale", "scores": {}}
    aside = [d for e, d in events if e == "interactive_lawyer_method_aside"][-1]
    assert aside["charge_rule"]["rule"] == "read_hypothesis"
    # the scores: a record's (integrato, escluso, elements), the numbers choose_charge ranks by
    records = {"612 c.p.": {"requisites": [{"id": "R1", "kind": "requisito", "status": M.PROVATO},
                                           {"id": "R2", "kind": "requisito", "status": M.NON_RISULTA},
                                           {"id": "R3", "kind": "requisito", "status": "da_provare"}]}}
    got = flow.read_source(records, ["612 c.p.", "624 c.p."], {"line": "subordinata"})
    assert got == {"rule": "read_hypothesis", "line": "subordinata", "scores": {"612 c.p.": [1, 1, 3]}}


def test_the_frames_charge_carries_the_case_frame_rule(monkeypatch):
    events = _events(monkeypatch)
    monkeypatch.setattr(case_frame, "bound", lambda state, field, posture=None: "999 c.p.")
    monkeypatch.setattr(case_frame, "alternative_charges", lambda state: [])
    row = _row("999", "c.p.", "Pena", NO_ELEMENT)
    state = {"analysis": {}, "text": "Il PM contesta a Tizio una pena.", "request": "", "intake": {},
             "records": {}, "sentences": [], "questions": []}
    flow.decompose_records(state, IMPUTATO, [row], set(), fetch=lambda refs: [])
    assert state["method_refused_reason"] == "undecomposed" and state["method_refused_why"] == {"rule": "case_frame"}
    assert [d["charge_rule"] for e, d in events if e == "interactive_lawyer_method_aside"] == [{"rule": "case_frame"}]


def test_a_missing_reason_logs_unknown_and_the_refusal_branch_carries_the_charge(monkeypatch):
    events = _events(monkeypatch)
    flow.step_aside_answer({"records": {}}, DRAFT, [], "Il caso.")
    got = [d for e, d in events if e == "interactive_step_aside_answer"][-1]
    assert got["reason"] == "unknown" and got["article"] == "" and got["why"] == {}
    flow.step_aside_answer({"records": {}, "method_refused_reason": "placeholder", "method_refused_article": "640 c.p.",
                            "method_refused_why": {"rule": "rubrica_match"}}, DRAFT, [], "Il caso.")
    got = [d for e, d in events if e == "interactive_step_aside_answer"][-1]
    assert got["reason"] == "placeholder" and got["article"] == "640 c.p." and got["why"] == {"rule": "rubrica_match"}
    src = (SRC / "interactive_flow.py").read_text(encoding="utf-8")
    assert 'state["method_refused_article"] = str(charge_held.get("article") or "")' in src   # the refusal branch
    assert '"method_refused_why", "method_aside"):' in src                                    # reset per answer


# --- 2. row 77, fix 1: the scope gate keeps what the similar rulings point to ---------------------------------------------

def _read(*hyps):
    return {"hypotheses": [{"line": line, "institutions": list(insts), "rubricated": list(rub), "claim": claim}
                           for line, insts, rub, claim in hyps]}


SCOPE_ROWS = [
    _row("575", "c.p.", "Omicidio", _first_read="omicidio", _fused_rank=1),
    _row("610", "c.p.", "Violenza privata", _vote=True, _fused_rank=40),          # the similar rulings voted it
    _row("612", "c.p.", "Minaccia", _fused_rank=41),                              # its Capo, pointed to by nothing
    _row("640", "c.p.", "Truffa", _vote_stats={"rulings": 3}, _fused_rank=42),    # MIN_RULINGS similar rulings
    _row("624", "c.p.", "Furto", _map=True, _fused_rank=43),                      # a map hypothesis
]


def test_a_voted_row_the_scope_gate_would_drop_passes_it_and_sits_in_the_cap():
    read = _read(("principale", ["omicidio"], ["Omicidio"], "la morte di Caio"))
    without = flow.code_scope(SCOPE_ROWS, "Tizio ha ucciso Caio.", read)
    assert _labels(without["_rows"]) == ["575 c.p."]                              # 79.19.2: all four dropped
    held = flow.cap_anchors(SCOPE_ROWS, [], [])
    assert held == ["610 c.p.", "640 c.p.", "624 c.p."]
    got = flow.code_scope(SCOPE_ROWS, "Tizio ha ucciso Caio.", read, [], held)
    assert _labels(got["_rows"]) == ["575 c.p.", "610 c.p.", "640 c.p.", "624 c.p."]
    assert got["anchors_held"] == held and [x["article"] for x in got["dropped"]] == ["612 c.p."]   # never widened
    assert got["kept_by"] == {x: "indicata dalle sentenze simili" for x in held}
    capped, log = flow.grade_cap(got["_rows"], [], cap=1, anchors=held)
    assert _labels(capped) == held and log["anchors_kept"] == held       # inside the cap whatever their rank


def test_the_gate_keeps_a_held_label_like_a_rules_anchor():
    log = {}
    kept = T.code_scope_gate(SCOPE_ROWS, ["omicidio"], pass_only=["612 c.p."], log=log, held=["610 c.p."])
    assert _labels(kept) == ["575 c.p.", "610 c.p.", "612 c.p."]
    assert log["kept_by"] == {"610 c.p.": "indicata dalle sentenze simili", "612 c.p.": "ancora di una regola"}
    assert "units_empty" not in log


def test_the_flow_computes_the_held_labels_before_the_gate_from_the_fusion_selects_tags():
    src = (SRC / "interactive_flow.py").read_text(encoding="utf-8")
    assert "held_scope = _isolated(\"cap_anchors_scope\", lambda: cap_anchors(article_rows, pre, anchor_keep), [])" in src
    assert "anchor_keep, held_scope), None)" in src
    assert src.index('article_rows = picked["article_rows"]') < src.index("held_scope = ")      # tags already on the rows
    dg = (SRC / "defensive_generation.py").read_text(encoding="utf-8")
    start = dg.index("def _select_articles_fusion(")
    select = dg[start:dg.index("\ndef ", start)]
    for tag in ('row["_vote_stats"] = dict(vote_stats[ref])', 'row["_map"] = True', 'row["_vote"] = True'):
        assert tag in select            # the three tags are attached in the fusion select, before the flow's gates


# --- 3. row 77, fix 2: the map's own hypotheses graded -------------------------------------------------------------------

def test_the_maps_first_graded_hypotheses_are_candidates_without_a_channel():
    hyps = [(str(1000 + i), "c.c.") for i in range(17)]
    stats = {("1001", "c.c."): {"rulings": 3}}
    kept, why = st.confirmed_hypotheses(hyps, [("1000", "c.c.")], stats)
    assert kept == hyps[:15]
    assert why["channel"] == ["1000 c.c."] and why["rulings"] == ["1001 c.c."]
    assert why["map"] == [f"{1000 + i} c.c." for i in range(2, 15)] and why["rejected"] == ["1015 c.c.", "1016 c.c."]
    assert st.MAP_HYPOTHESES_GRADED == 15
    kept, why = st.confirmed_hypotheses(hyps, [("1000", "c.c.")], stats, graded=0)      # the two old ways alone
    assert kept == hyps[:2] and why["map"] == [] and len(why["rejected"]) == 15


def test_the_added_hypotheses_are_counted_and_tagged_map_after_the_cap():
    dg = (SRC / "defensive_generation.py").read_text(encoding="utf-8")
    assert 'vlog("map_hypotheses_added", {"count": len(why.get("map") or [])' in dg
    assert "map_rulings = [ref for ref in confirmed if ref not in channel_refs]" in dg
    assert "capped += [ref for ref in map_rulings if ref not in keep and ref not in capped]" in dg   # after the cap
    assert "if ref in map_channel or ref in map_rulings:" in dg and 'row["_map"] = True' in dg
    src = (SRC / "source_tiers.py").read_text(encoding="utf-8")
    assert 'for a, c in out["articles"][:MAP_HYPOTHESES_GRADED]]' in src     # the logged ones are the graded ones


def test_a_map_hypothesis_reaches_grading_and_the_fit_still_removes_it(monkeypatch):
    rows = [_row("1218", "c.c.", "Responsabilità del debitore", "Il debitore che non esegue esattamente la prestazione "
                 "dovuta è tenuto al risarcimento del danno.", _map=True, _fused_rank=99, _signals=["map"])] + \
        [{"_article": (str(1000 + i), "c.c."), "_fused_rank": i, "s": {}} for i in range(1, 4)]
    held = flow.cap_anchors(rows, [], [])
    capped, log = flow.grade_cap(rows, [], cap=2, anchors=held)
    assert "1218 c.c." in _labels(capped) and log["anchors_kept"] == ["1218 c.c."]     # graded whatever its rank
    monkeypatch.setenv("APPLICABILITY_FIT", "true")
    state = {"analysis": {}, "text": "Marta Galli chiede il risarcimento alla Edilnova S.r.l.", "request": "",
             "intake": {}, "records": {}, "sentences": [], "questions": [], "ranked": ["1218 c.c."]}
    seen = {}

    def fit(wanted, have):          # the judge says no: the removal recorded as applicability_fit does
        seen["wanted"] = list(wanted)
        state["fit_removed"] = [{"article": "1218 c.c.", "fit": "no"}]
        return {"1218 c.c."}
    attore = {"rito": "ordinario", "family": "civile", "role": "attore", "stage": "pre_filing"}
    flow.decompose_records(state, attore, rows, set(), fetch=lambda refs: [], fit=fit)
    assert seen["wanted"] == ["1218 c.c."] and state["decompose_why"]["1218 c.c."] == "fit_removed"
    assert "1218 c.c." not in state["records"]


# --- 4. row 77, fix 3: the count with no unit of the case's own ------------------------------------------------------------

def test_units_empty_counts_what_the_gate_drops_and_changes_nothing():
    rows = [_row("575", "c.p.", "Omicidio", _multi_signal=True),          # no row of the read's institution: no unit
            _row("610", "c.p.", "Violenza privata"), _row("612", "c.p.", "Minaccia"),
            _row("52", "c.p.", "Difesa legittima"),                                      # no unit: never dropped
            _row("640", "c.p.", "Truffa", _vote=True)]
    log = {}
    kept = T.code_scope_gate(rows, ["lesioni personali"], log=log, held=["640 c.p."])
    assert log["units"] == [] and log["units_empty"] == 3
    assert _labels(kept) == ["52 c.p.", "640 c.p."] and [x["article"] for x in log["dropped"]] == ["575 c.p.", "610 c.p.", "612 c.p."]
    log = {}
    T.code_scope_gate(rows, ["omicidio"], log=log)
    assert "units_empty" not in log
