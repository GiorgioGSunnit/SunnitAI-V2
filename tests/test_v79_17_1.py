"""v79.17.1, items 6-8 -- the questions' rows through the family gate with the frame's verified rito (family_gate
.context(rito=…) / decide / rito_refused, special_laws.rito_families, case_frame.verified_rito, interactive_flow
.gate_quesito_rows / quesiti_counters: item 6); an article with no schema element is context only (analysis_step
.has_elements / listed_norms / no_element_articles / line_tier / _verify_line, counter candidates_no_elements_excluded:
item 7); the replay's seconds per stage (scripts/replay_cases.py stage_seconds: item 8); the pretesa's binding always
has a set (applicability.pretesa_recall / recall_verify, judge; interactive_flow.binding_pretesa «recalled»: item 9) and an
article whose code is outside the frame's rito is a subordinate line only when tagged (analysis_step.outside_rito,
line_tier, _verify_line, counter analysis_line_outside_rito_rejected: item 9). Synthetic texts and replies only."""
import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import test_v79_10a as G  # noqa: E402
import test_v79_15_0_analysis as T  # noqa: E402
import test_v79_16_1 as X  # noqa: E402
import test_v79_17_0 as P  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
from src.rag import applicability as ap  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import family_gate  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from src.rag import special_laws  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_cases_171", ROOT / "scripts" / "replay_cases.py")
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


# --- 9. the pretesa's binding always has a set: the recall ---------------------------------------------------------------

NORMS_9 = [P.NORM, dict(P.NORM, article="1384 c.c.", rubrica="Riduzione della penale",
                        text="La penale può essere diminuita equamente dal giudice. Il giudice valuta l'interesse."),
           dict(P.NORM, article="1218 c.c.", rubrica="Responsabilità del debitore", text="Il debitore è tenuto al risarcimento.")]
PRETESA_9 = "domanda: «pagamento di una penale»"


def _verdicts(governs):
    """The fit's first reply: every norm «si», its governa as given (None: the key left out)."""
    items = []
    for n in NORMS_9:
        it = {"articolo": n["article"], "fit": "si", "perche": "x", "cita_norma": [f"A{ap.article_token(n['article'])}.1"],
              "cita_fatti": ["S1"], "condizione": "", "ambito": "fattispecie"}
        if governs.get(n["article"]) is not None:
            it["governa"] = governs[n["article"]]
        items.append(it)
    return json.dumps({"giudizi": items})


class _Fit:
    """The fit faked: the batch reply, then the recall's reply (a string, or an exception raised); every call kept."""

    def __init__(self, batch, recall=None):
        self.batch, self.recall, self.seen = batch, recall, []

    def __call__(self, messages, kwargs):
        self.seen.append(messages[1].content)
        if messages[1].content.startswith("PRETESA") and "ARTICOLI APPLICABILI" in messages[1].content:
            if isinstance(self.recall, Exception):
                raise self.recall
            return self.recall
        return self.batch


def test_recall_verify_keeps_the_ids_it_gave_and_drops_the_rest():
    got = ap.recall_verify('{"governa_pretesa": ["a1218cc", "A1382cc", "1382 c.c.", "A9999cc", "altro"]}', NORMS_9)
    assert got == (["1382 c.c.", "1218 c.c."], ["A9999cc", "altro"])          # the norms' order, each once
    assert ap.recall_verify('{"governa_pretesa": []}', NORMS_9) == ([], [])
    assert ap.recall_verify('["A1384cc"]', NORMS_9) == (["1384 c.c."], [])
    assert ap.recall_verify('{"x": 1}', NORMS_9) == ([], ["nessuna lista nella risposta"])
    human = ap.recall_messages(PRETESA_9, NORMS_9)[1].content
    assert human.startswith("PRETESA (ciò che è chiesto o contestato nel caso):\ndomanda: «pagamento di una penale»\n\n"
                            "ARTICOLI APPLICABILI (id: articolo):\n- A1382cc: 1382 c.c. (Effetti della clausola penale)\n")
    assert len(ap.recall_prompt_hash()) == 12 and ap.recall_prompt_hash() != ap.prompt_hash()


def test_the_recall_runs_when_no_verdict_governs_the_pretesa_and_its_ids_bind():
    model = _Fit(_verdicts({"1382 c.c.": [], "1384 c.c.": ["Q1"], "1218 c.c.": None}), '{"governa_pretesa": ["A1382cc", "A0cc"]}')
    res = ap.judge(NORMS_9, P.FACTS, "convenuto", call=model, pretesa=PRETESA_9, quesiti=["La penale è riducibile?"])
    c = res["counters"]
    assert len(model.seen) == 2 and "- A1218cc: 1218 c.c." in model.seen[1]      # the three fitted articles offered
    assert c["fit_governs_pretesa"] == 0 and c["fit_pretesa_recall"] == 1 and c["fit_pretesa_recall_empty"] == 0
    assert res["pretesa_recall"]["status"] == "ok" and res["pretesa_recall"]["articles"] == ["1382 c.c."]
    assert res["pretesa_recall"]["dropped"] == ["A0cc"] and res["pretesa_recall"]["ids_given"] == ["A1382cc", "A1384cc", "A1218cc"]
    by = res["by_article"]
    assert by["1382 c.c."]["governs"] == ["pretesa"] and by["1382 c.c."]["governs_recalled"] is True
    assert by["1384 c.c."]["governs"] == ["Q1"] and not by["1384 c.c."].get("governs_recalled")
    assert by["1218 c.c."]["governs"] is None
    # the binding reads the set from the verdicts, as the flow stores them
    state = {"applicability": {"governs_asked": True, "judgments": [
        {k: r.get(k) for k in ("article", "fit", "verified", "governs", "governs_recalled")} for r in res["judgments"]]},
        "block_counters": {}}
    b = flow.binding_pretesa(state)
    assert b["applied"] is True and b["principale"] == ["1382 c.c."] and b["recalled"] == ["1382 c.c."]
    assert b["quesiti"] == {"Q1": ["1384 c.c."]} and state["block_counters"]["binding_pretesa_applied"] == 1
    for name in ("fit_pretesa_recall", "fit_pretesa_recall_empty"):
        assert name in ap.COUNTERS and name in ap.FLOW_COUNTERS and name in R.FIT_COUNTERS


def test_an_empty_or_failed_recall_leaves_the_set_empty_and_is_counted():
    model = _Fit(_verdicts({"1382 c.c.": [], "1384 c.c.": [], "1218 c.c.": []}), '{"governa_pretesa": []}')
    res = ap.judge(NORMS_9, P.FACTS, "convenuto", call=model, pretesa=PRETESA_9)
    assert res["counters"]["fit_pretesa_recall"] == 1 and res["counters"]["fit_pretesa_recall_empty"] == 1
    assert res["pretesa_recall"]["articles"] == [] and all(r["governs"] == [] for r in res["judgments"])
    state = {"applicability": {"governs_asked": True, "judgments": res["judgments"]}, "block_counters": {}}
    assert flow.binding_pretesa(state)["applied"] is False and flow.binding_pretesa(state)["recalled"] == []
    model = _Fit(_verdicts({"1382 c.c.": []}), RuntimeError("down"))
    res = ap.judge(NORMS_9, P.FACTS, "convenuto", call=model, pretesa=PRETESA_9)
    assert res["pretesa_recall"]["status"] == "failed" and res["counters"]["fit_pretesa_recall"] == 1
    assert res["counters"]["fit_pretesa_recall_empty"] == 0 and res["by_article"]["1382 c.c."]["governs"] == []
    assert ap.recall_verify('{"governa_pretesa": ["A1382cc"]}', NORMS_9)[0] == ["1382 c.c."]


def test_no_recall_when_a_verdict_governs_the_pretesa_or_no_pretesa_is_given():
    model = _Fit(_verdicts({"1382 c.c.": ["pretesa"], "1384 c.c.": ["Q1"], "1218 c.c.": []}), '{"governa_pretesa": ["A1218cc"]}')
    res = ap.judge(NORMS_9, P.FACTS, "convenuto", call=model, pretesa=PRETESA_9, quesiti=["Q?"])
    assert len(model.seen) == 1 and res["pretesa_recall"]["status"] == "not_needed"
    assert res["counters"]["fit_pretesa_recall"] == 0 and res["counters"]["fit_governs_pretesa"] == 1
    model = _Fit(_verdicts({"1382 c.c.": []}), '{"governa_pretesa": ["A1382cc"]}')
    res = ap.judge(NORMS_9, P.FACTS, "convenuto", call=model, quesiti=["Q?"])          # quesiti alone: no pretesa
    assert len(model.seen) == 1 and res["counters"]["fit_pretesa_recall"] == 0
    # no fitted norm (every verdict unverified): nothing to recall, the call never made
    bad = json.dumps({"giudizi": [{"articolo": n["article"], "fit": "si", "perche": "x", "cita_norma": [], "cita_fatti": [],
                                   "condizione": "", "ambito": "fattispecie", "governa": []} for n in NORMS_9]})
    model = _Fit(bad, '{"governa_pretesa": ["A1382cc"]}')
    res = ap.judge(NORMS_9, P.FACTS, "convenuto", call=model, pretesa=PRETESA_9)
    assert len(model.seen) == 1 and res["pretesa_recall"]["status"] == "skipped" and res["counters"]["fit_pretesa_recall"] == 0


def test_the_replay_row_carries_the_binding_and_the_recall():
    state = {"applicability": {"pretesa_recall": {"status": "ok", "articles": ["1382 c.c."], "dropped": ["A0cc"]}},
             "binding_pretesa": {"principale": ["1382 c.c."], "recalled": ["1382 c.c."]}}
    row = R.measure("c", state, "a", 1.0)
    assert row["recall"] == {"status": "ok", "articles": ["1382 c.c."], "dropped": ["A0cc"], "principale": ["1382 c.c."],
                             "recalled": ["1382 c.c."]}
    assert "| binding / recall | principale [1382 c.c.]; recall ok, articles [1382 c.c.], recalled [1382 c.c.], dropped [A0cc] |" \
        in R.case_table(row)
    assert R._row_values(row)["recall.status"] == "ok" and R.measure("c", {}, "a", 1.0)["recall"] == {}


# --- 9. a code outside the frame's rito: a subordinate line only when tagged (the 1439 c.c. case) -------------------------

E1439 = ["raggiri usati da uno dei contraenti", "senza di essi l'altra parte non avrebbe contrattato"]
N1439 = T._norm("1439 c.c.", "Il dolo è causa di annullamento del contratto quando i raggiri usati da uno dei contraenti "
                "sono stati tali che, senza di essi, l'altra parte non avrebbe contrattato.", "Dolo", E1439, "D")


def _line_1439():
    return {"articolo": "1439 c.c.", "elemento_in_gioco": "A1439cc:E1", "tesi": "Dolo.", "cita_fatti": ["S3"],
            "cita_norma": ["A1439cc.1"]}


def _penal_with_1439(governs):
    norms = []
    for n in X._fitted(list(T.PENAL_NORMS) + [N1439], {a: ("si", True) for a in ("640 c.p.", "641 c.p.", "1439 c.c.")}):
        n = dict(n)
        if n["article"] in governs:
            n["governs"] = governs[n["article"]]
        norms.append(n)
    return A.make_input(X.NO_CHARGE, norms, T._sents(T.PENAL_SENTENCES, sistema=(9,)), T.PENAL_FRAME["quesiti"],
                        client="Paolo Sarti", client_roles=["imputato", "indagato"])


def test_outside_rito_reads_the_table_the_frame_and_the_code():
    inp = _penal_with_1439({})
    by = {n["article"]: n for n in inp["norms"]}
    assert A.frame_rito(inp["frame"]) == "penale"
    assert A.outside_rito(by["1439 c.c."], inp) == "penale" and A.outside_rito(by["640 c.p."], inp) == ""
    assert A.outside_rito_articles(inp) == ["1439 c.c."]
    civil = dict(inp, frame={"rito": {"valore": "civile"}})
    assert A.outside_rito(by["1439 c.c."], civil) == "" and A.outside_rito(by["640 c.p."], civil) == "civile"
    for frame in ({}, {"rito": {"valore": "amministrativo"}}, {"rito": {"valore": "sconosciuto"}}):
        assert A.outside_rito(by["1439 c.c."], dict(inp, frame=frame)) == ""
    assert A.outside_rito({"article": "21-nonies L. 241/1990"}, inp) == ""        # a special law: no code of the table
    assert A.outside_rito({"article": "24 Cost."}, inp) == ""
    assert A.outside_rito({"article": "1439 c.c.", "remedy": False}, dict(inp, frame={"rito": {"intake": "penale"}})) == "penale"


def test_a_code_outside_the_rito_is_in_no_leading_list_and_in_the_subordinate_only_when_tagged():
    inp = _penal_with_1439({})                                      # the fit not asked: no governs anywhere
    assert "1439 c.c." not in A.line_tier(inp)[1] and "1439 c.c." not in A.line_tier(inp, "linea_subordinata")[1]
    assert "1439 c.c." not in A.allowed_text(inp)
    assert "ARTICOLO: 1439 c.c." in A.build_messages(inp)[1].content     # listed: context, and the step reads it
    tagged = _penal_with_1439({"640 c.p.": ["pretesa"], "641 c.p.": [], "1439 c.c.": ["Q1"]})
    assert A.line_tier(tagged) == (A.TIER_SI, ["640 c.p."])
    assert A.line_tier(tagged, "linea_subordinata") == (A.TIER_SI, ["640 c.p.", "1439 c.c."])
    assert "- A1439cc (art. 1439 c.c.): A1439cc:E1, A1439cc:E2 [fit si verificato; governa: Q1] [solo linea subordinata]" \
        in A.allowed_text(tagged)
    untagged = _penal_with_1439({"640 c.p.": ["pretesa"], "641 c.p.": ["Q1"], "1439 c.c.": []})
    assert A.line_tier(untagged, "linea_subordinata")[1] == ["640 c.p.", "641 c.p."]
    assert A.prompt_hash() == "472423ea32d3"


def test_a_line_on_a_code_outside_the_rito_is_refused_unless_subordinate_and_tagged():
    # the 1439 c.c. case: proposed as the leading line, refused; the re-ask's 640 c.p. leads
    inp = _penal_with_1439({})
    model = T._Model(T._reply(_line_1439()),
                     T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"])))
    res = A.analyse(inp, call=model)
    assert res["rejected"][0]["why"] == "art. 1439 c.c.: codice fuori dal rito penale, mai linea principale, solo contesto: " \
                                        "ammessi art. 640 c.p., art. 641 c.p."
    assert res["counters"]["analysis_line_outside_rito_rejected"] == 1 and res["analysis"]["linea_principale"]["articolo"] == "640 c.p."
    # as the subordinate, untagged: refused; tagged with a quesito: kept
    untagged = _penal_with_1439({"640 c.p.": ["pretesa"], "641 c.p.": ["Q1"], "1439 c.c.": []})
    model = T._Model(T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"]),
                              subordinata=_line_1439()),
                     T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"])))
    res = A.analyse(untagged, call=model)
    why = [r["why"] for r in res["rejected"] if r["articolo"] == "1439 c.c."]
    assert why and why[0].startswith("art. 1439 c.c.: codice fuori dal rito penale e il fit non lo riferisce né alla pretesa "
                                     "né a un quesito: solo contesto, mai una linea: ammessi art. 640 c.p., art. 641 c.p.")
    assert res["counters"]["analysis_line_outside_rito_rejected"] == 1
    assert res["analysis"]["linea_subordinata"] is None or res["analysis"]["linea_subordinata"]["articolo"] != "1439 c.c."
    tagged = _penal_with_1439({"640 c.p.": ["pretesa"], "641 c.p.": [], "1439 c.c.": ["Q1"]})
    model = T._Model(T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"]),
                              subordinata=_line_1439()))
    res = A.analyse(tagged, call=model)
    assert res["analysis"]["linea_subordinata"]["articolo"] == "1439 c.c."
    assert res["counters"]["analysis_line_outside_rito_rejected"] == 0 and res["counters"]["analysis_reasked"] == 0
    assert "analysis_line_outside_rito_rejected" in A.COUNTERS and "analysis_line_outside_rito_rejected" in R.ANALYSIS_COUNTERS
    assert A.empty_counters()["analysis_line_outside_rito_rejected"] == 0
