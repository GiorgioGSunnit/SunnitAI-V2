"""v79.17.2 -- the governs-pretesa set holds substantive articles only and the recall always runs over them after the
records are decomposed (applicability.substantive / apply_recall, interactive_flow.pretesa_recall_step /
binding_pretesa «enforced», counters binding_set_size, binding_empty_honesty: item 1); the leading line must be in the
set, always -- nothing unjudged stands in for it, the fallbacks pick inside it, an empty set prints the honesty line and
«da determinare» (analysis_step.governs_sets / line_tier / _verify_line / finish / summary_text, counters
analysis_line_outside_set_rejected, analysis_binding_empty_honesty: item 2). Synthetic texts and replies only."""
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
import test_v79_15_0_analysis as T  # noqa: E402
import test_v79_16_1 as X  # noqa: E402
import test_v79_17_1 as N  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
from src.rag import applicability as ap  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_cases_172", ROOT / "scripts" / "replay_cases.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)

EL = [{"id": "R1", "testo": "un elemento"}]


# --- 1. the set: substantive articles only; the recall after the records are decomposed ------------------------------------

def test_substantive_needs_an_element_and_refuses_the_remedy_set_and_the_procedural_codes():
    assert ap.substantive({"article": "1385 c.c.", "elements": EL}) is True
    assert ap.substantive({"article": "1385 c.c.", "elements": []}) is False
    assert ap.substantive({"article": "1385 c.c."}) is False
    assert ap.substantive({"article": "1218 c.c.", "elements": EL}) is False          # the remedy set
    assert ap.substantive({"article": "2043 c.c.", "elements": EL}) is True
    assert ap.substantive({"article": "2043 c.c.", "elements": EL}, contractual=True) is False   # when_contractual
    assert ap.substantive({"article": "645 c.p.c.", "elements": EL}) is False          # a procedural code
    assert ap.substantive({"article": "74 c.p.p.", "elements": EL}) is False
    assert ap.substantive({"article": "29 c.p.a.", "elements": EL}) is False
    assert ap.substantive({"article": "640 c.p.", "elements": EL}) is True
    assert ap.procedural_codes() == ["cpc", "cpp", "cpa"]


def _judgment(article, fit="si", verified=True, governs=None, **more):
    return {"article": article, "fit": fit, "verified": verified, "governs": governs, **more}


def test_the_binding_set_excludes_a_tag_on_a_non_substantive_article_and_includes_the_recall():
    state = {"applicability": {"governs_asked": True, "pretesa": "domanda: «x»", "judgments": [
        _judgment("1385 c.c.", governs=["pretesa"], substantive=True),
        _judgment("1218 c.c.", governs=["pretesa", "Q1"], substantive=False),            # a remedy, tagged: out
        _judgment("1225 c.c.", fit="dipende", governs=["Q2"], substantive=True),
        _judgment("1453 c.c.", verified=False, governs=None, substantive=True, governs_recalled=True),  # recalled,
        _judgment("1454 c.c.", governs=[], substantive=True)]}, "block_counters": {}}             # unverified: in
    state["applicability"]["judgments"][3]["governs"] = ["pretesa"]
    b = flow.binding_pretesa(state)        # v79.17.3 (item 1): 1454 c.c. (verified «si», substantive) joins as fit_si
    # v79.18.0 (A1): by tier -- the verified «si» first, the verified «dipende» (1225 c.c.) next, the unverified last
    assert b["principale"] == ["1385 c.c.", "1454 c.c.", "1225 c.c.", "1453 c.c."] and b["recalled"] == ["1453 c.c."]
    assert b["tiers"] == {"1385 c.c.": "si_verificato", "1454 c.c.": "si_verificato", "1225 c.c.": "dipende_verificato",
                          "1453 c.c.": "non_verificato"}
    assert b["nonsubstantive_tagged"] == ["1218 c.c."] and b["quesiti"] == {"Q1": ["1218 c.c."], "Q2": ["1225 c.c."]}
    assert b["context_only"] == [] and b["enforced"] is True and b["applied"] is True
    c = state["block_counters"]
    assert c["binding_set_size"] == 4 and c["binding_empty_honesty"] == 0 and c["binding_pretesa_applied"] == 1
    assert c["binding_set_unverified"] == 1
    # every tag on a non-substantive article, the recall empty: the set is empty, enforced, the honesty counter 1
    state = {"applicability": {"governs_asked": True, "pretesa": "domanda: «x»", "judgments": [
        _judgment("1218 c.c.", governs=["pretesa"], substantive=False)]}, "block_counters": {}}
    b = flow.binding_pretesa(state)
    assert b["principale"] == [] and b["enforced"] is True and b["applied"] is False
    assert state["block_counters"]["binding_set_size"] == 0 and state["block_counters"]["binding_empty_honesty"] == 1
    # no pretesa given: nothing enforced, no honesty
    state = {"applicability": {"governs_asked": True, "pretesa": "", "judgments": []}, "block_counters": {}}
    assert flow.binding_pretesa(state)["enforced"] is False and state["block_counters"]["binding_empty_honesty"] == 0


def test_the_recall_step_runs_on_the_records_as_decomposed_and_recomputes_the_binding(monkeypatch):
    seen = []

    def call(messages, kwargs):
        seen.append(messages[1].content)
        return json.dumps({"governa_pretesa": ["A1667cc"]})
    monkeypatch.setattr(ap, "_default_call", call)
    state = {"records": {"1667 c.c.": {"rubrica": "Difformità e vizi dell'opera", "requisites": [{"id": "R1", "text": "vizi"}]},
                         "1453 c.c.": {"rubrica": "Risolubilità", "requisites": [{"id": "R1", "text": "x"}]},
                         "1176 c.c.": {"rubrica": "Diligenza", "requisites": []}},
             "applicability": {"governs_asked": True, "pretesa": "domanda: «garanzia per i vizi»",
                               "rubriche": {"1667 c.c.": "Difformità e vizi dell'opera"}, "contractual": True,
                               "pretesa_recall": {"status": "not_needed"},
                               "judgments": [_judgment("1667 c.c.", verified=False, governs=None),      # the named article:
                                             _judgment("1453 c.c.", governs=[]),                       # unverified at the fit
                                             _judgment("1176 c.c.", governs=["Q1"]),
                                             _judgment("1218 c.c.", fit="no", governs=[])]},
             "block_counters": {"fit_seconds": 1.0}, "timings": {"applicability_s": 1.0}}
    recall = flow.pretesa_recall_step(state)
    assert len(seen) == 1 and "- A1667cc: 1667 c.c. (Difformità e vizi dell'opera)" in seen[0]
    assert "1453 c.c." not in seen[0] and "1176 c.c." not in seen[0] and "1218 c.c." not in seen[0]   # remedy / no element / no
    assert recall["status"] == "ok" and recall["articles"] == ["1667 c.c."]
    j = {x["article"]: x for x in state["applicability"]["judgments"]}
    assert j["1667 c.c."]["governs"] == ["pretesa"] and j["1667 c.c."]["governs_recalled"] is True
    assert j["1667 c.c."]["substantive"] is True and j["1453 c.c."]["substantive"] is False and j["1176 c.c."]["substantive"] is False
    assert state["binding_pretesa"]["principale"] == ["1667 c.c."] and state["binding_pretesa"]["recalled"] == ["1667 c.c."]
    c = state["block_counters"]
    assert c["fit_pretesa_recall"] == 1 and c["fit_pretesa_recall_empty"] == 0 and c["binding_set_size"] == 1
    assert c["fit_seconds"] >= 1.0 and state["timings"]["applicability_s"] == c["fit_seconds"]
    assert state["applicability"]["pretesa_recall"]["articles"] == ["1667 c.c."]
    assert flow.pretesa_recall_step({"applicability": {"pretesa": ""}}) == {} and len(seen) == 1
    for name in ("binding_set_size", "binding_empty_honesty"):
        assert name in R.FRAME_COUNTERS


def test_the_judge_leaves_the_recall_to_the_caller_when_told():
    model = N._Fit(N._verdicts({"1382 c.c.": [], "1384 c.c.": [], "1218 c.c.": []}), '{"governa_pretesa": ["A1382cc"]}')
    res = ap.judge(N.NORMS_9, N.P.FACTS, "convenuto", call=model, pretesa=N.PRETESA_9, recall=False)
    assert len(model.seen) == 1 and res["pretesa_recall"]["status"] == "not_needed" and res["counters"]["fit_pretesa_recall"] == 0
    recall = ap.apply_recall(res, N.NORMS_9, N.PRETESA_9, call=model)
    assert len(model.seen) == 2 and recall["articles"] == ["1382 c.c."] and res["counters"]["fit_pretesa_recall"] == 1
    assert res["by_article"]["1382 c.c."]["governs"] == ["pretesa"] and res["by_article"]["1218 c.c."]["substantive"] is False


# --- 2. the leading line in the set, always ---------------------------------------------------------------------------------

def _penal(governs, binding, extra=()):
    norms = []
    for n in X._fitted(list(T.PENAL_NORMS) + list(extra), {a: ("si", True) for a in governs}):
        n = dict(n)
        if n["article"] in governs and governs[n["article"]] is not None:
            n["governs"] = governs[n["article"]]
        norms.append(n)
    return A.make_input(X.NO_CHARGE, norms, T._sents(T.PENAL_SENTENCES, sistema=(9,)), T.PENAL_FRAME["quesiti"],
                        client="Paolo Sarti", client_roles=["imputato", "indagato"], binding=binding)


ENFORCED_640 = {"enforced": True, "principale": ["640 c.p."]}


def test_the_sets_come_from_the_flows_binding_and_an_unjudged_norm_no_longer_stands_in():
    inp = _penal({"640 c.p.": ["pretesa"], "641 c.p.": ["Q1"]}, ENFORCED_640)
    assert inp["binding"] == {"enforced": True, "pretesa": ["640 c.p."], "tiers": {}}
    gs = A.governs_sets(inp)
    assert gs == {"applied": True, "pretesa": ["640 c.p."], "quesiti": ["641 c.p."], "context": [], "enforced": True,
                  "tiers": {}}
    assert A.line_tier(inp) == (A.TIER_SI, ["640 c.p."]) and A.line_tier(inp, "linea_subordinata")[1] == ["640 c.p.", "641 c.p."]
    # 641 c.p. never judged (no fit, no governs): with the binding enforced it is in no list; without, it stays (79.17.0)
    inp = _penal({"640 c.p.": ["pretesa"], "641 c.p.": None}, ENFORCED_640)
    assert A.line_tier(inp, "linea_subordinata")[1] == ["640 c.p."]
    loose = _penal({"640 c.p.": ["pretesa"], "641 c.p.": None}, None)
    assert A.line_tier(loose, "linea_subordinata")[1] == ["640 c.p.", "641 c.p."] and A.governs_sets(loose)["enforced"] is False
    # the flow's set names an article this input does not hold (the 1385 c.c. case): the set among the norms is empty
    inp = _penal({"640 c.p.": [], "641 c.p.": ["Q1"]}, {"enforced": True, "principale": ["1385 c.c."]})
    gs = A.governs_sets(inp)
    assert gs["pretesa"] == [] and gs["applied"] is True and gs["enforced"] is True
    assert A.line_tier(inp)[1] == [] and A.line_tier(inp, "linea_subordinata")[1] == ["641 c.p."]
    assert A.prompt_hash() == "472423ea32d3"


def test_a_leading_line_outside_the_set_is_refused_and_the_subordinate_may_be_a_quesitos():
    inp = _penal({"640 c.p.": ["pretesa"], "641 c.p.": ["Q1"]}, ENFORCED_640)
    model = T._Model(T._reply(T._line("641 c.p.", T.E641[0], "Tesi.", ["S2"], ["A641cp.1"])),
                     T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"]),
                              subordinata=T._line("641 c.p.", T.E641[0], "Sub.", ["S2"], ["A641cp.1"])))
    res = A.analyse(inp, call=model)
    assert res["rejected"][0]["why"].startswith("art. 641 c.p.: governa solo un quesito, ammesso come linea subordinata")
    assert res["analysis"]["linea_principale"]["articolo"] == "640 c.p." and res["analysis"]["linea_subordinata"]["articolo"] == "641 c.p."
    assert res["counters"]["analysis_line_outside_set_rejected"] == 0 and res["counters"]["analysis_line_quesito_only"] == 1
    # an unjudged norm as the leading line: outside the set, refused (no tier stands in for the set any more)
    inp = _penal({"640 c.p.": ["pretesa"], "641 c.p.": None}, ENFORCED_640)
    model = T._Model(T._reply(T._line("641 c.p.", T.E641[0], "Tesi.", ["S2"], ["A641cp.1"])),
                     T._reply(T._line("640 c.p.", T.E640[0], "Tesi.", ["S3"], ["A640cp.1"]),
                              subordinata=T._line("641 c.p.", T.E641[0], "Sub.", ["S2"], ["A641cp.1"])))
    res = A.analyse(inp, call=model)
    whys = [r["why"] for r in res["rejected"]]
    assert whys[0] == "art. 641 c.p.: fuori dall'insieme che governa la pretesa: ammessi art. 640 c.p."
    assert whys[1] == "art. 641 c.p.: fuori dall'insieme che governa la pretesa e dai quesiti: ammessi art. 640 c.p."
    assert res["counters"]["analysis_line_outside_set_rejected"] == 2
    assert res["analysis"]["linea_principale"]["articolo"] == "640 c.p." and res["analysis"]["linea_subordinata"] is None
    assert "analysis_line_outside_set_rejected" in A.COUNTERS and "analysis_line_outside_set_rejected" in R.ANALYSIS_COUNTERS


def test_an_empty_set_prints_the_honesty_line_and_never_a_free_pick():
    # the set names an article this input does not hold: empty among the norms -- the model's 640 c.p. is refused, the
    # frame's charge (640 c.p.) is no pick, the rubrica fallback is no pick: «da determinare» and the honesty line
    binding = {"enforced": True, "principale": ["1385 c.c."]}
    norms = X._fitted(list(T.PENAL_NORMS), {"640 c.p.": ("si", True), "641 c.p.": ("si", True)})
    inp = A.make_input(T.PENAL_FRAME, norms, T._sents(T.PENAL_SENTENCES, sistema=(9,)), T.PENAL_FRAME["quesiti"],
                       client="Paolo Sarti", client_roles=["imputato"], binding=binding)
    assert inp["charge"] == "640 c.p." and inp["issues"]
    model = T._Model(T.GOOD_PENAL, T.GOOD_PENAL)
    res = A.analyse(inp, call=model)
    assert res["rejected"][0]["why"] == "art. 640 c.p.: Il corpus non copre la pretesa: nessun articolo applicabile la governa: nessuna linea principale"
    a = res["analysis"]
    assert a["linea_principale"]["stato"] == A.UNDETERMINED and a["linea_principale"]["articolo"] is None
    assert a["honesty"] == A.HONESTY_LINE and a["fallback_skipped_reason"] == A.SKIP_BINDING_EMPTY
    c = res["counters"]
    assert c["analysis_line_undetermined"] == 1 and c["analysis_binding_empty_honesty"] == 1
    assert c["analysis_frame_charge_missing"] == 0 and c["analysis_line_fallback_rubrica"] == 0
    assert c["analysis_line_outside_set_rejected"] == 2 and c["analysis_fallback_skipped_reason"] == "binding_empty"
    v = A.view(res, inp)
    A.attach_object(v, pretesa={"tipo": "accusa"}, pretesa_line="Pretesa: accusa «truffa».")
    assert A.summary_text(v) == ("Pretesa: accusa «truffa».\n\nIl corpus non copre la pretesa: nessun articolo applicabile la governa."
                                 "\n\nLinea principale: da determinare (linea non verificata).")
    assert "analysis_binding_empty_honesty" in A.COUNTERS and "analysis_binding_empty_honesty" in R.ANALYSIS_COUNTERS
    # a failed step with an empty enforced set: no fallback either
    res = A.analyse(inp, call=T._Model(RuntimeError("down")))
    assert res["analysis"]["linea_principale"]["stato"] == A.UNDETERMINED and res["analysis"]["honesty"] == A.HONESTY_LINE
    assert res["counters"]["analysis_frame_charge_missing"] == 0


def test_the_fallbacks_pick_inside_the_set_only():
    # the set holds 641 c.p.; the frame's charge is 640 c.p. (outside): on no verified line the frame's charge is no
    # pick; the rubrica fallback reads the set only
    binding = {"enforced": True, "principale": ["641 c.p."]}
    norms = X._fitted(list(T.PENAL_NORMS), {"640 c.p.": ("si", True), "641 c.p.": ("si", True)})
    inp = A.make_input(T.PENAL_FRAME, norms, T._sents(T.PENAL_SENTENCES, sistema=(9,)), T.PENAL_FRAME["quesiti"],
                       client="Paolo Sarti", client_roles=["imputato"], binding=binding)
    res = A.analyse(inp, call=T._Model(RuntimeError("down")))
    a = res["analysis"]
    assert res["counters"]["analysis_frame_charge_missing"] == 0 and a["linea_principale"]["articolo"] != "640 c.p."
    # v79.18.0 (A2): no rubrica match -> the set's best-tier article leads, every element «da_provare»
    assert a["honesty"] == "" and a["fallback_skipped_reason"] == A.SKIP_SET_LINE
    assert a["linea_principale"]["articolo"] == "641 c.p." and a["linea_principale"]["stato"] == A.FALLBACK_SET
    assert res["counters"]["analysis_line_fallback_set"] == 1 and res["counters"]["analysis_line_undetermined"] == 0
    assert A.rubrica_fallback(inp, ["641 c.p."]) is None or A.rubrica_fallback(inp, ["641 c.p."])["articolo"] == "641 c.p."
    assert (A.rubrica_fallback(inp) or {}).get("articolo") in (None, "640 c.p.")      # unconstrained: the frame's truffa


def test_the_replay_row_carries_the_honesty_line_and_the_new_counters():
    state = {"analysis_step": {"status": "ok", "analysis": {"linea_principale": A.undetermined_line(), "honesty": A.HONESTY_LINE},
                               "counters": {}, "rejected": []},
             "block_counters": {"binding_set_size": 0, "binding_empty_honesty": 1, "analysis_binding_empty_honesty": 1,
                                "analysis_line_outside_set_rejected": 2}}
    row = R.measure("c", state, "a", 1.0)
    assert row["analysis"]["honesty"] == A.HONESTY_LINE and row["frame_counters"]["binding_empty_honesty"] == 1
    assert row["analysis_counters"]["analysis_line_outside_set_rejected"] == 2
    assert f"| honesty line | {A.HONESTY_LINE} |" in R.case_table(row)
    assert R._row_values(row)["analysis.honesty"] == A.HONESTY_LINE
