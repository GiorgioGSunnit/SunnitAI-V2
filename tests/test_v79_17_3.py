"""v79.17.3 -- the governs-pretesa set is the union of the tags, the recall's answer and every substantive article the fit
verified «si» (the «dipende» ones with no «si»); the honesty line only on an empty union; the pretesa's own article in the
set when its verdict is not a verified «no» and it has elements, a rejected frame-named article dropped from the pretesa
(interactive_flow.binding_pretesa / pretesa_recall_step, applicability.apply_recall «has_elements»). Synthetic only."""
import importlib.util
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(__file__))
from src.rag import applicability as ap  # noqa: E402
from src.rag import case_frame as cf  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("replay_cases_173", ROOT / "scripts" / "replay_cases.py")
R = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(R)


def _j(article, fit="si", verified=True, governs=None, substantive=True, **more):
    return {"article": article, "fit": fit, "verified": verified, "governs": governs, "substantive": substantive, **more}


def _state(judgments, frame_article=None, pretesa="domanda: «x»"):
    state = {"applicability": {"governs_asked": True, "pretesa": pretesa, "judgments": judgments}, "block_counters": {}}
    if frame_article is not None:
        state["case_frame"] = {"frame": {"addebito_o_domanda": {"tipo": "domanda", "contenuto": "x", "articolo": frame_article,
                                                                "frase": "«x»"}}, "verified": ["addebito_o_domanda"]}
    return state


def test_the_set_is_the_union_of_the_tags_the_recall_and_the_fits_verified_si(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    state = _state([_j("1385 c.c.", governs=["pretesa"]),                      # tagged
                    _j("1454 c.c.", governs=["Q1"]),                           # verified si: the fit judged it -- in
                    _j("1225 c.c.", fit="dipende", governs=["Q2"]),            # dipende: out while a «si» exists
                    _j("1453 c.c.", governs=[], substantive=False),            # remedy: never
                    _j("2932 c.c.", verified=False, governs=None, governs_recalled=True),   # recalled, unverified: in
                    _j("1386 c.c.", governs=[])])                              # verified si, context tag: in (fit_si)
    state["applicability"]["judgments"][4]["governs"] = ["pretesa"]
    b = flow.binding_pretesa(state)
    # v79.18.0 (A1): the set has tiers -- the verified «si», then the verified «dipende» (1225 c.c. is in now), then
    # the unverified «si» / «dipende» (2932 c.c., recalled); the order is the tiers'
    assert b["principale"] == ["1385 c.c.", "1454 c.c.", "1386 c.c.", "1225 c.c.", "2932 c.c."]
    assert b["reasons"] == {"1385 c.c.": ["tag", "fit_si"], "1454 c.c.": ["fit_si"], "1386 c.c.": ["fit_si"],
                            "1225 c.c.": ["fit_dipende"], "2932 c.c.": ["recall", "fit_non_verificato"]}
    assert b["tiers"] == {"1385 c.c.": "si_verificato", "1454 c.c.": "si_verificato", "1386 c.c.": "si_verificato",
                          "1225 c.c.": "dipende_verificato", "2932 c.c.": "non_verificato"}
    assert b["fit_si"] == ["1385 c.c.", "1454 c.c.", "1386 c.c."] and b["fit_dipende"] == ["1225 c.c."]
    assert b["fit_non_verificato"] == ["2932 c.c."]
    assert b["recalled"] == ["2932 c.c."] and b["context_only"] == ["1453 c.c."] and b["applied"] is True and b["enforced"] is True
    assert b["quesiti"] == {"Q1": ["1454 c.c."], "Q2": ["1225 c.c."]}
    assert state["block_counters"]["binding_set_size"] == 5 and state["block_counters"]["binding_empty_honesty"] == 0
    assert state["block_counters"]["binding_set_unverified"] == 1
    # no verified «si»: the verified «dipende» ones lead the set, the unverified ones follow
    state = _state([_j("1225 c.c.", fit="dipende", governs=["Q2"]), _j("1453 c.c.", fit="dipende", governs=[], substantive=False),
                    _j("1386 c.c.", verified=False, governs=None)])
    b = flow.binding_pretesa(state)
    assert b["principale"] == ["1225 c.c.", "1386 c.c."] and b["reasons"] == {"1225 c.c.": ["fit_dipende"],
                                                                              "1386 c.c.": ["fit_non_verificato"]}
    # no substantive article with any «si» / «dipende» verdict (an unverified «no» is none), no tag, no recall: the union
    # is empty -- the honesty counter
    state = _state([_j("1453 c.c.", governs=["pretesa"], substantive=False), _j("1386 c.c.", fit="no", verified=False, governs=None),
                    _j("2946 c.c.", fit="no", governs=[])])
    b = flow.binding_pretesa(state)
    assert b["principale"] == [] and b["nonsubstantive_tagged"] == ["1453 c.c."]
    assert state["block_counters"]["binding_empty_honesty"] == 1 and state["block_counters"]["binding_set_size"] == 0
    # v79.18.0 (A1): an unverified «si» alone is a set -- untrusted, present: no honesty line
    state = _state([_j("1453 c.c.", governs=["pretesa"], substantive=False), _j("1386 c.c.", verified=False, governs=None)])
    b = flow.binding_pretesa(state)
    assert b["principale"] == ["1386 c.c."] and b["tiers"] == {"1386 c.c.": "non_verificato"}
    assert state["block_counters"]["binding_empty_honesty"] == 0


def test_the_pretesas_own_article_enters_the_set_unless_a_verified_no_or_no_element(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    # v79.18.2 (item 3): an unverified verdict on the pretesa's article is no «pretesa» reason -- the article stays in
    # the set through the unverified tier alone (A1), never through «≠ no»
    state = _state([_j("1385 c.c.", verified=False, governs=None, substantive=True, has_elements=True),
                    _j("1454 c.c.", governs=["Q1"])], frame_article="1385 c.c.")
    b = flow.binding_pretesa(state)
    assert b["pretesa_article"] == "1385 c.c." and b["reasons"]["1385 c.c."] == ["fit_non_verificato"]
    assert b["principale"] == ["1454 c.c.", "1385 c.c."]          # v79.18.0 (A1): the verified «si» before the unverified
    # a verified «si» / «dipende» on it: the «pretesa» reason
    state = _state([_j("1385 c.c.", governs=[], substantive=True, has_elements=True)], frame_article="1385 c.c.")
    assert flow.binding_pretesa(state)["reasons"]["1385 c.c."] == ["pretesa", "fit_si"]
    state = _state([_j("1385 c.c.", fit="dipende", governs=[], substantive=True, has_elements=True)], frame_article="1385 c.c.")
    assert flow.binding_pretesa(state)["reasons"]["1385 c.c."] == ["pretesa", "fit_dipende"]
    # a verified «no»: out
    state = _state([_j("640 c.p.", fit="no", governs=[], has_elements=True), _j("590 c.p.", governs=["pretesa"])],
                   frame_article="640 c.p.")
    b = flow.binding_pretesa(state)
    assert b["principale"] == ["590 c.p."] and b["pretesa_article"] == "640 c.p." and "640 c.p." not in b["reasons"]
    # no element: out (the has_elements flag, else the substantive one); unverified with elements: out too (item 3)
    state = _state([_j("645 c.p.c.", verified=False, governs=None, substantive=False, has_elements=False)], frame_article="645 c.p.c.")
    assert flow.binding_pretesa(state)["principale"] == []
    state = _state([_j("645 c.p.c.", verified=False, governs=None, substantive=False, has_elements=True)], frame_article="645 c.p.c.")
    assert flow.binding_pretesa(state)["principale"] == []
    state = _state([_j("645 c.p.c.", governs=[], substantive=False, has_elements=True)], frame_article="645 c.p.c.")
    assert flow.binding_pretesa(state)["principale"] == ["645 c.p.c."]         # verified «si» + elements
    # the frame off: no pretesa article read
    monkeypatch.setenv("CASE_FRAME", "false")
    state = _state([_j("1385 c.c.", verified=False, governs=None)], frame_article="1385 c.c.")
    assert flow.binding_pretesa(state)["pretesa_article"] == ""


def test_a_frame_named_article_the_fit_rejects_leaves_the_pretesa_before_the_recall(monkeypatch):
    monkeypatch.setenv("CASE_FRAME", "true")
    seen = []

    def call(messages, kwargs):
        seen.append(messages[1].content)
        return json.dumps({"governa_pretesa": []})
    monkeypatch.setattr(ap, "_default_call", call)
    state = {"records": {"590 c.p.": {"rubrica": "Lesioni personali colpose", "requisites": [{"id": "R1", "text": "lesione"}]},
                         "640 c.p.": {"rubrica": "Truffa", "requisites": [{"id": "R1", "text": "artifizi"}]}},
             "case_frame": {"frame": {"addebito_o_domanda": {"tipo": "accusa", "contenuto": "lesioni personali colpose",
                                                             "articolo": "640 c.p.", "contro": "Matteo Ruggeri", "frase": "«x»"}},
                            "verified": ["addebito_o_domanda"]},
             "applicability": {"governs_asked": True, "pretesa": "accusa: «lesioni personali colpose» (art. 640 c.p.) contro Matteo Ruggeri",
                               "rubriche": {}, "contractual": False, "pretesa_recall": {},
                               "judgments": [_j("590 c.p.", governs=["pretesa"]),
                                             _j("640 c.p.", fit="no", governs=[], perche="nessun artifizio")]},
             "block_counters": {}, "timings": {}}
    flow.pretesa_recall_step(state)
    field = state["case_frame"]["frame"]["addebito_o_domanda"]
    assert field["articolo"] is None and field["articolo_scartato"] == "640 c.p."
    assert state["applicability"]["pretesa"] == "accusa: «lesioni personali colpose» contro Matteo Ruggeri"
    assert seen and seen[0].startswith("PRETESA (ciò che è chiesto o contestato nel caso):\naccusa: «lesioni personali colpose» contro Matteo Ruggeri\n")
    assert "- A640cp" not in seen[0] and "- A590cp: 590 c.p." in seen[0]
    c = state["block_counters"]
    assert c["frame_pretesa_article_dropped"] == 1 and c["fit_pretesa_recall"] == 1 and c["fit_pretesa_recall_empty"] == 1
    b = state["binding_pretesa"]
    assert b["principale"] == ["590 c.p."] and b["reasons"] == {"590 c.p.": ["tag", "fit_si"]} and b["pretesa_article"] == ""
    assert cf.charge_label(state["case_frame"]) == ""
    # the frame's article the fit kept («si» verified): stays, and is in the set as the pretesa's own
    state2 = {"records": {"640 c.p.": {"rubrica": "Truffa", "requisites": [{"id": "R1", "text": "artifizi"}]}},
              "case_frame": {"frame": {"addebito_o_domanda": {"tipo": "accusa", "contenuto": "truffa", "articolo": "640 c.p.", "frase": "«x»"}},
                             "verified": ["addebito_o_domanda"]},
              "applicability": {"governs_asked": True, "pretesa": "accusa: «truffa» (art. 640 c.p.)", "rubriche": {}, "contractual": False,
                                "pretesa_recall": {}, "judgments": [_j("640 c.p.", governs=[])]},
              "block_counters": {}, "timings": {}}
    flow.pretesa_recall_step(state2)
    assert state2["case_frame"]["frame"]["addebito_o_domanda"]["articolo"] == "640 c.p."
    assert state2["block_counters"]["frame_pretesa_article_dropped"] == 0
    assert state2["binding_pretesa"]["reasons"] == {"640 c.p.": ["pretesa", "fit_si"]}
    assert "frame_pretesa_article_dropped" in R.FRAME_COUNTERS


def test_apply_recall_marks_has_elements_and_the_replay_row_carries_the_reasons():
    res = {"judgments": [{"article": "1385 c.c.", "fit": "si", "verified": True, "governs": []},
                         {"article": "1218 c.c.", "fit": "si", "verified": True, "governs": []}], "counters": {}}
    norms = [{"article": "1385 c.c.", "elements": [{"id": "R1", "testo": "x"}]}, {"article": "1218 c.c.", "elements": [{"id": "R1", "testo": "y"}]}]
    ap.apply_recall(res, norms, "", call=lambda m, k: "{}")
    j = {x["article"]: x for x in res["judgments"]}
    assert j["1385 c.c."]["has_elements"] is True and j["1385 c.c."]["substantive"] is True
    assert j["1218 c.c."]["has_elements"] is True and j["1218 c.c."]["substantive"] is False
    state = {"applicability": {"pretesa_recall": {"status": "ok", "articles": []}},
             "binding_pretesa": {"principale": ["1385 c.c.", "1454 c.c."], "recalled": [],
                                 "reasons": {"1385 c.c.": ["tag", "fit_si"], "1454 c.c.": ["fit_si"]}, "pretesa_article": "1385 c.c."}}
    row = R.measure("c", state, "a", 1.0)
    assert row["recall"]["reasons"] == {"1385 c.c.": ["tag", "fit_si"], "1454 c.c.": ["fit_si"]}
    assert ", reasons 1385 c.c.: tag+fit_si; 1454 c.c.: fit_si, pretesa article 1385 c.c. |" in R.case_table(row)
