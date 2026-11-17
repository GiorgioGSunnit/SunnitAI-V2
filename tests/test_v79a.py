"""v79 block A (79.0) -- measurement, on synthetic records only:

A1. the code's candidates reach the selection: a candidate the code enumerated without element ids (an eccezione, a
    remedy with nothing proven) is verifiable by construction -- before v79 it was «unverifiable», never main, and the
    posture's fallback led with no thesis eligible; the reply's valutazioni read in either shape (object or list) and
    with a decorated key («[T1]»)
A2. the thesis selection's counts (offered / parsed / valued / barred / unverifiable / minoritaria / chosen / fallback)
    in the plan log, the thesis log and the state; the nightly prints them per cell against the baseline
A3. the reply kept: the score, then a fixed tie-break (the parsed reply's digest), both logged; --deterministic (one
    sample at temperature 0) on nightly_sim.py and thesis_replay.py
"""

import importlib.util
import json
import os
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("THESIS_DETERMINISTIC", "")


def _script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _req(i, text, status):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


SENTS = [{"id": "S1", "text": "Alfa consegna una merce a Beta.", "source": "caso", "certainty": "accertato"}]


def _posture(family, role, stage):
    from src.rag import posture
    return {"rito": family, "family": family, "role": role, "stage": stage,
            "entry": posture.entry_for(family, role, stage)}


def _ctx(recs, posture):
    from src.rag import lawyer_method as M
    return M.prepare({"records": recs, "sentences": SENTS}, posture, [], "Alfa consegna una merce a Beta.", "Alfa", "",
                     list(recs), [], [], [], [], link_call=lambda *a: {}, case_id_="v79a")


def _no_review(monkeypatch):
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


def _custody_claim(*statuses):
    texts = ["la cosa è in custodia del convenuto", "il danno è cagionato dalla cosa"]
    return {"2051 c.c.": {"rubrica": "Danno cagionato da cosa in custodia", "side": "threat", "charged": 1,
                          "requisites": [_req(i, texts[i - 1], st) for i, st in enumerate(statuses, 1)]}}


def _all_valued(cands):
    return {"valutazioni": {c["id"]: {"pro": "p", "contro": "c", "realismo": "medio", "argomento": "a"} for c in cands
                            if c.get("argued") == "model"}}


# --- A1 ----------------------------------------------------------------------------------------------------------------

def test_a_code_candidate_without_ids_reaches_the_selection_and_the_fallback_does_not_lead(monkeypatch):
    from src.rag import lawyer_method as M
    _no_review(monkeypatch)
    ctx = _ctx(_custody_claim("provato", "provato"), _posture("civile", "convenuto", "primo_grado"))
    plan = M.method_plan(ctx, call=lambda m: [_all_valued(M.thesis_candidates(ctx))])
    sols, main, _ = M.selection(ctx, plan)
    eccezioni = [t for t in sols if t.get("kind") == "eccezione"]
    assert eccezioni and not any(t.get("unverifiable") for t in eccezioni)
    assert sols[main].get("candidate") and not sols[main].get("fallback")
    counts = ctx["selection_counts"]
    assert counts["offered"] == counts["parsed"] == len(ctx["candidates"])
    assert counts["unverifiable"] == 0 and counts["main_is_fallback"] is False and counts["chosen"] == sols[main]["candidate"]


def test_a_thesis_of_the_models_own_declaring_nothing_stays_unverifiable():
    from src.rag import lawyer_method as M
    plan = {"soluzioni": [{"tesi": "Tesi senza elementi", "realismo": "alto", "contesta": [], "poggia_su": []},
                          {"tesi": "Tesi del codice senza id", "realismo": "medio", "contesta": [], "poggia_su": [],
                           "candidate": "T2", "kind": "eccezione"}]}
    sols, main = M.coherent_solutions(plan, {}, {}, (), [], {})
    assert sols[0].get("unverifiable") and not sols[1].get("unverifiable") and main == 1


def test_the_attore_remedies_with_nothing_proven_are_eligible(monkeypatch):
    from src.rag import lawyer_method as M
    _no_review(monkeypatch)
    recs = {"1667 c.c.": {"rubrica": "Difformità e vizi dell'opera", "side": "weapon",
                          "requisites": [_req(1, "l'opera presenta vizi", "da provare"),
                                         _req(2, "i vizi sono denunciati entro sessanta giorni", "da provare")]}}
    ctx = _ctx(recs, _posture("civile", "attore", "pre_filing"))
    plan = M.method_plan(ctx, call=lambda m: [{}])
    sols, _, _ = M.selection(ctx, plan)
    remedies = [t for t in sols if t.get("kind") == "remedy"]
    assert remedies and all(M.eligible(t) for t in remedies)


@pytest.mark.parametrize("vals", [
    {"[T1]": {"realismo": "alto", "pro": "p"}},
    {"T1 - Contestare l'elemento": {"realismo": "alto", "pro": "p"}},
    {"t1": {"realismo": "alto", "pro": "p"}},
    [{"id": "T1", "realismo": "alto", "pro": "p"}],
    [{"tesi": "T1", "realismo": "alto", "pro": "p"}],
])
def test_the_valuations_are_read_in_every_shape_the_reply_gives(vals):
    from src.rag import lawyer_method as M
    cands = [{"id": "T1", "kind": "contest_open", "tesi": "L'elemento non è provato", "contesta": ["R1"],
              "poggia_su": [], "label": M.SUSTAINABLE, "argued": "model"}]
    sols = M.apply_valuations({"valutazioni": vals}, cands)
    assert sols[0]["realismo"] == "alto" and sols[0]["pro"] == "p"
    score, why = M.score_sample({"valutazioni": vals}, cands, {"R1": {"status": "da provare"}})
    assert score == 10.0 and not why


def test_a_key_naming_no_candidate_is_still_dropped():
    from src.rag import lawyer_method as M
    cands = [{"id": "T1", "kind": "contest_open", "tesi": "x", "contesta": ["R1"], "poggia_su": [],
              "label": M.SUSTAINABLE, "argued": "model"}]
    log = {}
    M.apply_valuations({"valutazioni": {"T1": {"realismo": "alto"}, "T9": {"realismo": "alto"},
                                        "Mia tesi": {"realismo": "alto"}}}, cands, log)
    assert log["dropped"] == ["T9", "MIA TESI"]


# --- A2 ----------------------------------------------------------------------------------------------------------------

def test_the_counts_are_logged_in_the_plan_log_and_the_thesis_log(monkeypatch):
    from src.rag import lawyer_method as M
    _no_review(monkeypatch)
    seen = []
    monkeypatch.setattr(M, "_log", lambda event, payload, ms=None: seen.append((event, payload)))
    recs = {"572 c.p.": {"rubrica": "Maltrattamenti contro familiari e conviventi", "side": "threat", "charged": 1,
                         "requisites": [_req(1, "la condotta è abituale", "provato"),
                                        _req(2, "la vittima è un convivente", "provato")]}}
    ctx = _ctx(recs, _posture("penale", "imputato", "primo_grado"))
    M.method_plan(ctx, call=lambda m: [_all_valued(M.thesis_candidates(ctx))])
    plan_log = next(p for e, p in seen if e == "plan")
    counts = plan_log["counts"]
    assert set(M.COUNT_KEYS) <= set(counts)
    assert counts["minoritaria"] == 2 and counts["barred"] == 0 and counts["chosen"] == "T3"     # v79.7: last
    assert counts["chosen_kind"] == "contenimento" and counts["main_is_fallback"] is False     # a candidate, not inserted
    entry = json.loads(Path(ctx["thesis_log"]).read_text())
    assert entry["counts"] == counts and entry["choice"]["by"] == "score"


def test_the_fallback_is_counted_when_the_selection_inserts_it():
    from src.rag import lawyer_method as M
    ctx = {"candidates": [{"id": "T1", "argued": "model"}], "records": {}, "assessment": {"all_integrated": True},
           "posture": _posture("penale", "imputato", "primo_grado")}
    plan = {"soluzioni": [{"tesi": "Contestare l'elemento", "realismo": "basso", "minoritaria": True,
                           "contesta": ["R1"], "poggia_su": [], "candidate": "T1"}]}
    counts = M.selection_counts(ctx, plan, {"valutazioni": {"T1": {"realismo": "basso"}}})
    assert counts["main_is_fallback"] is True and counts["chosen"] == "fallback" and counts["valued"] == 1


def test_the_flow_puts_the_counts_in_the_state():
    src = (HERE.parent / "src" / "rag" / "interactive_flow.py").read_text(encoding="utf-8")
    assert 'state["thesis_counts"] = ctx.get("selection_counts")' in src


def test_the_nightly_prints_the_counts_per_cell_against_the_baseline():
    nightly = _script("nightly_sim")
    now = [{"id": "c1", "checks": [], "thesis_counts": {"offered": 4, "parsed": 4, "valued": 2, "barred": 0,
                                                        "unverifiable": 0, "minoritaria": 2, "chosen": "T1",
                                                        "main_is_fallback": False}}]
    base = {"version": "78.1.1", "cells": [{"id": "c1", "thesis_counts": {"offered": 4, "parsed": 4, "valued": 2,
                                                                          "barred": 0, "unverifiable": 2,
                                                                          "minoritaria": 2, "chosen": "fallback",
                                                                          "main_is_fallback": True}}]}
    text = "\n".join(nightly.counts_lines(now, base))
    assert "| c1 | 4 | 4 | 2 | 0 | 2 → 0 | 2 | fallback → T1 | yes → no |" in text
    assert "(baseline 78.1.1)" in nightly.report(now, None, [], "79.0.0", [], base, "best-of-3")


# --- A3 ----------------------------------------------------------------------------------------------------------------

def test_the_choice_is_the_score_then_a_digest_never_the_order_of_arrival():
    from src.rag import lawyer_method as M
    a = {"parsed": {"valutazioni": {"T1": {"realismo": "alto"}}}, "score": 10.0}
    b = {"parsed": {"valutazioni": {"T1": {"realismo": "medio"}}}, "score": 10.0}
    worse = {"parsed": {}, "score": None}
    first, log1 = M.choose([a, b, worse])
    second, log2 = M.choose([worse, b, a])
    assert [a, b, worse][first] is [worse, b, a][second]
    assert log1["tied"] == [0, 1] and log1["tie_break"].startswith("sha1")
    better = {"parsed": {"valutazioni": {}}, "score": 11.0}
    assert M.choose([a, better])[0] == 1 and M.choose([])[0] is None


def test_deterministic_mode_is_one_sample_at_temperature_zero(monkeypatch):
    from src.rag import ai_chat, lawyer_method as M
    assert M.sampling() == (M.SAMPLES, M.SAMPLE_TEMPERATURE)
    monkeypatch.setenv("THESIS_DETERMINISTIC", "1")
    assert M.sampling() == (1, 0.0)
    seen = {}

    def call_chat_n(messages, n, max_tokens=None, temperature=0.7, thinking=None):
        seen.update(n=n, temperature=temperature)
        return ['{"valutazioni": {}}'] * n
    monkeypatch.setattr(ai_chat, "call_chat_n", call_chat_n)
    M._samples([], None)
    assert seen == {"n": 1, "temperature": 0.0}
    entry = json.loads(Path(M.thesis_log({"case_id": "d1"}, [], "{}", {}, {}, 1.0)).read_text())
    assert entry["n"] == 1 and entry["temperature"] == 0.0 and entry["deterministic"] is True


def test_the_nightly_flag_sets_deterministic_mode_for_its_run(monkeypatch, tmp_path):
    nightly = _script("nightly_sim")
    monkeypatch.setattr(nightly, "cells", lambda: [])
    out = tmp_path / "n.json"
    nightly.main(["--env", str(tmp_path / "none"), "--no-stages", "--deterministic", "--json-out", str(out)])
    assert os.environ["THESIS_DETERMINISTIC"] == "1"
    assert json.loads(out.read_text())["mode"].startswith("deterministic")


def test_the_replay_flag_sends_one_sample_at_temperature_zero(monkeypatch, tmp_path):
    replay = _script("thesis_replay")
    log = tmp_path / "thesis_c1.json"
    log.write_text(json.dumps({"case_id": "c1", "messages": [], "max_tokens": 100, "n": 3, "temperature": 0.7}))
    seen = {}

    def call(log_, base_url, model, api_key, thinking, max_tokens, n=1, temperature=0.0, effort=""):
        seen.update(n=n, temperature=temperature)
        return {"raw_reply": "{}", "parsed_best": {}, "samples": [], "best": 0, "finish_reason": "stop", "seconds": 0}
    monkeypatch.setattr(replay, "call", call)
    monkeypatch.setattr(replay, "check", lambda raw, log_: {"main": "", "main_is_fallback": False, "barred": [],
                                                           "theses": 0, "declared": 0, "unknown_ids": []})
    out = tmp_path / "ab.json"
    replay.main(["--log", str(log), "--base-url", "http://x/v1", "--model", "m", "--deterministic", "--out", str(out)])
    assert seen == {"n": 1, "temperature": 0.0} and json.loads(out.read_text())["replay"]["deterministic"] is True
