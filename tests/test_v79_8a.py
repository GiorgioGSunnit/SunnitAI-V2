"""v79.8 block A -- latency, on synthetic rows and states only:

A1. the matter gate before the grading drops every retrieval hit outside the case's matters (a vector hit as a keyword
    one), never a multi-signal row, a whole-name first-read hit or a kept article
A2. at most GRADE_CAP articles graded: the kept ones (named, anchors, the first read's) always, then by retrieval rank
A3. one thesis sample (the scorer cannot tell the samples apart)
A4. the clients' objections calls run concurrently
A5. the counters: graded_articles, samples_used, llm_calls, total_s, the funnel -- in the timings line and the nightly
"""

import importlib.util
import threading
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


def _row(n, code, rubrica="", rank=None, **tags):
    row = {"_article": (n, code), "s": {"titolo_articolo": rubrica}, **tags}
    if rank is not None:
        row["_fused_rank"] = rank
    return row


# --- A1 ------------------------------------------------------------------------------------------------------------------

def test_every_hit_outside_the_cases_matters_goes_before_grading():
    from src.rag import topic_groups
    rows = [_row("624", "c.p.", "Furto", _multi_signal=True),
            _row("2043", "c.c.", "Risarcimento per fatto illecito"),                 # a vector hit, another code
            _row("1490", "c.c.", "Garanzia per i vizi della cosa venduta"),
            _row("625", "c.p.", "Circostanze aggravanti"),                           # the case's Libro
            _row("62", "c.p.", "Circostanze attenuanti comuni", _first_read="attenuanti"),
            _row("1218", "c.c.", "Responsabilità del debitore")]
    log = {}
    kept = topic_groups.matter_gate(rows, ["1218 c.c."], log, every=True)
    assert [r["_article"][0] for r in kept] == ["624", "625", "62", "1218"]
    assert [(d["article"], d["hit"]) for d in log["dropped"]] == [("2043 c.c.", "vector"), ("1490 c.c.", "vector")]
    assert len(topic_groups.matter_gate(rows, ["1218 c.c."], {})) == len(rows)       # keyword hits only: as before


def test_no_known_matter_drops_nothing():
    from src.rag import topic_groups
    rows = [_row("2043", "c.c."), _row("624", "c.p.")]
    assert topic_groups.matter_gate(rows, [], {}, every=True) == rows


# --- A2 ------------------------------------------------------------------------------------------------------------------

def test_the_grading_reads_at_most_thirty_and_the_kept_ones_always():
    from src.rag import interactive_flow as flow
    rows = [_row(str(100 + i), "c.c.", rank=i) for i in range(1, 61)]
    rows.append(_row("1803", "c.c.", "Nozione", rank=500))              # the first read's institution, ranked last
    out, log = flow.grade_cap(rows, ["1803 c.c."])
    labels = [f"{r['_article'][0]} {r['_article'][1]}" for r in out]
    assert len(labels) == flow.GRADE_CAP == 30 and "1803 c.c." in labels
    assert labels[:29] == [f"{100 + i} c.c." for i in range(1, 30)] and log["kept"] == 30
    assert log["dropped"][0] == "130 c.c." and len(log["dropped"]) == 31
    few = rows[:5]
    assert flow.grade_cap(few, [])[0] == few


def test_the_cap_runs_before_the_grading_in_the_flow():
    src = (HERE.parent / "src" / "rag" / "interactive_flow.py").read_text(encoding="utf-8")
    assert src.index("grade_cap(article_rows, keep_now)") < src.index('graded = _isolated("grading"')
    assert "every=True" in src


# --- A3 ------------------------------------------------------------------------------------------------------------------

def test_one_thesis_sample():
    from src.rag import lawyer_method as M
    assert M.SAMPLES == 1 and M.sampling()[0] == 1


# --- A4 ------------------------------------------------------------------------------------------------------------------

def test_independent_jobs_run_at_once_and_keep_their_order():
    from src.rag import interactive_flow as flow
    started = threading.Barrier(3, timeout=5)

    def job(i):
        def run():
            started.wait()              # all three must be running together, or this times out
            return i * 10
        return run
    assert flow.concurrently([job(1), job(2), job(3)]) == [10, 20, 30]
    assert flow.concurrently([lambda: 7]) == [7]


def test_the_clients_objections_go_through_concurrently():
    src = (HERE.parent / "src" / "rag" / "interactive_flow.py").read_text(encoding="utf-8")
    start = src.index("def method_inputs_clients(")
    body = src[start:src.index("\ndef ", start + 10)]
    assert "with_objections=False" in body and "concurrently(" in body and "method_objections(ctx, mplan)" in body


# --- A5 ------------------------------------------------------------------------------------------------------------------

def test_the_counters_in_the_timings_line():
    from src.rag import interactive_flow as flow
    state = {"timings": {"graded_round1": 28, "graded_answer": 4, "round1_llm": 9, "first_read_llm": 1,
                         "answer_llm": 14},
             "thesis_counts_by_client": {"Tizio": {"samples": 1}, "Caio": {"samples": 1}},
             "grade_funnel": {"retrieved": 90, "matter_gated": 12, "hits": 61, "capped": ["1 c.c."] * 31,
                              "graded": 30},
             "block_counters": {"b_unadmitted": 0}}
    c = flow.case_counters(state, 212.4)
    assert c == {"graded_articles": 32, "samples_used": 2, "llm_calls": 24, "total_s": 212.4,
                 "funnel": {"retrieved": 90, "matter_gated": 12, "hits": 61, "capped": 31, "graded": 30},
                 "unadmitted_candidates": 0, "main_outside_institution": 0, "uncertain_integrated": 0,
                 "fragments_dropped": {}, "conditions_half": 0, "conditions_rejoined": 0, "b_unadmitted": 0}


def test_the_nightly_prints_the_counters_before_and_after():
    nightly = _script("nightly_sim")
    now = [{"id": "c1", "checks": [], "counters": {"graded_articles": 30, "llm_calls": 20, "funnel": {"graded": 30}}}]
    base = {"cells": [{"id": "c1", "counters": {"graded_articles": 81, "llm_calls": 20}}]}
    text = "\n".join(nightly.counter_lines(now, base))
    assert "| graded_articles | 81 → 30 |" in text and "| llm_calls | 20 |" in text and "| funnel.graded | 30 |" in text
