"""v60o (the seven-case sim: kept 63-65, graded 36-45): every kept candidate graded -- the user-named articles, the rules'
and tables' anchors and the first read's hits in a call of their own, first; the prefilter's rows after; the round
limit never cuts the kept set -- and the invariant graded == kept per run, logged; identical rubriche in
check_rubrica_search counted as found when the result set holds the article. The model and the graph are stubbed."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import fact_resolver as fr  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
from test_v60n import RubricaSession, store, _found  # noqa: E402,F401  (store: the fixture)

KEEP = ["640 c.p.", "646 c.p.", "648 c.p.", "43 c.p.", "117 c.p."]       # named, anchors, first read -- the sim's lost


def _rows(n=65):
    rows = [{"_article": (str(k), "c.p."), "s": {"plain_text": f"testo {k}", "titolo_articolo": f"r{k}"}}
            for k in range(1000, 1000 + n)]
    return rows + [{"_article": tuple(label.split(" ")), "s": {"plain_text": "t", "titolo_articolo": "r"}} for label in KEEP]


@pytest.fixture
def grader(monkeypatch):
    """The graded check with the prefilter's top N kept in order and every model call recorded (its labels, retry); one
    worker, so the calls run in their submission order; skip: labels never graded; grades: {label: grade}."""
    calls, skip, grades = [], set(), {}
    monkeypatch.delenv(dg.GRADER_VOTE_FLAG, raising=False)
    monkeypatch.delenv(dg.GRADER_FIXED_ORDER_FLAG, raising=False)
    monkeypatch.setattr(dg, "_APPLICABILITY_WORKERS", 1)
    monkeypatch.setattr(dg, "_prefiltered", lambda texts, case, top, *a: (dict(list(texts.items())[:top]),
                                                                          list(texts)[top:]))

    def batch(case, texts, retry=False, *a, **k):
        calls.append(list(texts))
        read = {label: {"grade": grades.get(label, "2")} for label in texts if label not in skip}
        return read, "" if len(read) == len(texts) else f"{len(texts) - len(read)} missing"
    monkeypatch.setattr(dg, "_graded_batch", batch)
    monkeypatch.setattr(dg, "_prompt_reach", lambda *a, **k: set())      # the prompt estimate admits nothing
    return calls, skip, grades


def _limits(prefilter=50):
    return {"graded": True, "grade_kept": True, "grade_all": True, "rounds": flow.ROUND1_GRADE_ROUNDS,
            "prefilter": prefilter, "prefilter_keep": list(KEEP)}


def test_every_kept_candidate_is_graded_the_priority_ones_first(grader):
    calls, _, _ = grader
    with dg.grading_limits(_limits()) as limits:
        rows = dg._filter_applicable_articles("Il caso.", "", _rows(), {})
    assert calls[0] == KEEP                                 # their own batch, submitted first
    assert len(calls) == 6 and sum(len(c) for c in calls[1:]) == 50     # the prefilter's 50, whatever the round limit
    assert limits["calls"]["calls"] == 6 and limits["calls"]["longest_s"] >= 0 and limits["calls"]["longest"]
    graded = {f"{r['_article'][0]} {r['_article'][1]}" for r in rows if isinstance(r.get("_grade"), int)}
    assert set(KEEP) <= graded and len(graded) == 55
    assert limits["invariant"] == {"kept": 55, "graded": 55, "ok": True, "missing": [], "first": KEEP, "no_text": []}
    assert limits["kept_n"] == limits["graded_n"] == 55 and limits["rounds_run"] == 1
    assert flow.grading_invariant(limits)["ok"] is True


def test_the_fixed_order_flag_never_moves_the_priority_ones_out_of_the_first_batch(grader, monkeypatch):
    monkeypatch.setenv(dg.GRADER_FIXED_ORDER_FLAG, "true")
    calls, _, _ = grader
    with dg.grading_limits(_limits()):
        dg._filter_applicable_articles("Il caso.", "", _rows(), {})
    assert set(calls[0]) == set(KEEP)


def test_a_kept_candidate_left_without_a_grade_breaks_the_invariant_and_says_which(grader):
    calls, skip, _ = grader
    skip.add("646 c.p.")
    with dg.grading_limits(_limits()) as limits:
        dg._filter_applicable_articles("Il caso.", "", _rows(), {})
    inv = flow.grading_invariant(limits)
    assert inv["ok"] is False and inv["missing"] == ["646 c.p."] and inv["kept"] == 55 and inv["graded"] == 54


def test_the_exempt_articles_are_graded_too_and_never_removed(grader):
    calls, _, grades = grader
    rows = _rows(10)
    rows[-5]["_heading_match"] = True                       # 640: found by its own rubrica
    grades.update({"640 c.p.": "0", "646 c.p.": "0"})
    with dg.grading_limits(_limits()) as limits:
        out = dg._filter_applicable_articles("Il cliente chiede dell'art. 117 c.p.", "", rows, {})
    by = {f"{r['_article'][0]} {r['_article'][1]}": r for r in out if r.get("_article")}
    assert "640 c.p." in calls[0] and "117 c.p." in calls[0]           # graded, the user-named one with them
    assert by["640 c.p."]["_grade"] == 0 and by["640 c.p."]["_applicability"] == "marginale"     # exempt: kept
    assert "646 c.p." not in by                                         # not exempt: grade 0 removes it
    assert by["117 c.p."]["_grade"] == 2 and limits["invariant"]["ok"] is True
    with dg.grading_limits({**_limits(), "grade_kept": False}):
        out = dg._filter_applicable_articles("Il cliente chiede dell'art. 117 c.p.", "", _rows(10), {})
    assert "_grade" not in {f"{r['_article'][0]} {r['_article'][1]}": r for r in out}["117 c.p."]   # as before


def test_a_keep_label_with_no_text_breaks_the_invariant(grader):
    rows = _rows(10)
    rows[-1]["s"] = {"titolo_articolo": "r"}                # 117: no text to grade
    with dg.grading_limits(_limits()) as limits:
        dg._filter_applicable_articles("Il caso.", "", rows, {})
    inv = flow.grading_invariant(limits)
    assert inv["ok"] is False and inv["no_text"] == ["117 c.p."] and inv["missing"] == ["117 c.p."]
    assert inv["kept"] == inv["graded"] + 1


def test_the_invariant_with_nothing_to_grade_and_with_a_failed_check():
    assert flow.grading_invariant({})["ok"] is True
    broken = flow.grading_invariant({"candidates_n": 40, "graded_n": 0})
    assert broken["ok"] is False and "not graded" in broken["why"]


def test_without_grade_kept_the_prompt_estimate_rules_as_before(grader):
    calls, _, _ = grader
    with dg.grading_limits({**_limits(), "grade_kept": False}) as limits:
        dg._filter_applicable_articles("Il caso.", "", _rows(), {})
    assert calls == [KEEP] and "invariant" not in limits    # v60d: only the estimate's and the keep list


def test_round_one_grades_the_named_then_the_anchors_then_the_first_read():
    rows = [{"_article": ("52", "c.p.")}, {"_article": ("575", "c.p.")}, {"_article": ("1062", "c.c.")}]
    keep = flow.grading_keep(rows, "Il cliente chiede dell'art. 575 c.p.", ["1062 c.c."], ["52 c.p.", "575 c.p."])
    assert keep == ["575 c.p.", "1062 c.c.", "52 c.p."]
    assert flow.grading_keep(rows, "Nessun articolo.", [], ["52 c.p."]) == ["52 c.p."]


def test_grading_breadth_reports_the_kept_set():
    out = flow.grading_breadth([], {"candidates_n": 120, "prefilter": 50, "prefilter_kept": ["1 c.p."] * 50,
                                    "prefilter_keep": KEEP, "kept_n": 55, "graded_n": 55})
    assert out["kept"] == 55 and out["graded"] == 55


# --- identical rubriche -----------------------------------------------------------------------------------------------------

CONFISCA = {"240": "Confisca", "322-ter": "Confisca (1)", "416-bis": "Associazioni di tipo mafioso anche straniere",
            "446": "Confisca", "466-bis": "Confisca", "600-septies": "Confisca", "640-quater": "Confisca.",
            "644": "Usura", "648-quater": "Confisca"}


def test_identical_rubriche_are_all_the_search_result(store, monkeypatch):
    from test_v60n import RUBRICHE
    for n, r in CONFISCA.items():
        monkeypatch.setitem(RUBRICHE, n, r)
    found, log = _found("Confisca", RubricaSession(CONFISCA))
    assert found == ["240", "322-ter", "446", "466-bis", "600-septies", "640-quater", "648-quater"]
    assert log["rubrica"]["how"] == "esatta" and fr.EXACT_KEPT >= 7


def test_the_check_script_counts_each_identical_rubrica_found_in_the_shared_result(store, monkeypatch, capsys):
    from test_v60n import RUBRICHE
    for n, r in CONFISCA.items():
        monkeypatch.setitem(RUBRICHE, n, r)
    spec = importlib.util.spec_from_file_location("check_rubrica_search", HERE.parent / "scripts" / "check_rubrica_search.py")
    check = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(check)
    session = RubricaSession(CONFISCA)

    class Session:
        def __enter__(self):
            return session

        def __exit__(self, *a):
            return False
    import types
    fake_main = types.ModuleType("src.rag.main")
    fake_main.NEO4J_DATABASE, fake_main.driver = "db", type("D", (), {"session": lambda self, **k: Session()})()
    monkeypatch.setitem(sys.modules, "src.rag.main", fake_main)
    monkeypatch.setattr("src.rag.doc_lookup._fetch_doc_names", lambda driver, db: [{"id": "cp", "name": "Codice Penale"}])
    code = check.main(["--names"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert "| 600-septies | Confisca (6 identiche) |" in out and out.count("| yes |") == 7
    assert "7/7 searches found their articles" in out


# --- v60p: every grading call of round 1 at once, bounded by _APPLICABILITY_WORKERS ----------------------------------------

def test_the_grading_calls_run_concurrently_and_a_retry_never_waits_for_the_others(grader, monkeypatch):
    import threading
    import time as _time
    _, skip, _ = grader
    monkeypatch.setattr(dg, "_APPLICABILITY_WORKERS", 3)
    lock, active, peak, events = threading.Lock(), [0], [0], []

    def batch(case, texts, retry=False, *a, **k):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            events.append(("start", retry, list(texts)[0], _time.monotonic()))
        _time.sleep(0.01 if "640 c.p." in texts else 0.15)          # the priority batch comes back first
        with lock:
            active[0] -= 1
            events.append(("end", retry, list(texts)[0], _time.monotonic()))
        read = {label: {"grade": "2"} for label in texts if label not in skip}
        return read, ""
    monkeypatch.setattr(dg, "_graded_batch", batch)
    skip.add("646 c.p.")                                     # missing from the priority batch: retried
    with dg.grading_limits(_limits()) as limits:
        dg._filter_applicable_articles("Il caso.", "", _rows(), {})
    assert peak[0] == 3                                      # never more than the workers, and every one used
    retry_start = next(t for kind, retry, _, t in events if kind == "start" and retry)
    last_first_end = max(t for kind, retry, _, t in events if kind == "end" and not retry)
    assert retry_start < last_first_end                      # the retry ran while the prefilter's batches were out
    stats = limits["calls"]
    assert stats["calls"] == 7 and stats["longest_s"] >= 0.15 and stats["wall_s"] < 0.15 * 7
    assert limits["invariant"]["missing"] == ["646 c.p."]
