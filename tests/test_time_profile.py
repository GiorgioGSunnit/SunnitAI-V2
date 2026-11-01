"""scripts/time_profile.py on a synthetic run: no pipeline."""

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("time_profile", ROOT / "scripts" / "time_profile.py")
tp = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(tp)


def _call(stage, start, seconds):
    return {"stage": stage, "start": start, "seconds": seconds, "ok": True}


def test_parallel_calls_count_once():
    assert tp.wall([_call("a", 0, 10), _call("b", 5, 10), _call("c", 20, 1)]) == 16


def test_calls_without_a_stage_are_read_by_their_place():
    calls = tp.named([_call("frame", 0, 1), _call("draft", 1, 1), _call("applicability", 10, 2),
                      _call("draft", 20, 1), _call("plan", 30, 1), _call("section_a", 40, 5),
                      _call("draft", 50, 1), _call("other_review_section", 51, 1), _call("other_mystery", 52, 1),
                      _call("rubrica_fit", 90, 9), _call("plan_json_retry", 31, 1)])
    got = [(c["stage"], c["group"]) for c in calls]
    assert ("router_tie", "retrieval") in got and ("ruling_fit", "graded check") in got
    assert ("post_writing_other", "post-writing") in got and ("writer_review", "post-writing") in got
    assert ("other_mystery", "post-writing") in got and ("plan", "plan") in got
    assert not any(stage == "rubrica_fit" for stage, _ in got)


def test_a_draft_without_a_plan_is_the_one_call_answer():
    assert tp.named([_call("applicability", 0, 1), _call("draft", 5, 9)])[1]["group"] == "sections"


def test_groups_take_the_longer_of_calls_and_windows_and_leave_the_rest_unattributed():
    rec = {"id": "c", "status": "ok", "seconds": 999, "events": [
        ["case_analysis_pipeline", {}, 100000.0], ["case_analysis_retrieval", {}, 30000.0],
        ["case_analysis_institution", {}, 4000.0], ["case_analysis_applicability", {"verdicts": {}}, None]]}
    out = tp.profile_case(rec, [_call("frame", 0, 10), _call("fact_signals", 0, 8), _call("distill", 11, 5)])
    assert out["total"] == 100 and out["groups"]["understanding"] == 10 and out["groups"]["retrieval"] == 30
    assert out["unattributed"] == 60 and out["events"]["case_analysis_institution"]["calls"] == 1


def test_a_run_is_read_and_rendered(tmp_path):
    (tmp_path / "cases" / "h1" / "llm").mkdir(parents=True)
    (tmp_path / "cases" / "h1.input.json").write_text("{}", encoding="utf-8")
    (tmp_path / "cases" / "h1.json").write_text(json.dumps(
        {"id": "h1", "status": "ok", "seconds": 50, "events": [["case_analysis_pipeline", {}, 50000.0]]}), encoding="utf-8")
    (tmp_path / "cases" / "h1" / "llm" / "01_frame.txt").write_text(
        "stage: frame\ncall: 01 (thread x)\nstarted: +2.5s\nduration: 10.0s\nstatus: ok\n\n===== SYSTEM =====\nstage: no\n",
        encoding="utf-8")
    cases, missing = tp.load_run(str(tmp_path))
    assert missing == [] and cases[0]["groups"]["understanding"] == 10
    text = tp.render(str(tmp_path), cases, missing)
    assert "3 at a time" in text and "| unattributed | 40.0 |" in text and "| understanding | frame | 1.0 | 1 |" in text
