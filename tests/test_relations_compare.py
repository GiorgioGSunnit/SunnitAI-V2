"""scripts/relations_compare.py: relations agent v0's A/B measurement read off saved dry runs (28 Oct)."""
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("relations_compare", HERE.parent / "scripts" / "relations_compare.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)
ev = rc.eval_module(str(HERE.parent))


def _record(pool, plan, answer, added=None, seconds=120):
    events = [{"event": "case_analysis_retrieval", "data": {"article_refs": pool}},
              {"event": "case_analysis_plan", "data": {"norme": plan}}]
    if added is not None:
        events.append({"event": "article_relations_added", "data": {"seeds": ["1490 c.c."], "added": added}})
    return {"events": events, "answer": answer, "status": "ok", "seconds": seconds}


EXPECTED = [{"code": "cc", "article": "1490"}, {"code": "cc", "article": "1495"},
            {"code": "consumo", "any_of": ["128", "129"]}]


def test_trace_reads_pool_plan_answer_and_the_additions():
    rec = _record(["1490 c.c.", "1495 c.c.", "1453 c.c."], ["1490 c.c."],
                  "Secondo l'art. 1490 c.c. e l'art. 1495 c.c. il compratore...",
                  added={"1495 c.c.": "related to 1490 c.c. (applied)", "1453 c.c.": "related to 1490 c.c. (cited)"})
    t = rc.trace(rec, EXPECTED, ev.mentions)
    assert [(r["label"], r["pool"], r["plan"], r["answer"]) for r in t["rows"]] == [
        ("1490 cc", True, True, True), ("1495 cc", True, False, True), ("128/129 consumo", False, False, False)]
    assert t["added"] == ["1453 c.c.", "1495 c.c."] and t["added_expected"] == ["1495 c.c."]
    assert t["added_in_answer"] == ["1495 c.c."]


def test_main_prints_the_totals(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    (out / "c2_A.json").write_text(json.dumps(_record(["1490 c.c."], ["1490 c.c."], "art. 1490 c.c.")), encoding="utf-8")
    (out / "c2_B.json").write_text(json.dumps(_record(["1490 c.c.", "1495 c.c."], ["1490 c.c.", "1495 c.c."],
                                                      "artt. 1490 e 1495 c.c.", added={"1495 c.c.": "related"})),
                                   encoding="utf-8")
    (out / "barca_A.json").write_text(json.dumps(_record([], [], "")), encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text(json.dumps({"c2": {"id": "c2", "expected": EXPECTED}}), encoding="utf-8")
    import sys
    argv = sys.argv
    sys.argv = ["relations_compare.py", "--dir", str(out), "--cases", str(cases), "--code-folder", str(HERE.parent)]
    try:
        rc.main()
    finally:
        sys.argv = argv
    text = capsys.readouterr().out
    assert "barca         missing a run (A)" in text
    assert "answer changed for: 1495 cc (gained)" in text
    assert "A (today)           expected 3: pool 1 (33%), plan 1 (33%), MENTIONED 1 (33%)" in text
    assert "B (with relations)  expected 3: pool 2 (67%), plan 2 (67%), MENTIONED 2 (67%)" in text
    assert "plan 2 articles, 0 not expected; answer names 2, 0 not expected" in text      # B's price line
    assert "relations added 1 articles: 1 of them expected, 1 reached the answer" in text
