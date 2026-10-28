"""scripts/relations_choice.py: can the relations help the choice among the articles already found (28 Oct)."""
import importlib.util
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("relations_choice", HERE.parent / "scripts" / "relations_choice.py")
rc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rc)

RELATED = {"1490": [["1492", 0.03], ["1495", 0.03]], "1492": [["1490", 0.03]], "2043": [["2051", 0.02]]}


def test_cc_numbers_and_symmetric_scores():
    assert rc.cc_numbers(["1490 c.c.", "1490 Codice Civile 2026", "640 c.p.", "337-ter c.c."]) == ["1490", "337-ter"]
    scores = rc.relation_scores(RELATED)
    assert scores[("1495", "1490")] == 0.03 and scores[("2051", "2043")] == 0.02 and ("1495", "1492") not in scores


def test_case_report_separation_and_rescue():
    scores = rc.relation_scores(RELATED)
    pool = ["1490", "1492", "1495", "2043", "1218", "1176"]
    r = rc.case_report(pool, plan=["1490", "1218"], expected={"1490", "1492", "1495"}, scores=scores)
    assert r["separation"] == 1.0                      # the three expected are connected, the others are not
    assert r["dropped_by_plan"] == ["1492", "1495"] and r["rescued"] == ["1492", "1495"] and r["k"] == 5
    assert r["unexpected_in_top"] == ["2043"]          # the top 5 also brings in an unexpected one
    assert rc.separation({"a": 1.0}, {"a"}) is None


def test_main(tmp_path, capsys):
    out = tmp_path / "out"
    out.mkdir()
    record = {"events": [{"event": "case_analysis_retrieval", "data": {"article_refs": ["1490 c.c.", "1492 c.c.", "1218 c.c."]}},
                         {"event": "case_analysis_plan", "data": {"norme": ["1490 c.c.", "1218 c.c."]}}]}
    (out / "barca_A.json").write_text(json.dumps(record), encoding="utf-8")
    (out / "barca_B.json").write_text(json.dumps(record), encoding="utf-8")
    relations = tmp_path / "rel.json"
    relations.write_text(json.dumps({"related": RELATED}), encoding="utf-8")
    cases = tmp_path / "cases.json"
    cases.write_text("{}", encoding="utf-8")
    argv = sys.argv
    sys.argv = ["relations_choice.py", "--dir", str(out), "--relations", str(relations), "--cases", str(cases)]
    try:
        rc.main()
    finally:
        sys.argv = argv
    text = capsys.readouterr().out
    assert "expected articles the plan dropped: 1; among the most connected: 1" in text
    assert text.count("barca") == 1                     # the A run only
