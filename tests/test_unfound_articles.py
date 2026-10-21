"""scripts/unfound_articles.py: the expected articles no exp_distill.py method
found in its top k, read from a results.json."""

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))

import unfound_articles as ua  # noqa: E402

DATA = {
    "k": 20,
    "cases": {
        "p1": [{"code": "cp", "article": "56"}, {"code": "cp", "article": "628"}],
        "c2": [{"code": "consumo", "any_of": ["128", "129", "130"]}, {"code": "cc", "article": "1490"}],
    },
    "results": {
        "p1": {"fusion_clean": [["cp", "628"]] + [["cp", str(i)] for i in range(100, 130)] + [["cp", "56"]],
               "latency": {"fusion_clean": 1.0}, "union_candidate_count": 31, "detail": {}},
        "c2": {"fusion_clean": [["consumo", "130"]], "hyde": [["cc", "1453"]]},
    },
}


def test_reports_what_no_method_has_in_its_top_k_with_its_best_rank():
    report = ua.unfound(DATA, k=20)
    assert [(e["article"], best) for e, best in report["p1"]] == [("56", (32, "fusion_clean"))]
    # any_of: one of the range in the top k is enough.
    assert [(e.get("article"), best) for e, best in report["c2"]] == [("1490", None)]


def test_methods_can_be_restricted(capsys, tmp_path):
    # hyde alone has neither the consumer range nor 1490 in its top k.
    assert [e.get("any_of") or e["article"] for e, _ in ua.unfound(DATA, 20, {"hyde"})["c2"]] == \
        [["128", "129", "130"], "1490"]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(DATA), encoding="utf-8")
    assert ua.main([str(path)]) == 0
    out = capsys.readouterr().out
    assert "2/4" in out and "56 cp  (best rank 32 in fusion_clean)" in out and "1490 cc  (never retrieved)" in out
