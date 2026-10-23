"""scripts/dryrun_strategy.py's report, offline: built from vlog events only, never any text."""

import importlib.util
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("dryrun_strategy", HERE.parent / "scripts" / "dryrun_strategy.py")
dr = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(dr)


def test_report_before_and_after_the_fix():
    before = [("case_analysis_retrieval", {"articles": 95, "rulings": 5}, 2000.0),
              ("case_analysis_applicability", {"checked": 89, "error": "Error code: 400 - maximum context length"}, 216.4),
              ("case_analysis_prompt_tokens", {"total": 16349, "budget": 13000, "step": 5,
                                               "articles_left_out": ["600 c.p."] * 40}, None)]
    lines = dr.summarise(before, "ERROR BadRequestError: maximum context length", None, 118.2, 14146)
    assert lines[0] == "document: 14146 characters" and lines[1].startswith("status: ERROR BadRequestError")
    assert "applicability call: checked 89, 0.2s, error: Error code: 400" in "\n".join(lines)
    assert "draft prompt: 16349 est. tokens, budget 13000, step 5, dropped to fit -, left out 40" in lines

    after = [("case_analysis_applicability", {"checked": 36, "batches": 3, "failed_batches": 0}, 9100.0),
             ("case_analysis_applicability", {"checked": ["116 c.p."] * 36, "dropped": [{}] * 4, "unchecked": 53}, None),
             ("case_analysis_prompt_tokens", {"total": 12880, "budget": 13000, "step": 6, "dropped_for_size": 21,
                                              "articles_left_out": ["x"] * 30}, None)]
    lines = dr.summarise(after, "ok", {"draft": "x" * 9000, "citations": [1, 2, 3]}, 96.0, 14146)
    text = "\n".join(lines)
    assert "applicability call: checked 36, batches 3, failed batches 0, 9.1s" in text
    assert "applicability result: 36 judged, 4 dropped, 53 unchecked" in text
    assert "dropped to fit 21" in text and "answer: 9000 characters, 3 citations" in text
