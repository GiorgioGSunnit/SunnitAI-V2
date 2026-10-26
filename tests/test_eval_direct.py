"""tests/eval_direct.py: the direct questions' scoring helpers and case file (no run, no DB)."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
sys.path.insert(0, str(Path(__file__).resolve().parent))

import eval_direct  # noqa: E402
import eval_narrative  # noqa: E402
from src.rag import source_tiers  # noqa: E402


def test_the_cases_are_twelve_direct_questions_with_known_codes():
    cases = json.loads(eval_direct.DEFAULT_DIRECT_CASES.read_text(encoding="utf-8"))["cases"]
    assert len(cases) == 12 and len({c["id"] for c in cases}) == 12
    assert {c["area"] for c in cases} >= {"penale", "civile", "procedura civile", "procedura penale", "amministrativo"}
    for case in cases:
        assert case["expected"] and all(item["code"] in eval_narrative.CODES for item in case["expected"])


def test_answer_refs_keep_the_codes_the_cites_edges_know():
    answer = "Si applicano l'art. 1490 c.c. e l'art. 21-nonies della legge 241/1990, oltre all'art. 360 c.p.c."
    assert eval_direct.answer_refs(answer, eval_narrative.mentions) == [("1490", "c.c."), ("360", "c.p.c.")]


def test_ruling_fit_separates_answer_articles_boilerplate_and_unrelated():
    fit = source_tiers.ruling_answer_fit(
        {"a": {("1490", "c.c."), ("2043", "c.c.")}, "b": {("360", "c.p.c.")}, "c": {("640", "c.p.")}},
        [("1490", "c.c."), ("360", "c.p.c.")])
    assert (fit["rulings"], fit["citing_answer_article"], fit["boilerplate_only"], fit["unrelated"]) == (3, 2, 1, 1)
    assert fit["per_ruling"]["a"] == ["1490 c.c."]
    assert source_tiers.ruling_answer_fit({}, [])["share_citing"] is None


def test_totals_add_up_the_cases():
    record = {"mentioned": 1, "expected": 2, "non_primary_hits": 3, "non_primary_after_filter": 0,
              "rulings": {"rulings": 2, "citing_answer_article": 1, "boilerplate_only": 0, "unrelated": 1},
              "id": "d1", "route": "direct", "status": "ok"}
    assert eval_direct.totals([record, record])["non_primary_hits"] == 6
    assert any("MENTIONED: 2/4" in line for line in eval_direct.summary_lines([record, record], "x"))
