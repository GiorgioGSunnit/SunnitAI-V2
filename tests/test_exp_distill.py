"""scripts/exp_distill.py's pure merge/scoring helpers, offline. The script
itself needs a live Neo4j and chat model (it's meant to run on prod/be2);
this file only exercises the logic that decides which (code, article)
wins across ranked lists and whether an expected article was recalled --
the part a bad merge rule or an off-by-one would silently corrupt the
whole experiment's table without ever raising an error.
"""

import importlib.util
import os
from pathlib import Path

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("exp_distill", HERE.parent / "scripts" / "exp_distill.py")
ed = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ed)


def test_row_ref_reads_code_and_article_from_a_retrieved_row():
    row = {"name": "624-bis.0.0", "doc_name": "Codice Penale 2026"}
    assert ed._row_ref(row) == ("cp", "624-bis")


def test_row_ref_is_none_without_a_recognisable_document_or_article():
    assert ed._row_ref({"name": "624-bis.0.0", "doc_name": "Qualcos'altro"}) is None
    assert ed._row_ref({"name": "Premessa", "doc_name": "Codice Civile 2026"}) is None


def test_ranked_refs_keeps_first_occurrence_order_and_drops_duplicates():
    rows = [
        {"name": "584.0.0", "doc_name": "Codice Penale 2026"},
        {"name": "589.0.0", "doc_name": "Codice Penale 2026"},
        {"name": "584.1", "doc_name": "Codice Penale 2026"},   # same article, a fragment -- not a new ref
    ]
    assert ed._ranked_refs(rows) == [("cp", "584"), ("cp", "589")]


def test_merge_best_rank_prefers_whichever_list_ranks_it_higher():
    vector = [("cp", "584"), ("cp", "581")]
    fulltext = [("cp", "589"), ("cp", "584")]
    # "584" is rank 2 in fulltext but rank 1 in vector -- the better rank wins.
    assert ed._merge_best_rank(vector, fulltext) == [("cp", "584"), ("cp", "589"), ("cp", "581")]


def test_merge_by_hits_prefers_an_article_several_calls_agree_on():
    # "589" only appears once (rank 1), "584" appears in two calls (rank 2 and rank 1).
    call_a = [("cp", "581"), ("cp", "584")]
    call_b = [("cp", "584")]
    call_c = [("cp", "589")]
    assert ed._merge_by_hits(call_a, call_b, call_c) == [("cp", "584"), ("cp", "581"), ("cp", "589")]


def test_merge_by_hits_breaks_a_tie_in_hit_count_by_best_rank():
    call_a = [("cp", "584"), ("cp", "589")]   # 584 rank 1, 589 rank 2
    call_b = [("cp", "589"), ("cp", "584")]   # 589 rank 1, 584 rank 2
    # both appear in both calls (hits=2 each); 589's best rank (1, from call_b)
    # beats 584's best rank (1, from call_a) only by insertion order -- actual
    # tie-break is the article first seen at its own best rank.
    merged = ed._merge_by_hits(call_a, call_b)
    assert set(merged) == {("cp", "584"), ("cp", "589")}


def test_code_key_maps_standard_abbreviations_and_falls_back_to_document_name():
    assert ed._code_key("c.p.") == "cp"
    assert ed._code_key("c.c.") == "cc"
    assert ed._code_key("Codice del Consumo 2026") == "consumo"
    assert ed._code_key("Fonte non classificata") is None


def test_recall_counts_expected_entries_found_within_top_k():
    ranked = [("cp", "584"), ("cp", "581"), ("cp", "43")]
    expected = [{"code": "cp", "article": "584"}, {"code": "cp", "article": "589"}]
    found, total = ed._recall(ranked, expected, 10)
    assert (found, total) == (1, 2)


def test_recall_handles_any_of_like_c2():
    ranked = [("consumo", "130"), ("cc", "1495")]
    expected = [{"code": "consumo", "any_of": ["128", "129", "130"], "label": "any"}]
    found, total = ed._recall(ranked, expected, 10)
    assert (found, total) == (1, 1)


def test_recall_respects_the_k_cutoff():
    ranked = [("cp", "1"), ("cp", "2"), ("cp", "3"), ("cp", "584")]
    expected = [{"code": "cp", "article": "584"}]
    assert ed._recall(ranked, expected, 3) == (0, 1)
    assert ed._recall(ranked, expected, 10) == (1, 1)


def test_dedupe_ordered_keeps_first_occurrence_in_priority_order():
    current = [("cp", "584")]
    baseline = [("cp", "584"), ("cp", "589")]
    distill = [("cp", "589"), ("cp", "43")]
    assert ed._dedupe_ordered(current, baseline, distill) == [("cp", "584"), ("cp", "589"), ("cp", "43")]


def test_neighbour_numbers_is_numeric_only():
    assert ed._neighbour_numbers("584") == ["583", "585", "584-bis"]
    assert ed._neighbour_numbers("1") == ["2", "1-bis"]   # no "0" neighbour
    assert ed._neighbour_numbers("624-bis") == []   # suffixed numbers are out of scope, by design


def test_noise_counts_non_expected_articles_in_the_top_k():
    ranked = [("cp", "584"), ("cp", "999"), ("cp", "589")]
    expected = [{"code": "cp", "article": "584"}, {"code": "cp", "article": "589"}]
    assert ed._noise(ranked, expected, 10) == 1
    assert ed._noise(ranked, expected, 1) == 0   # only "584" (expected) is within the top 1


def test_noise_respects_any_of():
    ranked = [("consumo", "130"), ("cc", "1495")]
    expected = [{"code": "consumo", "any_of": ["128", "129", "130"], "label": "any"},
                {"code": "cc", "article": "1495"}]
    assert ed._noise(ranked, expected, 10) == 0


def test_to_fetch_ref_maps_eval_code_to_a_search_name():
    assert ed._to_fetch_ref(("cp", "584")) == ("584", "c.p.")
    assert ed._to_fetch_ref(("consumo", "130")) == ("130", "Codice del Consumo")


def test_new_long_cases_are_well_formed():
    for case in ed._NEW_LONG_CASES:
        assert case["question"].strip()
        for item in case["expected"]:
            assert item["code"] in ed.ev.CODES
            assert "article" in item
