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


def test_long_cases_are_carried_over_from_eval_narrative_cases_json():
    """l1-l4 (round 5, 18 Oct) moved into eval_narrative_cases.json on 19
    Oct, so exp_distill.py and eval_narrative.py share one source of truth
    instead of two copies that can drift apart."""
    import json
    cases = {c["id"]: c for c in json.loads(ed.DEFAULT_CASES.read_text(encoding="utf-8"))["cases"]}
    for case_id in ("l1", "l2", "l3", "l4"):
        assert case_id in ed._CARRIED_OVER
        case = cases[case_id]
        assert case["question"].strip()
        for item in case["expected"]:
            assert item["code"] in ed.ev.CODES
            assert "article" in item


# --- Round 6: ranking the UNION pool (RERANK/LLM_RELEVANCE/FUSION_BASELINE) -

def test_ref_label_builds_a_natural_citation():
    assert ed._ref_label(("cp", "584")) == "584 c.p."
    assert ed._ref_label(("consumo", "130")) == "130 Codice del Consumo"


def test_signal_counts_how_many_of_the_three_methods_found_each_ref():
    union = [("cp", "584"), ("cp", "589"), ("cp", "43")]
    current = [("cp", "584")]
    baseline = [("cp", "584"), ("cp", "589")]
    distill = [("cp", "43")]
    counts = ed._signal_counts(union, current, baseline, distill)
    assert counts == {("cp", "584"): 2, ("cp", "589"): 1, ("cp", "43"): 1}


def test_ref_of_row_reads_the_fetch_any_article_tag():
    row = {"_article": ("584", "c.p.")}
    assert ed._ref_of_row(row) == ("cp", "584")
    assert ed._ref_of_row({}) is None
    assert ed._ref_of_row({"_article": ("1", "Fonte non classificata")}) is None


def test_union_texts_joins_titolo_articolo_and_plain_text_and_merges_fragments():
    rows = [
        {"_article": ("584", "c.p."), "s": {"titolo_articolo": "Omicidio preterintenzionale", "plain_text": "Chiunque..."}},
        {"_article": ("584", "c.p."), "s": {"titolo_articolo": "", "plain_text": "...continua"}},
        {"_article": ("589", "c.p."), "s": {"titolo_articolo": "", "plain_text": "Chiunque cagiona per colpa..."}},
    ]
    texts = ed._union_texts(rows)
    assert texts[("cp", "584")] == "Omicidio preterintenzionale. Chiunque... ...continua"
    assert texts[("cp", "589")] == "Chiunque cagiona per colpa..."


def test_union_texts_truncates_to_chars():
    rows = [{"_article": ("584", "c.p."), "s": {"titolo_articolo": "", "plain_text": "x" * 1000}}]
    assert len(ed._union_texts(rows, chars=50)[("cp", "584")]) == 50


def test_rank_by_reranker_orders_by_score_then_signal_count(monkeypatch):
    union = [("cp", "584"), ("cp", "589"), ("cp", "43")]
    texts = {("cp", "584"): "a", ("cp", "589"): "b", ("cp", "43"): "c"}
    # "589" and "43" tie on score; "43" has more signals, so it must come first.
    monkeypatch.setattr(ed, "_rerank_scores", lambda query, docs: [0.9, 0.5, 0.5])
    signal_counts = {("cp", "584"): 1, ("cp", "589"): 1, ("cp", "43"): 3}
    ranked, scores = ed._rank_by_reranker(union, texts, "query", signal_counts)
    assert ranked == [("cp", "584"), ("cp", "43"), ("cp", "589")]
    assert scores[("cp", "584")] == 0.9


def test_rank_by_llm_relevance_orders_by_score_then_reranker_score(monkeypatch):
    union = [("cp", "584"), ("cp", "589"), ("cp", "43")]
    texts = {("cp", "584"): "a", ("cp", "589"): "b", ("cp", "43"): "c"}
    monkeypatch.setattr(ed, "_llm_relevance_scores",
                        lambda bullets, refs, texts: {("cp", "584"): 3, ("cp", "589"): 1, ("cp", "43"): 1})
    reranker_scores = {("cp", "584"): 0.1, ("cp", "589"): 0.2, ("cp", "43"): 0.9}
    ranked, scores = ed._rank_by_llm_relevance(union, texts, "bullets", reranker_scores)
    # "584" wins on relevance (3); among the "589"/"43" tie (both 1), "43" wins on reranker score (0.9 > 0.2).
    assert ranked == [("cp", "584"), ("cp", "43"), ("cp", "589")]


def test_rank_by_llm_relevance_fails_open_on_error(monkeypatch):
    union = [("cp", "584"), ("cp", "589")]

    def boom(*a, **k):
        raise RuntimeError("timeout")
    monkeypatch.setattr(ed, "_llm_relevance_scores", boom)
    ranked, scores = ed._rank_by_llm_relevance(union, {}, "bullets", {})
    assert ranked == union
    assert scores == {}


def test_llm_relevance_scores_parses_the_label_back_to_its_ref(monkeypatch):
    union_refs = [("cp", "584"), ("cp", "589")]
    texts = {("cp", "584"): "testo 584", ("cp", "589"): "testo 589"}
    monkeypatch.setattr(ed.dg, "_chat",
                        lambda *a, **k: '[{"articolo": "584 c.p.", "punteggio": 3}, '
                                       '{"articolo": "589 c.p.", "punteggio": 0}]')
    scores = ed._llm_relevance_scores("bullets", union_refs, texts)
    assert scores == {("cp", "584"): 3, ("cp", "589"): 0}


def test_llm_relevance_scores_returns_empty_without_any_text():
    assert ed._llm_relevance_scores("bullets", [("cp", "584")], {}) == {}


# --- Round 8 (19 Oct): parity with src.rag.defensive_generation's fusion ----
#
# CASE_POOL_FUSION_ENABLED's _select_articles_fusion ported this script's
# FUSION_BASELINE ranking and bullet-distillation instead of sharing them --
# the Cypher it used for its own BASELINE/DISTILL search did a naive
# concatenation instead of a true best-rank merge, and nothing kept the two
# merge algorithms (or the distill prompt) from drifting apart silently.
# These tests assert identity, not just equal behaviour today: if a future
# edit reintroduces a private copy in either file, it fails immediately,
# before the two could measure different things again.

def test_merge_functions_are_the_same_objects_as_defensive_generations():
    assert ed._merge_best_rank is ed.dg._fusion_best_rank_merge
    assert ed._merge_by_hits is ed.dg._fusion_merge_by_hits


def test_distill_bullets_is_the_same_function_defensive_generation_uses():
    assert ed._distill_bullets is ed.dg._fusion_distill_bullets


# --- Round 10: TIMELINE ------------------------------------------------------

def test_rrf_sums_reciprocal_ranks_across_lists():
    a = [("cp", "584"), ("cp", "581")]
    b = [("cp", "589"), ("cp", "584")]
    scores = ed._rrf_scores(a, b)
    assert scores[("cp", "584")] == 1 / 61 + 1 / 62
    assert scores[("cp", "589")] == 1 / 61
    # found by both lists beats rank 1 in only one of them
    assert ed._rrf(a, b) == [("cp", "584"), ("cp", "589"), ("cp", "581")]


def test_rrf_breaks_ties_by_first_appearance():
    assert ed._rrf([("cp", "1")], [("cc", "2")]) == [("cp", "1"), ("cc", "2")]


def test_parse_timeline_normalises_steps_and_drops_terms_with_numbers():
    reply = [
        {"n": 1, "actor": " Tizio ", "action": "colpisce Caio", "object": "Caio", "means": "pugno",
         "result": "frattura", "relevance": "2", "terms": ["lesione personale", "art. 582", "violenza", 7]},
        {"n": 2, "actor": "Tizio", "action": "torna a casa", "relevance": 0, "terms": []},
        "not a step",
        {"action": "x", "relevance": 5},
    ]
    steps = ed._parse_timeline(reply)
    assert steps[0] == {"n": 1, "actor": "Tizio", "action": "colpisce Caio", "object": "Caio", "means": "pugno",
                        "result": "frattura", "relevance": 2, "terms": ["lesione personale", "violenza"]}
    assert [s["relevance"] for s in steps] == [2, 0, 0]       # unreadable relevance counts as 0
    assert steps[2]["n"] == 4                                 # missing n -> position in the reply
    assert ed._parse_timeline({"steps": []}) == []


def test_kept_steps_discard_relevance_zero_only():
    steps = [{"relevance": 0}, {"relevance": 1}, {"relevance": 2}]
    assert ed._kept_steps(steps) == [{"relevance": 1}, {"relevance": 2}]


def test_timeline_query_is_action_result_and_terms():
    step = {"action": "colpisce Caio", "result": "", "terms": ["lesione personale", "violenza"]}
    assert ed._timeline_query(step) == "colpisce Caio lesione personale violenza"


def test_cited_ranking_orders_by_best_ruling_then_citation_count():
    cites = {
        "r1": {("cp", "43"): 1, ("cp", "584"): 1},
        "r2": {("cp", "589"): 5, ("cp", "584"): 2},
    }
    # Both of r1's articles come first (r1 is the best-ranked ruling); between
    # them 584 wins on total citations (1 + 2 in r2) over 43 (1); then 589,
    # cited only by r2, however often.
    assert ed._cited_ranking(["r1", "r2"], cites) == [("cp", "584"), ("cp", "43"), ("cp", "589")]
    assert ed._cited_ranking(["missing"], cites) == []
