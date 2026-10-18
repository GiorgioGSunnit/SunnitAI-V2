"""CASE_POOL_FUSION_ENABLED (round 6, 19 Oct; requires CASE_ANALYSIS_ENABLED):
_select_articles_fusion replaces _select_articles_v2's article pool (v1's
own code-search + _choose_articles, union institutions) with institution-
resolved union raw-text search union distilled-bullet search, ranked by
fusion (how many of the three sources agree, then best rank) and capped at
_FUSION_POOL_CAP -- validated on scripts/exp_distill.py, where that ranking
(FUSION_BASELINE) beat every re-ranking method tried, at zero added latency.

_search_case_law/_search_similar_facts/_articles_cited_in_rulings/
_candidate_articles/_resolve_institution/_fetch_any/broaden_article_context
are mocked throughout (not re-exercised -- they have their own tests);
these tests isolate the new fusion-pool logic itself: capping, keeping
"keep" uncapped, merging three sources, and never calling _choose_articles.
"""

import json
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402


# --- Pure helpers ------------------------------------------------------------

def test_row_to_fusion_ref_reads_a_standard_code_article():
    row = {"s": {"name": "624-bis.0.0"}, "d": {"name": "Codice Penale 2026"}}
    assert dg._row_to_fusion_ref(row) == ("624-bis", "c.p.")


def test_row_to_fusion_ref_tags_a_non_standard_document_by_its_own_name():
    row = {"s": {"name": "130.0.0"}, "d": {"name": "Codice del Consumo 2026"}}
    assert dg._row_to_fusion_ref(row) == ("130", "Codice del Consumo 2026")


def test_row_to_fusion_ref_rejects_a_non_article_section():
    assert dg._row_to_fusion_ref({"s": {"name": "Premessa"}, "d": {"name": "Codice Civile 2026"}}) is None
    assert dg._row_to_fusion_ref({"s": {"name": "584.0.0"}, "d": {"name": ""}}) is None
    assert dg._row_to_fusion_ref({}) is None


def test_fusion_merge_by_hits_prefers_more_agreeing_sources():
    institution = [("c.p.", "584")]
    raw_text = [("c.p.", "589"), ("c.p.", "584")]
    distill = [("c.p.", "584")]
    # "584" is in all three sources; "589" only in one.
    assert dg._fusion_merge_by_hits(institution, raw_text, distill) == [("c.p.", "584"), ("c.p.", "589")]


# --- _fusion_distill_bullets --------------------------------------------------

def test_fusion_distill_bullets_parses_the_reply(monkeypatch):
    monkeypatch.setattr(dg, "_chat", lambda *a, **k: json.dumps(
        [{"categoria": "condotta", "testo": "sottrazione mediante violenza"},
         {"categoria": "", "testo": ""}]))   # an empty one is dropped
    bullets = dg._fusion_distill_bullets("facts")
    assert bullets == [{"categoria": "condotta", "testo": "sottrazione mediante violenza"}]


def test_fusion_distill_bullets_returns_empty_on_a_non_list_reply(monkeypatch):
    monkeypatch.setattr(dg, "_chat", lambda *a, **k: '{"not": "a list"}')
    assert dg._fusion_distill_bullets("facts") == []


# --- _select_articles_fusion --------------------------------------------------

@pytest.fixture
def fusion_env(monkeypatch):
    monkeypatch.setattr(dg, "_search_case_law", lambda *a, **k: [])
    monkeypatch.setattr(dg, "_search_similar_facts", lambda *a, **k: [])
    monkeypatch.setattr(dg, "_articles_cited_in_rulings", lambda *a, **k: {})
    monkeypatch.setattr(dg, "_resolve_institution", lambda *a, **k: [])
    monkeypatch.setattr(dg, "_fusion_distill_bullets", lambda case: [])

    import src.rag.doc_lookup as doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])

    def fake_fetch_any(session, documents, refs):
        return [{"d": {"id": "doc"}, "s": {"name": f"{n}.0.0", "plain_text": f"testo {n}"}, "_article": (n, c)}
                for n, c in refs]
    monkeypatch.setattr(dg, "_fetch_any", fake_fetch_any)

    import src.rag.nodes.retrieval as retrieval_module
    monkeypatch.setattr(retrieval_module, "broaden_article_context",
                        lambda state, driver, database: {"raw_result": state["raw_result"]})

    def boom(*a, **k):
        raise AssertionError("_choose_articles must not be called on the fusion path")
    monkeypatch.setattr(dg, "_choose_articles", boom)
    return monkeypatch


def test_fusion_never_calls_choose_articles(fusion_env):
    fusion_env.setattr(dg, "_candidate_articles", lambda *a, **k: ([], []))
    fusion_env.setattr(dg, "_fusion_text_search", lambda *a, **k: [])
    picked = dg._select_articles_fusion(object(), object(), "db", "facts", "", {"questioni": []})
    assert picked["selected"] == []


def test_keep_is_added_on_top_of_the_cap_not_counted_against_it(fusion_env):
    """"keep" (named by the user / ruling agreement) is never capped, even
    when the fusion pool alone already fills _FUSION_POOL_CAP."""
    keep = [("584", "c.p.")]
    fusion_env.setattr(dg, "_candidate_articles", lambda *a, **k: (keep, []))
    # 15 fusion candidates from raw-text search -- more than the cap (12).
    many = [(str(100 + i), "c.p.") for i in range(15)]
    fusion_env.setattr(dg, "_fusion_text_search", lambda *a, **k: many)
    picked = dg._select_articles_fusion(object(), object(), "db", "facts", "", {"questioni": []})
    assert picked["selected"][0] == ("584", "c.p.")              # keep, first
    assert len(picked["selected"]) == 1 + dg._FUSION_POOL_CAP    # keep + the cap, not keep + 15
    assert ("584", "c.p.") not in picked["selected"][1:]          # not duplicated into the capped part


def test_fusion_merges_institution_rawtext_and_distill(fusion_env, monkeypatch):
    """An article three sources agree on outranks one only a single source
    found, inside the capped pool."""
    fusion_env.setattr(dg, "_candidate_articles", lambda *a, **k: ([], []))
    monkeypatch.setattr(dg, "_resolve_institution",
                        lambda session, documents, istituto, parole_chiave=None: [
                            {"_article": ("589", "c.p.")}])
    calls = {"n": 0}

    def fake_text_search(session, text, index, k):
        calls["n"] += 1
        if calls["n"] == 1:   # raw-text search, on the case itself
            return [("589", "c.p."), ("43", "c.p.")]
        return []             # no bullets (distill returns none in fusion_env)
    fusion_env.setattr(dg, "_fusion_text_search", fake_text_search)

    analysis = {"questioni": [{"istituti": [{"nome": "omicidio colposo", "codice": "c.p."}],
                               "parole_chiave": []}]}
    picked = dg._select_articles_fusion(object(), object(), "db", "facts", "", analysis)
    # "589" (institution + raw-text) outranks "43" (raw-text only).
    assert picked["selected"].index(("589", "c.p.")) < picked["selected"].index(("43", "c.p."))
    assert picked["institutions_resolved"] == [("589", "c.p.")]


def test_fusion_distill_failure_does_not_lose_institution_or_rawtext_refs(fusion_env, monkeypatch):
    monkeypatch.setattr(dg, "_candidate_articles", lambda *a, **k: ([], []))
    monkeypatch.setattr(dg, "_resolve_institution",
                        lambda session, documents, istituto, parole_chiave=None: [
                            {"_article": ("589", "c.p.")}])
    monkeypatch.setattr(dg, "_fusion_text_search", lambda *a, **k: [("43", "c.p.")])

    def boom(case):
        raise RuntimeError("distill LLM call failed")
    monkeypatch.setattr(dg, "_fusion_distill_bullets", boom)

    analysis = {"questioni": [{"istituti": [{"nome": "omicidio colposo", "codice": "c.p."}],
                               "parole_chiave": []}]}
    picked = dg._select_articles_fusion(object(), object(), "db", "facts", "", analysis)
    assert set(picked["selected"]) == {("589", "c.p."), ("43", "c.p.")}


# --- run_case_analysis_pipeline: flag routing --------------------------------

def test_pipeline_uses_fusion_only_when_both_flags_are_on(monkeypatch):
    calls = []
    monkeypatch.setattr(dg, "_select_articles_fusion",
                        lambda *a, **k: calls.append("fusion") or {
                            "first_rulings": [], "similar": [], "keep": [], "named": [],
                            "candidates": [], "chosen": [], "selected": [], "article_rows": [], "spotted": []})
    monkeypatch.setattr(dg, "_select_articles_v2",
                        lambda *a, **k: calls.append("v2") or {
                            "first_rulings": [], "similar": [], "keep": [], "named": [],
                            "candidates": [], "chosen": [], "selected": [], "article_rows": [], "spotted": []})
    monkeypatch.setattr(dg, "_analyse_case_v2", lambda *a, **k: {"questioni": []})
    monkeypatch.setattr(dg, "_search_case_law", lambda *a, **k: [])
    monkeypatch.setattr(dg, "_ruling_passages", lambda *a, **k: [])
    monkeypatch.setattr(dg, "_reading_order", lambda *a, **k: [])
    monkeypatch.setattr(dg, "_extract_principles", lambda *a, **k: "")
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: "draft")

    import src.rag.main as main

    class _FakeSession:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    class _FakeDriver:
        def session(self, **kwargs):
            return _FakeSession()
    monkeypatch.setattr(main, "driver", _FakeDriver())
    monkeypatch.setattr(main, "NEO4J_DATABASE", "db")

    monkeypatch.delenv("CASE_ANALYSIS_ENABLED", raising=False)
    monkeypatch.delenv("CASE_POOL_FUSION_ENABLED", raising=False)
    dg.run_case_analysis_pipeline("facts", "it")
    assert calls == []   # neither flag on: v1's own _select_articles runs (not mocked here), not fusion/v2

    monkeypatch.setenv("CASE_POOL_FUSION_ENABLED", "true")   # fusion alone, without CASE_ANALYSIS_ENABLED
    dg.run_case_analysis_pipeline("facts", "it")
    assert calls == []   # still neither -- fusion requires CASE_ANALYSIS_ENABLED

    monkeypatch.setenv("CASE_ANALYSIS_ENABLED", "true")
    dg.run_case_analysis_pipeline("facts", "it")
    assert calls == ["fusion"]   # both flags on
