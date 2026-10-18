"""CASE_POOL_FUSION_ENABLED (round 6, 19 Oct; requires CASE_ANALYSIS_ENABLED):
_select_articles_fusion adds a ranked pool on top of _select_articles_v2's
own selection: CURRENT (v2's full "selected") union BASELINE (raw-text
search) union DISTILL (distilled-bullet search), ranked by fusion (how many
of the three sources agree, then best rank) and capped at _FUSION_POOL_CAP.

Round 8 (19 Oct): the first version of this function used institution-
resolution alone for CURRENT (skipping _select_articles_v2 and its own
_choose_articles call) -- a simplification scripts/exp_distill.py's own
measured FUSION_BASELINE numbers never tested, and the cause of a prod
parity bug (check_applicability.py --fusion ranked a textbook and three
wrong-code institution matches into Sempronio's top 12, where exp_distill.py
measured Sempronio at 5/5 @12). _select_articles_fusion now calls
_select_articles_v2 itself for CURRENT, so these tests mock _select_
articles_v2 directly (not v1's own _search_case_law/_candidate_articles/
_resolve_institution/_choose_articles, which have their own tests in
test_case_analysis_v2.py) and isolate fusion's own logic: pulling CURRENT/
keep/named/institutions_resolved from v2, merging it with BASELINE/DISTILL,
capping, keeping "keep" uncapped, and seeding cross-reference broadening
only from named + institution-resolved articles.
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


def test_row_to_fusion_ref_rejects_a_book_or_manual_by_name():
    """Round 8, 19 Oct: "document_type = 'primary'" alone let a textbook
    ("Manuale di diritto processuale penale") into a prod ranked pool --
    its document_type property is apparently mislabelled. Reject by name
    too, the same words issue_spotting._NOT_A_LAW rejects for resolve_code."""
    row = {"s": {"name": "176.0.0"}, "d": {"name": "Manuale di diritto processuale penale"}}
    assert dg._row_to_fusion_ref(row) is None


def test_fusion_best_rank_merge_promotes_an_item_fulltext_ranks_better():
    """Round 8, 19 Oct: _fusion_text_search used to concatenate vector's
    refs entirely ahead of fulltext's own ranking, regardless of what
    fulltext found -- "584" (vector rank 3) stayed last. A true best-rank
    merge promotes it ahead of "582" (vector rank 2, not found by fulltext
    at all) once fulltext ranks it #1."""
    vector = [("c.p.", "581"), ("c.p.", "582"), ("c.p.", "584")]
    fulltext = [("c.p.", "584")]
    assert dg._fusion_best_rank_merge(vector, fulltext) == [("c.p.", "581"), ("c.p.", "584"), ("c.p.", "582")]


def test_fusion_merge_by_hits_prefers_more_agreeing_sources():
    current = [("c.p.", "584")]
    baseline = [("c.p.", "589"), ("c.p.", "584")]
    distill = [("c.p.", "584")]
    # "584" is in all three sources; "589" only in one.
    assert dg._fusion_merge_by_hits(current, baseline, distill) == [("c.p.", "584"), ("c.p.", "589")]


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

def _v2_result(selected=(), keep=(), named=(), institutions_resolved=(), spotted=()):
    return {"first_rulings": [], "similar": [], "keep": list(keep), "named": list(named),
            "candidates": [], "chosen": [], "selected": list(selected), "article_rows": [],
            "spotted": list(spotted), "institutions_resolved": list(institutions_resolved)}


@pytest.fixture
def fusion_env(monkeypatch):
    monkeypatch.setattr(dg, "_select_articles_v2", lambda *a, **k: _v2_result())
    monkeypatch.setattr(dg, "_fusion_distill_bullets", lambda case: [])
    monkeypatch.setattr(dg, "_fusion_text_search", lambda *a, **k: [])

    import src.rag.doc_lookup as doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])

    def fake_fetch_any(session, documents, refs):
        return [{"d": {"id": "doc"}, "s": {"name": f"{n}.0.0", "plain_text": f"testo {n}"}, "_article": (n, c)}
                for n, c in refs]
    monkeypatch.setattr(dg, "_fetch_any", fake_fetch_any)

    import src.rag.nodes.retrieval as retrieval_module
    monkeypatch.setattr(retrieval_module, "broaden_article_context",
                        lambda state, driver, database: {"raw_result": state["raw_result"]})
    return monkeypatch


def test_fusion_pulls_current_directly_from_select_articles_v2(fusion_env):
    """CURRENT is _select_articles_v2's own "selected" -- not a
    reimplementation of institution-resolution or v1's candidate search."""
    fusion_env.setattr(dg, "_select_articles_v2",
                       lambda *a, **k: _v2_result(selected=[("584", "c.p.")]))
    picked = dg._select_articles_fusion(object(), object(), "db", "facts", "", {"questioni": []})
    assert picked["selected"] == [("584", "c.p.")]


def test_keep_is_added_on_top_of_the_cap_not_counted_against_it(fusion_env):
    """"keep" (named by the user / ruling agreement, from v2) is never
    capped, even when the fusion pool alone already fills _FUSION_POOL_CAP."""
    keep = [("584", "c.p.")]
    fusion_env.setattr(dg, "_select_articles_v2", lambda *a, **k: _v2_result(keep=keep))
    # 15 baseline candidates -- more than the cap (12).
    many = [(str(100 + i), "c.p.") for i in range(15)]
    fusion_env.setattr(dg, "_fusion_text_search", lambda *a, **k: many)
    picked = dg._select_articles_fusion(object(), object(), "db", "facts", "", {"questioni": []})
    assert picked["selected"][0] == ("584", "c.p.")              # keep, first
    assert len(picked["selected"]) == 1 + dg._FUSION_POOL_CAP    # keep + the cap, not keep + 15
    assert ("584", "c.p.") not in picked["selected"][1:]          # not duplicated into the capped part


def test_fusion_merges_current_baseline_and_distill(fusion_env, monkeypatch):
    """An article several sources agree on outranks one only a single
    source found, inside the capped pool."""
    fusion_env.setattr(dg, "_select_articles_v2",
                       lambda *a, **k: _v2_result(selected=[("589", "c.p.")]))
    calls = {"n": 0}

    def fake_text_search(session, text, index, k):
        calls["n"] += 1
        if calls["n"] == 1:   # BASELINE, on the case itself
            return [("589", "c.p."), ("43", "c.p.")]
        return []             # no bullets (distill returns none in fusion_env)
    fusion_env.setattr(dg, "_fusion_text_search", fake_text_search)

    picked = dg._select_articles_fusion(object(), object(), "db", "facts", "", {"questioni": []})
    # "589" (CURRENT + BASELINE) outranks "43" (BASELINE only).
    assert picked["selected"].index(("589", "c.p.")) < picked["selected"].index(("43", "c.p."))


def test_fusion_distill_failure_does_not_lose_current_or_baseline_refs(fusion_env, monkeypatch):
    fusion_env.setattr(dg, "_select_articles_v2",
                       lambda *a, **k: _v2_result(selected=[("589", "c.p.")]))
    fusion_env.setattr(dg, "_fusion_text_search", lambda *a, **k: [("43", "c.p.")])

    def boom(case):
        raise RuntimeError("distill LLM call failed")
    monkeypatch.setattr(dg, "_fusion_distill_bullets", boom)

    picked = dg._select_articles_fusion(object(), object(), "db", "facts", "", {"questioni": []})
    assert set(picked["selected"]) == {("589", "c.p."), ("43", "c.p.")}


def test_crossref_expands_only_from_named_and_institution_resolved_articles(fusion_env, monkeypatch):
    """Round 8, 19 Oct: broadening from the whole capped pool pulled in
    cross-references with no connection to what fusion specifically added
    (1, 3, 444, 450, 470 c.p.p. on Sempronio). Only user-named and
    institution-resolved articles seed it -- the same rule _select_
    articles_v2 already applies to its own institution rows."""
    fusion_env.setattr(dg, "_select_articles_v2", lambda *a, **k: _v2_result(
        selected=[("584", "c.p."), ("999", "c.p.")], named=[("584", "c.p.")],
        institutions_resolved=[]))

    seen_seed_articles = []

    def fake_broaden(state, driver, database):
        seen_seed_articles.extend(r["_article"] for r in state["raw_result"])
        return {"raw_result": state["raw_result"]}
    import src.rag.nodes.retrieval as retrieval_module
    monkeypatch.setattr(retrieval_module, "broaden_article_context", fake_broaden)

    dg._select_articles_fusion(object(), object(), "db", "facts", "", {"questioni": []})
    assert seen_seed_articles == [("584", "c.p.")]   # not "999", which v2 didn't name or resolve


def test_spotted_rows_from_v2_are_carried_over_without_a_second_issue_spotting_call(fusion_env):
    """v2's own base (_select_articles) already runs issue spotting once;
    fusion reuses its "spotted" rows instead of calling spotted_rows again
    on its own (differently built) article_rows."""
    spotted_row = {"d": {"id": "consumo-doc"}, "s": {"name": "130.0.0", "plain_text": "testo 130"},
                   "_article": ("130", "Codice del Consumo")}
    fusion_env.setattr(dg, "_select_articles_v2", lambda *a, **k: _v2_result(spotted=[spotted_row]))
    picked = dg._select_articles_fusion(object(), object(), "db", "facts", "", {"questioni": []})
    assert spotted_row in picked["article_rows"]
    assert picked["spotted"] == [spotted_row]


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
