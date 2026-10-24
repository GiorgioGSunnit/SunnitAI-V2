"""VOTE_POOL_ENABLED (src/rag/vote_pool.py): the articles cited by the rulings most similar to the
case's facts, IDF-weighted, joined to the fusion pool. Offline: the database and the embeddings are faked."""

import json
import math
import os

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import pytest  # noqa: E402

from src.rag import vote_pool as vp  # noqa: E402

A, B, C, D, E = [(str(n), "c.p.") for n in (1, 2, 3, 4, 5)]


@pytest.mark.parametrize("text, expected", [
    ("append:6", ("append", 6)), ("APPEND:0", ("append", 0)), ("rrf:1:0.5", ("rrf", 1.0, 0.5)),
    (None, ("append", 6)), ("", ("append", 6)), ("rrf:1", ("append", 6)), ("append:x", ("append", 6)),
    ("rrf:-1:1", ("append", 6)),
])
def test_modes(text, expected):
    assert vp.parse_mode(text) == expected


def test_weighted_rrf_merges_by_rank_and_only_the_ratio_counts():
    pool, voted = [A, B, C], [C, D]
    assert vp.weighted_rrf(pool, voted, 1, 1)[0] == C          # in both lists
    assert vp.weighted_rrf(pool, voted, 1, 0.5) == vp.weighted_rrf(pool, voted, 2, 1)
    assert vp.weighted_rrf(pool, [], 1, 1) == pool


def test_append_puts_the_first_new_votes_after_the_head():
    pool, voted = [A, B, C, D], [B, E, C, (6, "c.c.")]
    assert vp.new_votes(voted, pool[:2], 2) == [E, C]
    assert vp.append_votes(pool, voted, 2, 2) == [A, B, E, C, D]
    assert vp.combine(pool, voted, ("append", 2), 2) == [A, B, E, C, D]
    assert vp.combine(pool, voted, ("rrf", 1, 1), 2) == vp.weighted_rrf(pool, voted, 1, 1)


def test_tagged_are_the_added_and_the_selected_among_the_votes_top():
    voted = [B, E, C, D]
    assert vp.tagged(voted, [A, B, C, E], [E], ("append", 2)) == {B, E}
    assert vp.tagged(voted, [A, D], [], ("rrf", 1, 1)) == {D}


def test_idf_from_counts_and_a_card_article_without_counts_weighs_as_cited_once():
    idf = vp.idf_from(1000, [["c.p.", "584", 10], ["c.p.p.", "606", 900], ["", "1", 3]])
    assert idf == {("584", "c.p."): math.log(100), ("606", "c.p.p."): math.log(1000 / 900)}
    sims = {"r1": 1.0, "r2": 0.5}
    assert vp.vote(sims, {"r1": {("584", "c.p."), ("606", "c.p.p.")}, "r2": {("606", "c.p.p.")}}, idf,
                   math.log(1000)) == [("584", "c.p."), ("606", "c.p.p.")]
    assert vp.vote({"r1": 1.0}, {"r1": [("1785-bis", "c.c.")]}, idf, math.log(1000)) == [("1785-bis", "c.c.")]


def test_ruling_sims_normalise_each_channel():
    sims = vp.ruling_sims([{"doc_id": "r1", "score": 0.8}, {"doc_id": "r2", "score": 0.4}],
                          [{"doc_id": "r2", "score": 12.0}, {"doc_id": "r3", "score": 3.0}, {"score": 9}])
    assert sims == {"r1": 1.0, "r2": 1.0, "r3": 0.25}


@pytest.fixture
def fresh(monkeypatch):
    monkeypatch.setattr(vp, "_file_cache", {"key": None, "table": None})
    monkeypatch.setattr(vp, "_computed", {"table": None, "started": False})
    monkeypatch.delenv(vp.IDF_FILE_ENV, raising=False)
    return monkeypatch


def test_idf_from_a_file_or_exp_distills_cache(fresh, tmp_path):
    path = tmp_path / "vote_cache.json"
    path.write_text(json.dumps({"_idf": {"n": 100, "df": [["c.p.", "584", 10]]}, "sempronio": {}}), encoding="utf-8")
    fresh.setenv(vp.IDF_FILE_ENV, str(path))
    idf, default = vp.idf_table(None, "db")
    assert idf == {("584", "c.p."): math.log(10)} and default == math.log(100)


def test_without_a_file_the_counts_are_made_once_in_the_background(fresh):
    started = []

    class FakeThread:
        def __init__(self, target, args, **kwargs):
            started.append(args)

        def start(self):
            pass
    fresh.setattr(vp.threading, "Thread", FakeThread)
    assert vp.idf_table("driver", "db") is None and vp.idf_table("driver", "db") is None
    assert started == [("driver", "db")]                       # one count, not one per request


def test_the_vote_for_a_case_cards_first_then_citations(fresh):
    fresh.setattr(vp, "idf_table", lambda driver, database: ({("584", "c.p."): 2.0, ("43", "c.p."): 1.0}, 3.0))
    fresh.setattr(vp, "_similar_rulings", lambda session, text, index, k: (
        [{"doc_id": "r1", "score": 0.9}, {"doc_id": "r2", "score": 0.45}], []))
    fresh.setattr(vp, "_citations", lambda session, ids: {"r1": {("43", "c.p.")}, "r2": {("43", "c.p.")}})
    fresh.setattr(vp, "_card_refs", lambda: {"r1": [("584", "c.p.")]})
    logged = []
    fresh.setattr(vp, "vlog", lambda event, data=None, duration_ms=None: logged.append((event, data)))
    assert vp.ruling_vote(object(), object(), "db", "fatti", "idx") == [("584", "c.p."), ("43", "c.p.")]
    event, data = logged[-1]
    assert event == "case_pool_vote" and data["rulings"] == 2 and data["carded"] == 1


def test_the_vote_never_breaks_the_pool(fresh):
    fresh.setattr(vp, "idf_table", lambda driver, database: ({}, 1.0))
    fresh.setattr(vp, "_similar_rulings", lambda *a: (_ for _ in ()).throw(RuntimeError("index down")))
    fresh.setattr(vp, "vlog", lambda *a, **k: None)
    assert vp.ruling_vote(object(), object(), "db", "fatti", "idx") == []


# --- the filter: other areas' codes and Cassation boilerplate (2c dry run, 24 Oct) -------------------

@pytest.mark.parametrize("text, codes", [
    ("Il PM contesta a Sempronio l'art 584 cp.", {"c.p."}),
    ("ai sensi dell'art. 2051 c.c. e dell'art. 696-bis c.p.c.", {"c.c.", "c.p.c."}),
    ("il codice di procedura penale prevede", {"c.p.p."}),
    ("ricorso ex art. 29 c.p.a.", {"c.p.a."}),
    ("Tizio cade dalle scale", set()),
])
def test_named_codes(text, codes):
    assert vp.named_codes(text) == codes


def test_case_areas_from_the_frame_else_the_analysis():
    assert vp.case_areas({"frame": {"areas": ["penale", "responsabilita_civile"]}, "area": "penale"}) == \
        ["penale", "responsabilita_civile"]
    assert vp.case_areas({"area": "civile"}) == ["civile"]
    assert vp.case_areas({}) == [] and vp.case_areas(None) == []


def test_the_filter_drops_other_areas_and_boilerplate():
    votes = {"r1": [("584", "c.p."), ("360", "c.p.c."), ("2043", "c.c."), ("606", "c.p.p."), ("42", "Cost."),
                    ("111", "Cost.")]}
    kept, dropped = vp.filter_refs(votes, ["penale"], "giudizio_penale_primo_grado", "Tizio colpisce Caio")
    assert kept == {"r1": [("584", "c.p."), ("42", "Cost.")]}
    assert set(dropped["boilerplate"]) == {("360", "c.p.c."), ("606", "c.p.p."), ("111", "Cost.")}
    assert dropped["other_area"] == [("2043", "c.c.")]


def test_the_filter_keeps_a_named_code_and_boilerplate_at_the_cassation():
    votes = {"r1": [("2043", "c.c."), ("606", "c.p.p."), ("224", "c.p.")]}
    kept, _ = vp.filter_refs(votes, ["penale"], "sentenza_penale_appello", "danno ex art. 2043 c.c.")
    assert kept == {"r1": [("2043", "c.c."), ("606", "c.p.p."), ("224", "c.p.")]}
    kept, _ = vp.filter_refs(votes, ["responsabilita_civile"], "precontenzioso", "Tizio cade")
    assert kept == {"r1": [("2043", "c.c.")]}     # 606 c.p.p. boilerplate, 224 c.p. another area


def test_no_known_area_drops_only_boilerplate():
    kept, _ = vp.filter_refs({"r1": [("2043", "c.c."), ("584", "c.p."), ("2697", "c.c.")]}, [], "", "")
    assert kept == {"r1": [("2043", "c.c."), ("584", "c.p.")]}
