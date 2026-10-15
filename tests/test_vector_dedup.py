"""Duplicate Section copies in vector results (text mode only).

The corpus holds exact duplicate Section nodes: same document, name and text,
different node, up to 7 copies. With working (plain-text) vectors the copies
score identically and sit together, so the top 30 could hold 4 or 5 different
articles and the 5 context slots one article several times. In text mode,
vector search keeps one node per (document id, section name); token-ID mode is
left exactly as it was.
"""

import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.embedding_config as ec  # noqa: E402
import src.rag.lookups as lookups  # noqa: E402
from src.rag.nodes import retrieval  # noqa: E402

QUESTION = "Il Comune risponde della buca sulla strada?"

# The token-ID index query, character for character as it was before the change.
TOKEN_QUERY = (
    "CALL db.index.vector.queryNodes($index, $k, $embedding) "
    "YIELD node, score "
    "WHERE NOT ("
    "  ('LegalAct' IN labels(node) AND (node.name IS NULL OR node.name = '' OR node.name IN ['string', 'Fonte non classificata']))"
    "  OR ('Section' IN labels(node) AND (node.plain_text IS NULL OR node.plain_text = ''))"
    ") "
    "RETURN elementId(node) AS element_id, labels(node) AS labels, score ORDER BY score DESC"
)


def _row(eid, score, doc="cc", name=None):
    return {"element_id": eid, "labels": ["Section"], "score": score, "doc_id": doc, "name": name}


# Article 2051 three times (two copies at the top), then 1227 twice, then 2043.
ROWS = [_row("a1", 0.90, name="2051.0.0"), _row("a2", 0.90, name="2051.0.0"), _row("a3", 0.89, name="2051.0.0"),
        _row("b1", 0.80, name="1227.0.0"), _row("b2", 0.80, name="1227.0.0"),
        _row("c1", 0.70, name="2043.0.0")]


class _Session:
    def __init__(self, rows):
        self.rows, self.calls = list(rows), []

    def run(self, query, **params):
        self.calls.append((query, params))
        return list(self.rows)


@pytest.fixture
def mode(monkeypatch):
    def set_mode(query_mode):
        monkeypatch.setattr(ec, "QUERY_MODE", query_mode)
        monkeypatch.setattr(ec, "INDEX_SUFFIX", "")
        monkeypatch.setattr(ec, "VECTOR_PROPERTY", "embedding")
    monkeypatch.setattr(lookups, "_cached_embed_query", lambda text: (0.1, 0.2))
    return set_mode


def _search(rows, index="section_embeddings", **kwargs):
    session = _Session(rows)
    kwargs.setdefault("index_settings", lookups.VECTOR_INDEX_SETTINGS)
    return session, lookups.vector_lookup(session, QUESTION, indexes=[index], **kwargs)


# --- One node per section in a search ------------------------------------------

def test_duplicate_copies_collapse_to_the_best_one(mode, monkeypatch):
    mode("text")
    logged = []
    monkeypatch.setattr(lookups, "vlog", lambda event, data=None, **k: logged.append(data))
    session, matches = _search(ROWS)
    assert [m["element_id"] for m in matches] == ["a1", "b1", "c1"]   # first, best copy of each
    assert [m["score"] for m in matches] == [0.90, 0.80, 0.70]
    assert matches[0]["dedup_key"] == ("cc", "2051.0.0")
    query, params = session.calls[0]
    assert "OPTIONAL MATCH (d:Document)-[:CONTAINS]->(node)" in query and "node.name AS name" in query
    assert params["k"] == 240                                           # 30 x 8 candidates
    assert logged[-1]["duplicates_dropped"] == 3 and logged[-1]["fetched"] == 6


def test_k_different_sections_when_the_candidates_allow(mode):
    """Commentary keeps 3: twenty-four candidates, four copies each of four sections."""
    mode("text")
    rows = [_row(f"{art}{i}", 0.9 - n / 10, doc="cpc", name=art) for n, art in enumerate("WXYZ") for i in range(4)]
    session, matches = _search(rows, index="commentato_section_embeddings", k=3)
    assert session.calls[0][1]["k"] == 24
    assert [m["element_id"] for m in matches] == ["W0", "X0", "Y0"]


def test_over_fetch_is_capped(mode, monkeypatch):
    mode("text")
    monkeypatch.setitem(ec.TEXT_MODE_INDEX_SETTINGS, "section_embeddings", {"k": 60, "min_score": 0.55})
    session, _ = _search(ROWS)
    assert session.calls[0][1]["k"] == 240


def test_nodes_without_a_key_are_kept(mode):
    mode("text")
    rows = [_row("n1", 0.9, doc=None, name="2051.0.0"), _row("n2", 0.9, doc=None, name="2051.0.0"),
            _row("n3", 0.8, doc="cc", name=None), _row("n4", 0.8, doc="cc", name="")]
    _, matches = _search(rows)
    assert [m["element_id"] for m in matches] == ["n1", "n2", "n3", "n4"]
    assert all(m["dedup_key"] is None for m in matches)


def test_document_scoped_search_also_keeps_one_per_section(mode):
    mode("text")
    session, matches = _search(ROWS, doc_id="cc")
    query, params = session.calls[0]
    assert "d.id AS doc_id, s.name AS name" in query and params["k"] == 240
    assert params["min_score"] == 0.55
    assert [m["element_id"] for m in matches] == ["a1", "b1", "c1"]
    assert all(m["source"] == "vector:section_embeddings:doc_scoped" for m in matches)


def test_below_threshold_copies_are_dropped_before_counting(mode):
    mode("text")
    rows = ROWS + [_row("d1", 0.50, name="2697.0.0")]                   # under 0.55
    _, matches = _search(rows)
    assert "d1" not in [m["element_id"] for m in matches]


def test_indexes_that_are_not_migrated_are_not_deduplicated(mode):
    mode("text")
    session, matches = _search(ROWS, index="legalact_embeddings")
    assert session.calls[0][0] == TOKEN_QUERY and session.calls[0][1]["k"] == 2
    assert len(matches) == 6                                             # today's k=2 is not applied by the fake


# --- Token-ID mode: exactly as before ------------------------------------------------

def test_token_mode_is_unchanged(mode):
    mode("token_ids")
    session, matches = _search(ROWS)
    query, params = session.calls[0]
    assert query == TOKEN_QUERY and params["k"] == 30
    assert [m["element_id"] for m in matches] == ["a1", "a2", "a3", "b1", "b2", "c1"]   # copies kept
    assert all(set(m) == {"element_id", "labels", "source", "score"} for m in matches)
    assert lookups.merge_vector_matches(matches) is matches


# --- Merging the searches of several query variants ------------------------------------------

def _match(eid, score, source, key):
    return {"element_id": eid, "labels": ["Section"], "source": source, "score": score, "dedup_key": key}


def test_the_variant_merge_has_no_duplicate_sections(mode):
    mode("text")
    k2051, k1227 = ("cc", "2051.0.0"), ("cc", "1227.0.0")
    matches = [
        _match("a1", 0.85, "context:section_embeddings", k2051),
        _match("b1", 0.80, "context:section_embeddings", k1227),
        _match("a2", 0.91, "context_variant_1:section_embeddings", k2051),   # another copy, better score
        _match("b1", 0.82, "context_variant_2:section_embeddings", k1227),   # same node, other variant
        _match("x1", 0.70, "context:section_embeddings", None),
        _match("x1", 0.75, "context_variant_1:section_embeddings", None),     # keyless: merged with itself only
    ]
    merged = lookups.merge_vector_matches(matches)
    assert [(m["element_id"], m["score"]) for m in merged] == [("a2", 0.91), ("b1", 0.82), ("x1", 0.75)]
    assert merged[0]["sources"] == ["context:section_embeddings", "context_variant_1:section_embeddings"]

    # The context nodes: a copy of 2051 found by the article lookup joins the same node.
    art_row = {"s_eid": "a3", "d": {"id": "cc"}, "s": {"name": "2051.0.0"}}
    nodes = retrieval._aggregate_context_matches(matches, [art_row])
    assert [n["element_id"] for n in nodes] == ["a2", "b1", "x1"]
    assert nodes[0]["score"] == 999.0 and "bm25" in nodes[0]["sources"]     # bm25 kept for the override
    assert len({n["element_id"] for n in nodes}) == len(nodes)


def test_token_mode_aggregation_is_unchanged(mode):
    mode("token_ids")
    matches = [{"element_id": "a1", "labels": ["Section"], "source": "context:s", "score": 0.85},
               {"element_id": "a2", "labels": ["Section"], "source": "context:s", "score": 0.84},
               {"element_id": "a1", "labels": ["Section"], "source": "context_variant_1:s", "score": 0.9}]
    art_row = {"s_eid": "a3", "d": {"id": "cc"}, "s": {"name": "2051.0.0"}}
    nodes = retrieval._aggregate_context_matches(matches, [art_row])
    assert [(n["element_id"], n["score"], n["sources"]) for n in nodes] == [
        ("a3", 999.0, ["bm25"]),
        ("a1", 0.9, ["context:s", "context_variant_1:s"]),
        ("a2", 0.84, ["context:s"]),
    ]
