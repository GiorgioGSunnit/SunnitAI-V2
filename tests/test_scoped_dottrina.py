"""Commentary scoped to the resolved law (round 3, item 5).

With a law hint resolved, the dottrina search was skipped. With
SCOPED_DOTTRINA_ENABLED, the resolved law's own commentary (Codice Penale ->
Codice Penale Commentato ADE, fiscal law -> La Fiscalità delle Società) is
searched, and only it: the known article's passages first, else BM25 and
vector within that commentary. Off by default: today's skip.
"""

import inspect
import os
import re

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

from src.rag.nodes import retrieval  # noqa: E402

LAWS = {"cp": "Codice Penale 2026", "cc": "Codice Civile 2026", "tuir": "TUIR - Testo unico delle imposte sui redditi"}
SPECIALS = [{"id": "comm", "name": "Codice Penale Commentato ADE"},
            {"id": "fisc", "name": "La Fiscalità delle Società IAS IFRS"}]


def _row(doc, sid, text="Commento sulla proporzione nella legittima difesa."):
    return {"d": {"id": doc, "name": "x"}, "s": {"id": sid, "name": sid, "plain_text": text}}


class _Single:
    def __init__(self, value):
        self.value = value

    def single(self):
        return self.value


class _Data:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        return [dict(r) for r in self.rows]


class _Session:
    def __init__(self, article_rows=(), bm25_rows=(), vector_rows=()):
        self.article_rows, self.bm25_rows, self.vector_rows = article_rows, bm25_rows, vector_rows
        self.calls = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.calls.append((query, params))
        if "MATCH (d:Document {id: $id}) RETURN d.name" in query:
            return _Single({"name": LAWS.get(params["id"], "")})
        if "document_type = 'special'" in query:
            return _Data(SPECIALS)
        if "$name_re" in query:
            return _Data(self.article_rows)
        if "section_fulltext" in query:
            return _Data(self.bm25_rows)
        if "elementId(s) IN $eids" in query:
            return _Data(self.vector_rows)
        return _Data([])


class _Driver:
    def __init__(self, session):
        self._session = session

    def session(self, **config):
        return self._session


@pytest.fixture
def vectors(monkeypatch):
    asked = []

    def fake_vector_lookup(session, value, *, indexes, k, source_prefix, **kw):
        asked.append(indexes)
        return [{"element_id": "v1"}, {"element_id": "v2"}]

    monkeypatch.setattr(retrieval, "vector_lookup", fake_vector_lookup)
    return asked


def test_off_by_default_the_search_is_skipped_as_today(monkeypatch, vectors):
    monkeypatch.delenv("SCOPED_DOTTRINA_ENABLED", raising=False)
    session = _Session()
    assert retrieval.dottrina_search({"query": "art. 52 c.p.", "law_hint_doc_id": "cp"}, _Driver(session), "neo4j") == {}
    assert session.calls == [] and vectors == []


def test_a_known_article_gets_its_own_commentary_passages(monkeypatch, vectors):
    monkeypatch.setenv("SCOPED_DOTTRINA_ENABLED", "true")
    session = _Session(article_rows=[_row("comm", "c52")])
    out = retrieval.dottrina_search(
        {"query": "art. 52 c.p. legittima difesa: requisiti della proporzione", "law_hint_doc_id": "cp",
         "retrieval_keywords": ["legittima difesa"]}, _Driver(session), "neo4j")
    rows = out["special_dottrina_rows"]
    assert [(r["s"]["id"], r["_source"]) for r in rows] == [("c52", "special_dottrina_article")]
    article_query = next(p for q, p in session.calls if "$name_re" in q)
    assert article_query["ids"] == ["comm"]                       # the penal commentary only, not the tax book
    assert re.fullmatch(article_query["name_re"], "Art. 52")
    assert not any("section_fulltext" in q for q, _ in session.calls) and vectors == []


def test_without_an_article_the_search_stays_inside_the_commentary(monkeypatch, vectors):
    monkeypatch.setenv("SCOPED_DOTTRINA_ENABLED", "true")
    session = _Session(bm25_rows=[_row("comm", "b1"), _row("comm", "b2", text="  ")],   # no text: skipped
                       vector_rows=[_row("comm", "v1"), _row("comm", "b1")])            # b1 twice: once
    out = retrieval.dottrina_search(
        {"query": "requisiti della proporzione nella legittima difesa", "law_hint_doc_id": "cp",
         "retrieval_keywords": ["proporzione", "legittima difesa"],
         "raw_result": [{"s": {"id": "already"}}]}, _Driver(session), "neo4j")
    assert [(r["s"]["id"], r["_source"]) for r in out["special_dottrina_rows"]] == [
        ("b1", "special_dottrina_bm25"), ("v1", "special_dottrina")]
    assert vectors == [["commentato_section_embeddings"]]
    fetch = next(p for q, p in session.calls if "elementId(s) IN $eids" in q)
    assert fetch["ids"] == ["comm"] and fetch["eids"] == ["v1", "v2"]


def test_a_fiscal_law_gets_the_tax_commentary(monkeypatch, vectors):
    monkeypatch.setenv("SCOPED_DOTTRINA_ENABLED", "true")
    session = _Session(vector_rows=[_row("fisc", "f1")])
    out = retrieval.dottrina_search({"query": "deducibilità degli interessi passivi", "law_hint_doc_id": "tuir"},
                                    _Driver(session), "neo4j")
    assert [r["s"]["id"] for r in out["special_dottrina_rows"]] == ["f1"]
    assert vectors == [["fiscalita_section_embeddings"]]


def test_a_law_without_a_commentary_gets_none(monkeypatch, vectors):
    monkeypatch.setenv("SCOPED_DOTTRINA_ENABLED", "true")
    session = _Session(bm25_rows=[_row("comm", "b1")])
    assert retrieval.dottrina_search({"query": "art. 1418 c.c. nullità", "law_hint_doc_id": "cc"},
                                     _Driver(session), "neo4j") == {}
    assert vectors == []


@pytest.mark.parametrize("number, name, text, name_hit, text_hit", [
    ("52", "Art. 52", "secondo l'art. 52 c.p. la difesa", True, True),
    ("52", "52.1", "come previsto dall'articolo 52, comma 2", True, True),
    ("52", "520", "l'art. 520 c.p.", False, False),
    ("52", "52-bis", "l'art. 52-bis", False, False),
    ("624-bis", "Art. 624-bis", "il furto con strappo (art. 624 bis c.p.)", True, True),
])
def test_article_patterns(number, name, text, name_hit, text_hit):
    name_re, text_re = retrieval._commentary_article_patterns(number)
    assert bool(re.fullmatch(name_re, name)) is name_hit       # Neo4j =~ is a full match
    assert bool(re.fullmatch(text_re, text)) is text_hit


def test_both_skips_are_replaced_behind_the_flag():
    source = inspect.getsource(retrieval)
    assert source.count("_scoped_dottrina_rows(") == 3          # definition + context_retrieval + dottrina_search
    assert source.count('feature_flags.enabled("SCOPED_DOTTRINA_ENABLED")') == 2
