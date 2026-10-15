"""The legacy single-document lock (_dynamic_law_hint), behind LEGACY_LAW_HINT_FALLBACK.

On 15 Oct a narrative question with no document named (consumer c2, eviction
c3) was scoped to the one document whose name best matched its words, so
vector search and BM25 never saw the other code's articles. With the flag
false, only queries that name an article still get the guessed document (the
exact lookup needs the code); default true keeps today's behaviour.
"""

import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

from src.rag import feature_flags  # noqa: E402
from src.rag.nodes import retrieval  # noqa: E402

NARRATIVE = ("Il conduttore di un appartamento ad uso abitativo non paga il canone da tre mesi. Il locatore "
             "vuole riottenere l'immobile e recuperare i canoni arretrati. Quali strumenti processuali e "
             "sostanziali ha a disposizione?")
ARTICLE_QUERY = "art. 586 c.p. morte o lesioni come conseguenza di altro delitto"


@pytest.fixture
def guesses(monkeypatch):
    calls = []

    def fake_dynamic_law_hint(query, driver, database):
        calls.append(query)
        return "doc_guessed"

    monkeypatch.setattr(retrieval, "_dynamic_law_hint", fake_dynamic_law_hint)
    monkeypatch.delenv("LEGACY_LAW_HINT_FALLBACK", raising=False)
    return calls


def test_default_keeps_todays_guess(guesses):
    assert retrieval._law_hint_fallback(NARRATIVE, None, "neo4j") == "doc_guessed"
    assert guesses == [NARRATIVE]


def test_off_a_narrative_question_searches_every_document(guesses, monkeypatch):
    monkeypatch.setenv("LEGACY_LAW_HINT_FALLBACK", "false")
    assert retrieval._law_hint_fallback(NARRATIVE, None, "neo4j") == ""
    assert guesses == []


def test_off_a_query_naming_an_article_still_finds_its_code(guesses, monkeypatch):
    monkeypatch.setenv("LEGACY_LAW_HINT_FALLBACK", "false")
    assert retrieval._law_hint_fallback(ARTICLE_QUERY, None, "neo4j") == "doc_guessed"


@pytest.mark.parametrize("value, expected", [("maybe", True), ("", True), ("0", False), ("OFF", False), ("1", True)])
def test_unclear_values_fall_back_to_the_default(monkeypatch, value, expected):
    monkeypatch.setenv("LEGACY_LAW_HINT_FALLBACK", value)
    assert feature_flags.enabled("LEGACY_LAW_HINT_FALLBACK", True) is expected


def test_article_router_passes_no_lock_on_for_a_narrative_question(guesses, monkeypatch):
    """With no article in the question the router passes, and its law_hint_doc_id
    is what scopes context_retrieval's vector and BM25 search."""
    state = {"query": NARRATIVE, "keyword_article_refs": []}
    assert retrieval.article_router(dict(state), None, "neo4j")["law_hint_doc_id"] == "doc_guessed"
    monkeypatch.setenv("LEGACY_LAW_HINT_FALLBACK", "false")
    assert retrieval.article_router(dict(state), None, "neo4j")["law_hint_doc_id"] == ""
    # A document the intent classifier found stays, whatever the flag.
    explicit = {**state, "law_hint_doc_id": "doc_named_by_user"}
    assert retrieval.article_router(explicit, None, "neo4j")["law_hint_doc_id"] == "doc_named_by_user"


def test_all_three_fallbacks_go_through_the_flag():
    """article_router and context_retrieval's vector and BM25 scoping; the
    comparison path (explicitly named documents) keeps calling it directly."""
    import inspect
    source = inspect.getsource(retrieval)
    assert source.count("_law_hint_fallback(") == 4          # definition + 3 call sites
    assert source.count("_dynamic_law_hint(") == 2           # the wrapper's own call, the comparison path
