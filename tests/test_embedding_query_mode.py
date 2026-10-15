"""Question embedding and vector search, before and after the embedding fix.

Oct 2026: LangChain's OpenAIEmbeddings defaults send OpenAI (cl100k) token IDs
to the Qwen embedding server, which reads them with its own vocabulary: every
question and stored vector was embedded from gibberish, and the right section
was in the real index's top 30 for 0 of 400 test questions. The fix (plain text
plus Qwen's query instruction) is switched on by configuration during the
rollout, together with the re-embedded vectors. These tests pin that the
defaults change nothing, and what each switch changes.
"""

import importlib.util
import logging
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("OPENAI_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

from neo4j.exceptions import Neo4jError  # noqa: E402

import src.preprocessing.generate_embeddings as gen  # noqa: E402
import src.rag.ai_chat as ai_chat  # noqa: E402
import src.rag.embedding_config as ec  # noqa: E402
import src.rag.lookups as lookups  # noqa: E402
from src.rag.lookup_indexes import CONTEXT_VECTOR_INDEXES, ENTITY_VECTOR_INDEXES  # noqa: E402

QUESTION = "Quali sono le cause di nullità del contratto?"
_ENV_VARS = ("EMBEDDING_QUERY_MODE", "EMBEDDING_INDEX_SUFFIX", "EMBEDDING_PROPERTY",
             "EMBEDDING_MIN_SCORE_SECTION", "EMBEDDING_MIN_SCORE_COMMENTATO", "EMBEDDING_MIN_SCORE_FISCALITA")


def _fresh_config(monkeypatch, dotenv=None, **env):
    """A separate copy of embedding_config read with exactly this env, so the
    module the app uses is not touched. `dotenv`: what its load_dotenv() call
    finds in .env (nothing by default, so a developer's .env can't interfere)."""
    for name in _ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr("dotenv.load_dotenv",
                        lambda *a, **k: [monkeypatch.setenv(n, v) for n, v in (dotenv or {}).items()])
    spec = importlib.util.spec_from_file_location("embedding_config_under_test", ec.__file__)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def mode(monkeypatch):
    """Set the configuration the app reads at call time; undone after the test."""
    def set_mode(query_mode="token_ids", suffix="", prop="embedding"):
        monkeypatch.setattr(ec, "QUERY_MODE", query_mode)
        monkeypatch.setattr(ec, "INDEX_SUFFIX", suffix)
        monkeypatch.setattr(ec, "VECTOR_PROPERTY", prop)
    set_mode()
    return set_mode


# --- Reading the configuration ---------------------------------------------

def test_with_no_new_env_vars_the_configuration_is_todays(monkeypatch):
    cfg = _fresh_config(monkeypatch)
    assert cfg.QUERY_MODE == "token_ids"
    assert cfg.INDEX_SUFFIX == "" and cfg.VECTOR_PROPERTY == "embedding"
    assert cfg.vector_index_name("section_embeddings") == "section_embeddings"
    assert cfg.openai_embedding_kwargs() == {"timeout": 60}         # no check_embedding_ctx_length


def test_the_switches_are_read_from_the_env(monkeypatch):
    cfg = _fresh_config(monkeypatch, EMBEDDING_QUERY_MODE="TEXT", EMBEDDING_INDEX_SUFFIX="_v2",
                        EMBEDDING_PROPERTY="embedding_v2", EMBEDDING_MIN_SCORE_SECTION="0.6")
    assert cfg.QUERY_MODE == "text"
    assert cfg.openai_embedding_kwargs() == {
        "timeout": 60, "check_embedding_ctx_length": False,
        "model_kwargs": {"extra_body": {"truncate_prompt_tokens": 8000}}}
    # Only the three migrated indexes get the suffix.
    assert [cfg.vector_index_name(i) for i in ("section_embeddings", "commentato_section_embeddings",
                                               "fiscalita_section_embeddings")] == [
        "section_embeddings_v2", "commentato_section_embeddings_v2", "fiscalita_section_embeddings_v2"]
    assert cfg.vector_index_name("document_embeddings") == "document_embeddings"
    assert cfg.vector_index_name("ccnl_section_embeddings") == "ccnl_section_embeddings"
    assert cfg.VECTOR_PROPERTY == "embedding_v2"
    assert cfg.TEXT_MODE_INDEX_SETTINGS["section_embeddings"] == {"k": 30, "min_score": 0.6}
    assert cfg.TEXT_MODE_INDEX_SETTINGS["commentato_section_embeddings"] == {"k": 3, "min_score": 0.55}


def test_the_config_loads_dotenv_itself(monkeypatch):
    """lookups.py imports embedding_config before ai_chat (which also loads
    .env): a mode set only in .env must not depend on that import order."""
    cfg = _fresh_config(monkeypatch, dotenv={"EMBEDDING_QUERY_MODE": "text"})
    assert cfg.QUERY_MODE == "text"


def test_invalid_values_fall_back_to_todays_behaviour(monkeypatch):
    """The property name is written into Cypher: anything but an identifier is refused."""
    cfg = _fresh_config(monkeypatch, EMBEDDING_QUERY_MODE="plain", EMBEDDING_INDEX_SUFFIX="-v2",
                        EMBEDDING_PROPERTY="embedding) DETACH DELETE (s", EMBEDDING_MIN_SCORE_SECTION="high")
    assert (cfg.QUERY_MODE, cfg.INDEX_SUFFIX, cfg.VECTOR_PROPERTY) == ("token_ids", "", "embedding")
    assert cfg.TEXT_MODE_INDEX_SETTINGS["section_embeddings"]["min_score"] == 0.55


def test_the_hard_coded_v2_names_are_gone():
    """bfc7def pointed every section search at section_embeddings_v2, an index
    that doesn't exist: each search would have failed and fallen back to BM25."""
    assert lookups.VECTOR_INDEX_SETTINGS["section_embeddings"] == {"k": 30, "min_score": 0.25}
    assert lookups.LABEL_VECTOR_HINTS["Section"] == ["section_embeddings"]
    assert "section_embeddings" in CONTEXT_VECTOR_INDEXES and "section_embeddings" in ENTITY_VECTOR_INDEXES
    assert not [i for i in CONTEXT_VECTOR_INDEXES + ENTITY_VECTOR_INDEXES if i.endswith("_v2")]


# --- The question client -----------------------------------------------------

class _RecordingClient:
    """Stands in for openai's embeddings client and records what would be sent."""

    def __init__(self):
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        n = len(kwargs["input"])
        return {"data": [{"embedding": [0.1, 0.2], "index": i} for i in range(n)], "usage": {}}


def _client_model(monkeypatch, query_mode):
    monkeypatch.setattr(ec, "QUERY_MODE", query_mode)
    model = ai_chat._init_openai_embeddings()
    recorder = _RecordingClient()
    object.__setattr__(model, "client", recorder)
    monkeypatch.setattr(ai_chat, "embedding_model", model)
    return model, recorder


def test_text_mode_sends_the_question_as_plain_text_with_the_instruction(monkeypatch):
    model, recorder = _client_model(monkeypatch, "text")
    assert model.check_embedding_ctx_length is False and model.request_timeout == 60
    ai_chat._embed_query_with_prefix(QUESTION)
    sent = recorder.calls[0]
    assert sent["input"] == [ec.QUERY_INSTRUCTION + QUESTION]          # a string, not token IDs
    assert sent["input"][0].startswith("Instruct: Given a legal question in Italian")
    assert sent["extra_body"] == {"truncate_prompt_tokens": 8000}


def test_token_ids_mode_is_todays_client(monkeypatch):
    try:
        import tiktoken
        encoding = tiktoken.get_encoding("cl100k_base")
    except Exception as exc:                                           # needs the cached encoding file
        pytest.skip(f"cl100k_base not available offline: {exc}")
    model, recorder = _client_model(monkeypatch, "token_ids")
    assert model.check_embedding_ctx_length is True and model.request_timeout == 60
    assert not model.model_kwargs
    ai_chat._embed_query_with_prefix(QUESTION)
    sent = recorder.calls[0]
    ids = sent["input"][0]
    assert isinstance(ids, list) and all(isinstance(t, int) for t in ids)   # token IDs, as before
    assert encoding.decode(ids) == QUESTION                               # no instruction added
    assert "extra_body" not in sent


# --- Vector search -------------------------------------------------------------

class _Session:
    def __init__(self, rows=(), error=None):
        self.rows, self.error, self.calls = list(rows), error, []

    def run(self, query, **params):
        self.calls.append((query, params))
        if self.error:
            raise self.error
        return list(self.rows)


_ROWS = [{"element_id": f"s{i}", "labels": ["Section"], "score": s} for i, s in enumerate((0.8, 0.6, 0.3, 0.2))]


@pytest.fixture
def search(monkeypatch):
    monkeypatch.setattr(lookups, "_cached_embed_query", lambda text: (0.1, 0.2))

    def run(indexes, **kwargs):
        session = _Session(_ROWS)
        matches = lookups.vector_lookup(session, QUESTION, indexes=indexes, **kwargs)
        return session, matches
    return run


def test_default_search_is_todays(mode, search):
    session, matches = search(["section_embeddings"], index_settings=lookups.VECTOR_INDEX_SETTINGS)
    query, params = session.calls[0]
    assert params["index"] == "section_embeddings" and params["k"] == 30
    assert [m["score"] for m in matches] == [0.8, 0.6, 0.3]            # today's 0.25 threshold

    session, _ = search(["section_embeddings"], index_settings=lookups.VECTOR_INDEX_SETTINGS, doc_id="d1")
    query, params = session.calls[0]
    assert "s.embedding IS NOT NULL" in query and "cosine(s.embedding, $embedding)" in query
    assert params["min_score"] == 0.25

    # The dottrina searches pass no settings: no threshold today, 3 results.
    session, matches = search(["commentato_section_embeddings"], k=3, source_prefix="special_dottrina")
    assert session.calls[0][1]["k"] == 3 and len(matches) == 4         # nothing filtered


def test_text_mode_uses_the_text_thresholds(mode, search):
    mode(query_mode="text")
    session, matches = search(["section_embeddings"], index_settings=lookups.VECTOR_INDEX_SETTINGS)
    assert [m["score"] for m in matches] == [0.8, 0.6]                  # 0.55, not 0.25
    session, _ = search(["section_embeddings"], index_settings=lookups.VECTOR_INDEX_SETTINGS, doc_id="d1")
    assert session.calls[0][1]["min_score"] == 0.55                     # doc-scoped search too
    # The dottrina searches, called as retrieval.py calls them (no settings), now filter.
    for index in ("commentato_section_embeddings", "fiscalita_section_embeddings"):
        session, matches = search([index], k=3, source_prefix="special_dottrina")
        # 24 candidates fetched (3 x 8 since 6db1125, to allow for duplicate copies), at most 3 kept.
        assert session.calls[0][1]["k"] == 24 and [m["score"] for m in matches] == [0.8, 0.6]
    # Indexes that are not migrated keep their token-ID thresholds.
    session, matches = search(["legalact_embeddings"], index_settings=lookups.VECTOR_INDEX_SETTINGS)
    assert session.calls[0][1]["k"] == 2 and [m["score"] for m in matches] == [0.8, 0.6, 0.3]   # k=2, min 0.3


def test_document_vectors_are_not_searched_in_text_mode(mode, search):
    """document_embeddings still holds 35 token-ID vectors: a plain-text question
    matches them at random above its 0.25 threshold."""
    session, matches = search(["section_embeddings", "document_embeddings"],
                              index_settings=lookups.VECTOR_INDEX_SETTINGS)
    assert [p["index"] for _, p in session.calls] == ["section_embeddings", "document_embeddings"]
    mode(query_mode="text")
    session, matches = search(["section_embeddings", "document_embeddings"],
                              index_settings=lookups.VECTOR_INDEX_SETTINGS)
    assert [p["index"] for _, p in session.calls] == ["section_embeddings"]
    assert {m["source"] for m in matches} == {"vector:section_embeddings"}


def test_suffix_and_property_switch_only_the_migrated_indexes(mode, search):
    mode(query_mode="text", suffix="_v2", prop="embedding_v2")
    session, matches = search(["section_embeddings", "legalact_embeddings", "commentato_section_embeddings"],
                              index_settings=lookups.VECTOR_INDEX_SETTINGS)
    assert [p["index"] for _, p in session.calls] == [
        "section_embeddings_v2", "legalact_embeddings", "commentato_section_embeddings_v2"]
    assert {m["source"] for m in matches} >= {"vector:section_embeddings"}   # labels keep base names
    session, _ = search(["section_embeddings"], index_settings=lookups.VECTOR_INDEX_SETTINGS, doc_id="d1")
    query = session.calls[0][0]
    assert query.count("s.embedding_v2") == 2 and "s.embedding " not in query


def test_a_failed_vector_search_logs_why(mode, monkeypatch, caplog):
    """The error used to be only in `extra`, which the log format doesn't print."""
    monkeypatch.setattr(lookups, "_cached_embed_query", lambda text: (0.1, 0.2))
    mode(suffix="_v2")
    error = Neo4jError("There is no such vector schema index: section_embeddings_v2")
    with caplog.at_level(logging.WARNING, logger="src.rag.lookups"):
        assert lookups.vector_lookup(_Session(error=error), QUESTION, indexes=["section_embeddings"]) == []
    assert "Vector lookup failed on section_embeddings_v2" in caplog.text
    assert "no such vector schema index" in caplog.text


# --- V2's writer -------------------------------------------------------------------

def test_the_writer_follows_the_mode(mode, monkeypatch):
    monkeypatch.setenv("EMBEDDING_PROVIDER", "openai")
    monkeypatch.setenv("EMBEDDING_MODEL", "qwen3-embedding-8b")
    assert gen._add_doc_prefix("testo").startswith("Instruct: Represent this Italian legal document")
    assert gen._init_embedding_model().check_embedding_ctx_length is True
    mode(query_mode="text")
    assert gen._add_doc_prefix("testo") == "testo"                      # no instruction on documents
    writer = gen._init_embedding_model()
    assert writer.check_embedding_ctx_length is False
    assert writer.model_kwargs == {"extra_body": {"truncate_prompt_tokens": 8000}}
