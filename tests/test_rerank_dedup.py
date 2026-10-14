"""One row per section before reranking (text mode only).

V2-1 removed duplicate Section copies (same document, same name, up to 7
identical nodes) from vector search. Copies still reached the reranker through
keyword search, the article lookup, Cypher results and cross-references; it
scores each copy the same, and its own merge works on s.id, which differs
between copies, so three copies of one article took three of the answer's 12
slots. rerank_results now removes them first, also when the reranker is off or
fails. Token-ID mode is unchanged.
"""

import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import requests  # noqa: E402

import src.rag.embedding_config as ec  # noqa: E402
import src.rag.reranker as reranker  # noqa: E402


def _row(sid, doc="cc", name="2051.0.0", source=None, **extra):
    row = {"d": {"id": doc, "name": "Codice Civile"}, "s": {"id": sid, "name": name, "plain_text": "testo"}}
    if source:
        row["_source"] = source
    row.update(extra)
    return row


@pytest.fixture
def text_mode(monkeypatch):
    monkeypatch.setattr(ec, "QUERY_MODE", "text")
    logged = []
    monkeypatch.setattr(reranker, "vlog", lambda event, data=None, *a, **k: logged.append((event, data)))
    return logged


def test_copies_collapse_and_a_protected_copy_wins(text_mode):
    rows = [_row("a1", source="cypher"), _row("b1", name="1227.0.0"), _row("a2", source="bm25"),
            _row("a3"), _row("c1", name="2043.0.0", source="cypher"),
            _row("c2", name="2043.0.0", source="broaden_article_crossref")]
    out = reranker.dedup_rows(rows)
    # One row per section, each at the position of its first copy.
    assert [(r["s"]["id"], r.get("_source")) for r in out] == [
        ("a2", "bm25"), ("b1", None), ("c2", "broaden_article_crossref")]
    assert text_mode[-1] == ("rerank_dedup", {"input_count": 6, "rerank_duplicates_dropped": 3})


def test_the_first_protected_copy_stays_when_several_are_protected(text_mode):
    out = reranker.dedup_rows([_row("a1", source="bm25"), _row("a2", source="broaden_article_crossref")])
    assert [r["s"]["id"] for r in out] == ["a1"]


def test_sources_lists_are_merged(text_mode):
    out = reranker.dedup_rows([_row("a1", sources=["context"]), _row("a2", sources=["bm25", "context_variant_1"])])
    assert out[0]["sources"] == ["bm25", "context", "context_variant_1"]


def test_copies_from_different_documents_both_stay(text_mode):
    """Comparison answers put the same article of two documents side by side."""
    rows = [_row("a1", doc="cc_2025", source="comparison"), _row("a2", doc="cc_2026", source="comparison")]
    assert reranker.dedup_rows(rows) == rows


def test_rows_without_a_key_are_kept(text_mode):
    rows = [{"s": {"id": "x", "name": "2051.0.0"}}, {"s": {"id": "y", "name": "2051.0.0"}},   # no d
            {"d": {"id": "cc"}}, {"d": {"id": "cc"}},                                       # no s
            _row("n1", name=""), _row("n2", name=""), _row("n3", doc=""), {"_source": "clarification"}]
    assert reranker.dedup_rows(rows) == rows


def test_dedup_happens_with_the_reranker_disabled(text_mode, monkeypatch):
    monkeypatch.setattr(reranker, "_RERANKER_ENABLED", False)
    out = reranker.rerank_results("domanda", [_row("a1"), _row("a2"), _row("b1", name="1227.0.0")])
    assert [r["s"]["id"] for r in out] == ["a1", "b1"]


def test_dedup_happens_when_the_reranker_fails(text_mode, monkeypatch):
    monkeypatch.setattr(reranker, "_RERANKER_ENABLED", True)

    def down(*a, **k):
        raise requests.ConnectionError("reranker down")

    monkeypatch.setattr(requests, "post", down)
    out = reranker.rerank_results("domanda", [_row("a1"), _row("a2", source="bm25"), _row("b1", name="1227.0.0")])
    assert [(r["s"]["id"], r.get("_source")) for r in out] == [("a2", "bm25"), ("b1", None)]


def test_the_reranker_scores_each_section_once(text_mode, monkeypatch):
    monkeypatch.setattr(reranker, "_RERANKER_ENABLED", True)
    sent = {}

    class _Response:
        def raise_for_status(self):
            pass

        def json(self):
            n = len(sent["documents"])
            return {"results": [{"index": i, "relevance_score": 1 - i / 10} for i in range(n)]}

    def post(url, headers=None, json=None, timeout=None):
        sent.update(json)
        return _Response()

    monkeypatch.setattr(requests, "post", post)
    rows = [_row("a1"), _row("a2"), _row("a3"), _row("b1", name="1227.0.0"), _row("c1", name="2043.0.0")]
    out = reranker.rerank_results("domanda", rows, top_k=12)
    assert len(sent["documents"]) == 3
    assert [r["s"]["id"] for r in out] == ["a1", "b1", "c1"]


def test_token_mode_returns_the_input_unchanged(monkeypatch):
    monkeypatch.setattr(ec, "QUERY_MODE", "token_ids")
    monkeypatch.setattr(reranker, "_RERANKER_ENABLED", False)
    rows = [_row("a1"), _row("a2", source="bm25"), _row("a3")]
    assert reranker.dedup_rows(rows) is rows
    assert reranker.rerank_results("domanda", rows) is rows
