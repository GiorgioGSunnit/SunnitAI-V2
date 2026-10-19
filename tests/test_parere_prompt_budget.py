"""The strategy draft's prompt fits the serving context (held-out h9, prod 19 Oct:
~15,018 input tokens + 5,000 for the answer against 20,000, and the user got an
error). _fit_parere_prompt trims the ruling passages first, then the other
articles, and keeps the articles named in the case, cross-referenced, matched by
heading or found by several signals; _call_chat's own retry keeps a real margin.
Synthetic text only: the held-out cases are never used for tests."""

import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.ai_chat as ai_chat  # noqa: E402
import src.rag.defensive_generation as dg  # noqa: E402
from src.rag.nodes.cypher import CROSSREF_SOURCE  # noqa: E402

SYSTEM = "Istruzioni per il parere. " * 200                       # ~5,000 characters, like _parere_system


def _article(number, text_char, **tags):
    return {"d": {"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"}, "_article": (number, "c.p."),
            "s": {"name": f"{number}.0.0", "plain_text": (text_char * 1600)}, **tags}


def _ruling(i):
    return {"d": {"id": f"LEGAL_DOC::r{i}", "name": f"Sentenza di prova n. {i}/2026"},
            "s": {"name": "1_1", "plain_text": "x"}, "_passage": f"Motivazione {i}. " + "m" * 2600}


def _long_case():
    case = "Tizio, nel corso di una lite, colpisce Caio; il PM contesta l'art. 584 c.p. "
    return (case + "Seguono molti dettagli della vicenda narrata dal cliente. " * 200)[:11000]


def _many_articles():
    protected = [_article("584", "N"),                                   # named in the case
                 _article("581", "C", _source=CROSSREF_SOURCE),
                 _article("43", "H", _heading_match=True),
                 _article("589", "M", _multi_signal=True)]
    others = [_article(str(600 + i), "o") for i in range(16)]
    return protected + others


def _tokens(system, human):
    return dg._estimate_tokens(system) + dg._estimate_tokens(human)


def test_a_prompt_that_fits_is_the_one_built_before():
    rows, rulings = _many_articles()[:3], [_ruling(1), _ruling(2)]
    analysis = {"questioni": [], "fatti": []}
    articles, ruling_block, human = dg._fit_parere_prompt(SYSTEM, "Caso breve.", analysis, rows, rulings)
    assert articles == dg._grouped_source_block(rows, [], dg._ARTICLE_CHARS)
    assert ruling_block == dg._source_block(rulings, dg._PASSAGE_CHARS + 100)
    assert human == dg._parere_human("Caso breve.", analysis, articles, ruling_block)


def test_a_long_case_is_trimmed_to_fit_rulings_first_protected_articles_kept():
    case, rows, rulings = _long_case(), _many_articles(), [_ruling(i) for i in range(10)]
    analysis = {"questioni": [], "fatti": []}
    before = dg._parere_human(case, analysis, dg._grouped_source_block(rows, [], dg._ARTICLE_CHARS),
                              dg._source_block(rulings, dg._PASSAGE_CHARS + 100))
    assert _tokens(SYSTEM, before) > dg._DRAFT_INPUT_BUDGET                 # the h9 situation

    articles, ruling_block, human = dg._fit_parere_prompt(SYSTEM, case, analysis, rows, rulings)
    assert _tokens(SYSTEM, human) <= dg._DRAFT_INPUT_BUDGET
    assert case in human                                                   # the case is never cut
    for char in "NCHM":                                                    # protected: full article text
        assert char * dg._ARTICLE_CHARS in articles
    assert "o" * 600 not in articles and "o" * 400 in articles             # the others: shortened, not dropped
    assert "Sentenza di prova n. 5/2026" not in ruling_block               # rulings: fewer and shorter first
    assert "m" * 700 not in ruling_block


def test_the_last_step_shortens_protected_articles_rather_than_fail():
    case, analysis = _long_case(), {"questioni": [], "fatti": []}
    rows = [_article(str(i), "H", _heading_match=True) for i in range(1, 25)]   # 24 protected, ~36,000 chars
    articles, _, human = dg._fit_parere_prompt(SYSTEM, case, analysis, rows, [_ruling(i) for i in range(10)])
    assert _tokens(SYSTEM, human) <= dg._DRAFT_INPUT_BUDGET
    assert "H" * 800 in articles and "H" * 801 not in articles
    assert articles.count("[art. ") == 24 and case in human               # every article still there


def test_call_chat_retry_keeps_a_real_margin(monkeypatch):
    """The server reports "at least N" prompt tokens: a 16-token margin asked for
    one token more than fit (prod, 19 Oct)."""
    calls = []

    class _Model:
        def __init__(self, kw=None):
            self.kw = kw or {}

        def bind(self, **kw):
            return _Model(kw)

        def invoke(self, messages):
            calls.append(self.kw.get("max_tokens"))
            if len(calls) == 1:
                raise RuntimeError("This model's maximum context length is 20000 tokens. However, your request "
                                   "has 20018 input tokens: the prompt contains at least 15001 input tokens")
            return type("Reply", (), {"content": " ok "})()

    monkeypatch.setattr(ai_chat, "chat_model", _Model())
    assert ai_chat._call_chat([], max_tokens=5000) == "ok"
    assert calls == [5000, 20000 - 15001 - 300]


@pytest.mark.parametrize("message, overflow", [
    ("This model's maximum context length is 20000 tokens", True),
    ("Error code: 400 - maximum context length exceeded", True),
    ("Connection reset by peer", False),
])
def test_context_overflow_is_recognised(message, overflow):
    assert dg._is_context_overflow(RuntimeError(message)) is overflow
