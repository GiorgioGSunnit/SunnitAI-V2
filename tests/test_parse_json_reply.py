"""parse_json_reply: the first JSON value in a model reply. On production
(16 Oct) a strict json.loads failed with "Extra data" (the model wrote the list
and then more text) and the strategy pipeline lost its article choice."""

import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402
from src.rag.ai_chat import parse_json_reply  # noqa: E402


@pytest.mark.parametrize("raw, value", [
    ('["art. 628 c.p.", "art. 582 c.p."]', ["art. 628 c.p.", "art. 582 c.p."]),
    ('["art. 628 c.p."]\n\n["art. 582 c.p."]', ["art. 628 c.p."]),                      # "Extra data"
    ('["art. 628 c.p."]\nHo scelto la rapina perché...', ["art. 628 c.p."]),
    ('Ecco gli articoli:\n```json\n["art. 628 c.p."]\n```', ["art. 628 c.p."]),
    ('Considero [art. 628 c.p.] e quindi:\n["art. 628 c.p."]', ["art. 628 c.p."]),       # prose brackets first
    ('{"parte_assistita": "Tizio"} fine', {"parte_assistita": "Tizio"}),
])
def test_the_first_json_value_is_read(raw, value):
    assert parse_json_reply(raw) == value


@pytest.mark.parametrize("raw", ["", None, "Nessun articolo pertinente.", "[art. 628 c.p.]"])
def test_no_json_raises(raw):
    with pytest.raises(ValueError):
        parse_json_reply(raw)


def test_the_article_choice_survives_extra_text(monkeypatch):
    rows = [{"_article": ("628", "c.p."), "s": {"plain_text": "Chiunque, per procurare a sé..."}},
            {"_article": ("582", "c.p."), "s": {"plain_text": "Chiunque cagiona ad alcuno una lesione..."}}]
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: '["art. 628 c.p.", "art. 582 c.p."]\n\nNote: ...')
    assert dg._choose_articles("Rapina con lesioni.", rows) == [("628", "c.p."), ("582", "c.p.")]
