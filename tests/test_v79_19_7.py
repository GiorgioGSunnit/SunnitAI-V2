"""v79.19.7 -- scripts/article_cards_bulk.py --docs-only: with --doc, the order groups (a) top and (b) the four codes are
skipped, in the cards pass and in the --schemas pass; the dry run reports the --doc documents only. A fake documents map,
no model, no database."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("article_cards_bulk_7", ROOT / "scripts" / "article_cards_bulk.py")
bulk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bulk)

CODE, LAW = "Codice Civile 2026", "Codice Rosso 2026"


def _art(n, text="Chiunque fa qualcosa è punito."):
    return {"number": n, "code": "x", "rubrica": f"Rubrica {n}", "text": text}


DOCUMENTS = {CODE: {"1": _art("1"), "2043": _art("2043"), "9": _art("9", "(abrogato)")},
             LAW: {"1": _art("1"), "2": _art("2"), "3": _art("3", "(abrogato)")}}


def test_docs_only_yields_the_doc_articles_only_in_the_cards_order():
    items, bare = bulk.card_items([(CODE, "2043")], DOCUMENTS, [CODE], [LAW], docs_only=True)
    assert [(d, a["number"]) for d, a in items] == [(LAW, "1"), (LAW, "2")]
    assert [(d, a["number"]) for d, a in bare] == [(LAW, "3")]
    # without the flag: the top one, then the code by number, then the --doc
    items, bare = bulk.card_items([(CODE, "2043")], DOCUMENTS, [CODE], [LAW])
    assert [(d, a["number"]) for d, a in items] == [(CODE, "2043"), (CODE, "1"), (LAW, "1"), (LAW, "2")]
    assert [(d, a["number"]) for d, a in bare] == [(CODE, "9"), (LAW, "3")]
    assert bulk.order_refs([(CODE, "2043")], {CODE: ["1"]}, {LAW: ["1"]}, docs_only=True) == [(LAW, "1")]


def test_the_schemas_pass_honours_docs_only():
    cards = [(CODE, "2043"), (LAW, "2")]
    items, bare, carded = bulk.schema_items(cards, DOCUMENTS, [CODE], [LAW], docs_only=True)
    assert [(d, a["number"]) for d, a in items] == [(LAW, "1"), (LAW, "2")] and carded == []
    assert [(d, a["number"]) for d, a in bare] == [(LAW, "3")]
    items, _, carded = bulk.schema_items(cards, DOCUMENTS, [CODE], [LAW])
    assert [(d, a["number"]) for d, a in items] == [(CODE, "2043"), (LAW, "2"), (CODE, "1"), (LAW, "1")]
    assert carded == cards
    # a code document not read (docs-only skips reading it) is no error
    assert bulk.schema_items([], {LAW: DOCUMENTS[LAW]}, [CODE], [LAW], docs_only=True)[0][0][0] == LAW


def test_docs_only_needs_a_doc(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["article_cards_bulk.py", "--docs-only", "--dry-run"])
    with pytest.raises(SystemExit) as exc:
        bulk.main()
    assert exc.value.code == 2


def test_the_dry_run_reports_the_doc_documents_only(monkeypatch, capsys, tmp_path):
    """main() with the graph faked: --docs-only --doc LAW --dry-run reads no code document, counts no top, and
    prints the --doc articles alone."""
    read = []

    class Session:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def run(self, *a, **k): raise AssertionError("no query expected beyond read_document")

    class Driver:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def session(self, **k): return Session()

    import types
    fake_neo4j = types.SimpleNamespace(GraphDatabase=types.SimpleNamespace(driver=lambda *a, **k: Driver()),
                                       READ_ACCESS="READ")
    monkeypatch.setitem(sys.modules, "neo4j", fake_neo4j)
    monkeypatch.setattr(bulk, "read_document", lambda session, name, code: (read.append(name), DOCUMENTS[name])[1])
    monkeypatch.setattr(bulk, "code_document", lambda session, code: (_ for _ in ()).throw(AssertionError("codes read")))
    monkeypatch.setattr(bulk, "frequency", lambda paths: (_ for _ in ()).throw(AssertionError("top counted")) if paths
                        else bulk.Counter())
    monkeypatch.setenv("NEO4J_URI", "bolt://x")
    monkeypatch.setattr(sys, "argv", ["article_cards_bulk.py", "--docs-only", "--doc", LAW, "--dry-run",
                                      "--file", str(tmp_path / "cards.jsonl"), "--env", str(tmp_path / "none.env")])
    bulk.main()
    out = capsys.readouterr().out
    assert read == [LAW]
    assert out.startswith("2 articles: only the 1 --doc documents (--docs-only); 1 only \"(abrogato)\", no card; 2 to make")
    assert f"art. 1 {LAW}" in out and f"art. 2 {LAW}" in out and CODE not in out
