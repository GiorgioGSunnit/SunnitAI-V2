"""article_text.ArticleStore and the batched article reads (3 Nov): named properties only, each article read once per
request, the rows handed out as copies, the per-article semantics of the former loops kept."""

import os

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402
from src.rag import article_text as AT  # noqa: E402

VECTORS = ("embedding", "heading_embedding", "embedding_v2", "vettore", "embedding_text")


class Session:
    """Documents in traversal order and their Sections; answers the prefix query and _sections' UNWIND query."""

    def __init__(self, docs, sections):
        self.docs, self.sections, self.queries = docs, sections, []

    def run(self, query, **params):
        self.queries.append(query)
        if "prefixes" in params:
            rows = [{"prefix": p, "id": d["id"]} for p in params["prefixes"] for d in self.docs
                    if d["document_type"] == "primary" and d["name"].startswith(p)]
        else:
            rows = [{"doc": it["doc"], "num": it["num"], "d": {"id": it["doc"], "name": it["doc"], "file_name": None},
                     "s": {**s, "titolo_articolo": s.get("titolo_articolo")}, "next": None}
                    for it in params["items"] for s in self.sections.get(it["doc"], [])
                    if s["name"] == it["num"] or s["name"].startswith((it["num"] + ".", it["num"] + "_"))]
        return type("R", (), {"data": lambda self_: rows})()


def _cc(*names_texts):
    return [{"name": n, "plain_text": t, "type": "Comma"} for n, t in names_texts]


def test_the_queries_name_their_properties_and_never_a_vector():
    for query in (AT._SECTIONS_QUERY, AT._PREFIX_DOCS_QUERY):
        assert "RETURN d, s" not in query and not any(v in query for v in VECTORS)
    assert AT.SECTION_MAP.startswith("s {.id, .name") and "plain_text" in AT.SECTION_MAP
    assert AT.present({"name": "1", "titolo_articolo": None}) == {"name": "1"}     # a property the node lacks: absent


def test_an_article_is_read_once_per_request_and_handed_out_as_copies():
    session = Session([], {"doc": _cc(("61", "Primo."), ("61.1.0", "Primo."))})
    store = AT.ArticleStore()
    first = store.sections(session, [("doc", "61")])
    first[("doc", "61")][0]["s"]["plain_text"] = "changed by a caller"
    again = store.sections(session, [("doc", "61"), ("doc", "62")])
    assert again[("doc", "61")][0]["s"]["plain_text"] == "Primo." and ("doc", "62") not in again
    assert len(session.queries) == 2 and store.stats() == {"queries": 2, "articles_asked": 3, "articles_read": 2}
    store.sections(session, [("doc", "62"), ("doc", "61")])
    assert len(session.queries) == 2                     # an article with no rows is not read again either


def test_fetch_many_takes_the_first_spelling_with_rows_as_fetch_article_does():
    session = Session([], {"doc": _cc(("21-novies.1", "Legacy."), ("21-novies.2", "Legacy due."))})
    found = AT.fetch_many(session, [("doc", "21-nonies"), ("doc", "7")])
    assert [r["s"]["name"] for r in found[("doc", "21-nonies")]] == ["21-novies.1", "21-novies.2"]
    assert found[("doc", "7")] == [] and len(session.queries) == 1
    assert found[("doc", "21-nonies")] == AT.fetch_article(Session([], session.sections), "doc", "21-nonies")


def test_fetch_articles_keeps_the_name_order_the_cut_and_the_documents_order_of_ties():
    docs = [{"id": "B", "name": "Codice Civile", "document_type": "primary"},
            {"id": "A", "name": "Codice Civile aggiornato", "document_type": "primary"},
            {"id": "X", "name": "Codice Civile commentato", "document_type": "special"}]
    frags = [(f"1.{k}.0", f"Frammento {k}.") for k in range(1, 200)]
    session = Session(docs, {"B": _cc(*frags), "A": _cc(*frags), "X": _cc(*frags)})
    rows = dg._fetch_articles(session, [("1", "c.c."), ("1", "c.c.")])
    # two documents' rows by Section name, the first 300 (B's 1.1.0 before A's: the documents' order), the special one out
    assert [r["d"]["id"] for r in rows] == ["B", "A", "B", "A"] and len(session.queries) == 2
    assert rows[0] is not rows[2] and rows[0]["s"] is not rows[2]["s"]
