"""Issue spotting (round 3, item 2): for a narrative question, the articles the
model names are resolved against the real document names, fetched exactly by
number, checked for applicability and added as protected rows. Off unless
ISSUE_SPOTTING_ENABLED.
"""

import json
import os

import pytest

os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")

import src.rag.defensive_generation as dg  # noqa: E402
from src.rag import issue_spotting as isp  # noqa: E402
from src.rag.nodes.cypher import ISSUE_SPOTTING_SOURCE, PROTECTED_SOURCES  # noqa: E402

DOCS = [
    {"id": "cp", "name": "Codice Penale 2026"},
    {"id": "cpp", "name": "Codice di procedura penale 2026"},
    {"id": "cpc", "name": "Codice di procedura civile Edizione 2026"},
    {"id": "cc", "name": "Codice Civile 2026"},
    {"id": "cons", "name": "Codice del Consumo", "aliases": ["d.lgs. 206/2005"]},
    {"id": "l392", "name": "Legge 27 luglio 1978, n. 392 - Disciplina delle locazioni di immobili urbani"},
    {"id": "comm", "name": "Codice Penale Commentato ADE"},
    {"id": "fisc", "name": "La Fiscalità delle Società IAS IFRS"},
    {"id": "ccnl", "name": "[CCNL] Commercio"},
]
NARRATIVE = ("Un consumatore acquista una lavatrice in un negozio; dopo tre mesi la macchina smette di "
             "funzionare per un difetto di fabbricazione. Il venditore rifiuta di intervenire sostenendo che "
             "deve rivolgersi al produttore. Quali tutele ha il consumatore e in quali termini?")


@pytest.mark.parametrize("code, doc_id", [
    ("c.p.", "cp"), ("codice penale", "cp"), ("Codice Penale italiano", "cp"), ("cod. pen.", "cp"),
    ("Codice di procedura penale", "cpp"), ("c.p.p.", "cpp"), ("c.p.c.", "cpc"), ("c.c.", "cc"),
    ("codice del consumo", "cons"), ("d.lgs. 206/2005", "cons"), ("Codice del consumo (d.lgs. 206/2005)", "cons"),
    ("legge 392/1978", "l392"), ("L. n. 392 del 1978", "l392"),
    ("legge 24/2017", None), ("codice della privacy", None), ("costituzione", None),
])
def test_codes_resolve_against_the_real_document_names(code, doc_id):
    found = isp.resolve_code(code, DOCS)
    assert (found or {}).get("id") == doc_id


def test_commentaries_and_agreements_are_never_the_target():
    assert isp.resolve_code("codice penale commentato", DOCS)["id"] == "cp"
    assert isp.resolve_code("ccnl commercio", DOCS) is None


def test_proposals_are_parsed_and_cleaned(monkeypatch):
    reply = [
        {"institution": "furto con strappo", "article": "Art. 624 bis", "code": "c.p."},
        {"institution": "rapina", "article": "628", "code": "codice penale"},
        {"institution": "doppione", "article": "628", "code": "Codice Penale"},
        {"institution": "senza numero", "article": "", "code": "c.p."},
        {"institution": "numero strano", "article": "abc", "code": "c.p."},
        "non un oggetto",
    ]
    monkeypatch.setattr(isp, "_call_chat", lambda *a, **k: "```json\n" + json.dumps(reply) + "\n```")
    assert isp.spot_issues(NARRATIVE) == [
        {"institution": "furto con strappo", "article": "624-bis", "code": "c.p."},
        {"institution": "rapina", "article": "628", "code": "codice penale"},
    ]
    monkeypatch.setattr(isp, "_call_chat", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("down")))
    assert isp.spot_issues(NARRATIVE) == []


class _Result:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        return self.rows


class _Session:
    """Articles by (document, canonical name)."""
    ARTICLES = {
        ("cons", "130"): "Il venditore è responsabile nei confronti del consumatore per qualsiasi difetto...",
        ("cc", "1490.0.0"): "Il venditore è tenuto a garantire che la cosa venduta sia immune da vizi...",
        ("cc", "1495.0.0"): "Il compratore decade dal diritto alla garanzia, se non denunzia...",
        ("cp", "116.0.0"): "Qualora il reato commesso sia diverso da quello voluto da taluno dei concorrenti...",
        ("cp", "628.0.0"): "Chiunque, per procurare a sé o ad altri un ingiusto profitto, mediante violenza...",
    }
    TITLES = {   # the rubrica, as Section.title holds it on production
        ("cp", "628.0.0"): "Rapina", ("cp", "625.0.0"): "Circostanze aggravanti",
        ("cp", "582.0.0"): "Lesione personale", ("cp", "590.0.0"): "Lesioni personali colpose",
        ("cp", "52.0.0"): "Difesa legittima", ("cp", "624.0.0"): "Furto",
        ("cp", "624-bis.0.0"): "Furto in abitazione e furto con strappo",
        ("cp", "640.0.0"): "Truffa", ("cp", "640-bis.0.0"): "Truffa aggravata per il conseguimento di erogazioni pubbliche",
        ("cc", "1490.0.0"): "Garanzia per i vizi della cosa venduta", ("cc", "1218.0.0"): "Responsabilità del debitore",
        ("cc", "1453.0.0"): "Risolubilità del contratto per inadempimento",
    }

    def __init__(self):
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.queries.append((query, params))
        if "db.index.fulltext.queryNodes($index" in query:
            # The title fulltext index (round 6, 19 Oct) doesn't exist in this
            # fake corpus -- real Neo4j raises for a missing index; article_by_
            # title's own try/except then falls back to the CONTAINS scan below,
            # which these tests exercise.
            raise Exception("no such fulltext schema index: section_title_fulltext")
        if "toLower(coalesce(s.title" in query:     # Neo4j's CONTAINS: a substring test
            return _Result([{"name": name, "title": title} for (doc, name), title in self.TITLES.items()
                            if doc == params["doc_id"] and all(w in title.lower() for w in params["stems"])])
        if "UNWIND $items AS it" in query:                  # article_text.fetch_article
            rows = []
            for it in params["items"]:
                for name in (f"{it['num']}.0.0", it["num"]):
                    text = self.ARTICLES.get((it["doc"], name))
                    if text:
                        rows.append({"doc": it["doc"], "num": it["num"],
                                     "d": {"id": it["doc"], "name": next(d["name"] for d in DOCS if d["id"] == it["doc"])},
                                     "s": {"id": f"{it['doc']}::{name}", "name": name, "plain_text": text}})
            return _Result(rows)
        return _Result([])


@pytest.fixture
def models(monkeypatch):
    proposals = [
        {"institution": "garanzia di conformità", "article": "130", "code": "codice del consumo"},
        {"institution": "garanzia per vizi", "article": "1490", "code": "c.c."},
        {"institution": "denuncia dei vizi", "article": "1495", "code": "codice civile"},
        {"institution": "concorso anomalo", "article": "116", "code": "c.p."},
        {"institution": "responsabilità medica", "article": "7", "code": "legge 24/2017"},   # not in the corpus
    ]
    verdicts = [{"article": "116 c.p.", "verdict": "incompatibile", "requisito": "più concorrenti",
                 "fatto": "una sola parte agisce", "reason": "Non c'è concorso."}]
    monkeypatch.setattr(isp, "_call_chat", lambda *a, **k: json.dumps(proposals))
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: json.dumps(verdicts))
    logged = []
    monkeypatch.setattr(isp, "vlog", lambda event, data=None, *a, **k: logged.append((event, data)))
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    return logged


def test_spotted_articles_are_fetched_checked_and_tagged(models):
    existing = [{"d": {"id": "cc"}, "s": {"name": "1495.0.0"}}]          # already retrieved: not added twice
    rows = isp.spotted_rows(_Session(), DOCS, NARRATIVE, existing)
    assert [r["_article"] for r in rows] == [("130", "Codice del Consumo"), ("1490", "c.c.")]
    assert all(r["_source"] == ISSUE_SPOTTING_SOURCE for r in rows)
    assert ISSUE_SPOTTING_SOURCE in PROTECTED_SOURCES
    event, data = models[-1]
    assert event == "issue_spotting"
    assert data["unresolved"] == ["art. 7 legge 24/2017"]
    assert data["dropped_by_check"] == ["art. 116 c.p."] and data["added"] == 2


@pytest.mark.parametrize("doc_id, institution, number", [
    ("cp", "Rapina", "628"), ("cp", "legittima difesa", "52"), ("cp", "lesioni personali", "582"),
    ("cp", "Furto in abitazione", "624-bis"), ("cp", "Furto", "624"), ("cp", "truffa", "640"),
    ("cc", "inadempimento contrattuale", "1453"), ("cc", "garanzia per vizi", "1490"),
    ("cc", "Responsabilità", None),             # one word: only a title of exactly that word
    ("cp", "Falso in atto privato", None), ("cp", "", None),
])
def test_articles_are_found_by_their_title(doc_id, institution, number):
    assert isp.article_by_title(_Session(), doc_id, institution) == number


class _FulltextIndexSession(_Session):
    """Like _Session, but the title fulltext index exists and answers the
    field-qualified wildcard query directly -- the CONTAINS scan must never
    run (if it did, it would answer from the same TITLES fixture and the
    test couldn't tell the two paths apart; the explicit failure here does)."""

    def run(self, query, **params):
        self.queries.append((query, params))
        if "db.index.fulltext.queryNodes($index" in query:
            return _Result([{"name": name, "title": title} for (doc, name), title in self.TITLES.items()
                            if doc == params["doc_id"]])
        if "toLower(coalesce(s.title" in query:
            raise AssertionError("the CONTAINS scan ran despite the fulltext index answering")
        return _Result([])


def test_article_by_title_uses_the_fulltext_index_when_it_exists():
    session = _FulltextIndexSession()
    assert isp.article_by_title(session, "cp", "Rapina") == "628"
    assert any("db.index.fulltext.queryNodes($index" in q for q, _ in session.queries)


def test_the_title_wins_over_a_wrong_number(monkeypatch):
    """Production, 15 Oct: "Rapina: art. 625" (625 is the aggravating circumstances)."""
    monkeypatch.setattr(isp, "_call_chat", lambda *a, **k: json.dumps(
        [{"institution": "Rapina", "article": "625", "code": "c.p."},
         {"institution": "concorso anomalo", "article": "116", "code": "c.p."}]))   # no title: by number
    monkeypatch.setattr(dg, "_call_chat", lambda *a, **k: "[]")
    logged = []
    monkeypatch.setattr(isp, "vlog", lambda event, data=None, *a, **k: logged.append((event, data)))
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    rows = isp.spotted_rows(_Session(), DOCS, NARRATIVE, [])
    assert [r["_article"] for r in rows] == [("628", "c.p."), ("116", "c.p.")]
    assert [r["s"]["name"] for r in rows] == ["628.0.0", "116.0.0"]
    assert logged[-1][1]["by_title"] == ["Rapina: 625 -> 628"]


def test_the_node_is_off_by_default_and_only_for_narrative_questions(models, monkeypatch):
    class Driver:
        def session(self, **config):
            return _Session()

    monkeypatch.setattr("src.rag.doc_lookup._fetch_doc_names", lambda *a, **k: DOCS)
    monkeypatch.delenv("ISSUE_SPOTTING_ENABLED", raising=False)
    state = {"query": NARRATIVE, "raw_result": [{"d": {"id": "x"}, "s": {"name": "1"}, "_source": "bm25"}]}
    assert isp.issue_spotting(dict(state), Driver(), "neo4j") == {}
    monkeypatch.setenv("ISSUE_SPOTTING_ENABLED", "true")
    assert isp.issue_spotting({**state, "query": "art. 1490 c.c. vizi"}, Driver(), "neo4j") == {}
    assert isp.issue_spotting({**state, "is_comparison": True}, Driver(), "neo4j") == {}
    out = isp.issue_spotting(dict(state), Driver(), "neo4j")
    assert [r.get("_source") for r in out["raw_result"]] == ["bm25", ISSUE_SPOTTING_SOURCE, ISSUE_SPOTTING_SOURCE,
                                                            ISSUE_SPOTTING_SOURCE]


def test_the_node_fails_open(monkeypatch):
    class Broken:
        def session(self, **config):
            raise RuntimeError("database down")

    monkeypatch.setenv("ISSUE_SPOTTING_ENABLED", "true")
    monkeypatch.setattr("src.rag.doc_lookup._fetch_doc_names", lambda *a, **k: DOCS)
    assert isp.issue_spotting({"query": NARRATIVE, "raw_result": []}, Broken(), "neo4j") == {}


def test_the_node_sits_after_context_retrieval():
    """context_retrieval rebuilds raw_result, so anything added before it would be lost."""
    import src.rag.main as main
    graph = main.build_graph(compile_graph=True).get_graph()
    after = {e.source: e.target for e in graph.edges}
    assert after["context_retrieval"] == "issue_spotting"
    assert after["issue_spotting"] == "generate_cypher_intersection"


def test_the_parere_adds_spotted_articles_only_when_enabled(monkeypatch):
    """The parere's own analysis knows only six code abbreviations."""
    spotted_row = {"d": {"id": "cons", "name": "Codice del Consumo"}, "s": {"name": "130", "plain_text": "..."},
                   "_article": ("130", "Codice del Consumo"), "_source": ISSUE_SPOTTING_SOURCE}
    monkeypatch.setattr("src.rag.issue_spotting.spotted_rows", lambda session, docs, text, existing: [spotted_row])
    monkeypatch.setattr("src.rag.doc_lookup._fetch_doc_names", lambda *a, **k: DOCS)
    for name in ("_search_case_law", "_search_similar_facts", "_search_code_articles"):
        monkeypatch.setattr(dg, name, lambda *a, **k: [])
    monkeypatch.setattr(dg, "_articles_cited_in_rulings", lambda *a, **k: {})
    monkeypatch.setattr(dg, "_fetch_articles", lambda *a, **k: [])
    monkeypatch.setattr(dg, "_choose_articles", lambda *a, **k: [])
    monkeypatch.setattr("src.rag.nodes.retrieval.broaden_article_context", lambda state, *a, **k: {})
    analysis = {"questioni": [], "area": "civile"}
    monkeypatch.delenv("ISSUE_SPOTTING_ENABLED", raising=False)
    assert dg._select_articles(None, None, "neo4j", NARRATIVE, "", analysis)["article_rows"] == []
    monkeypatch.setenv("ISSUE_SPOTTING_ENABLED", "true")
    picked = dg._select_articles(None, None, "neo4j", NARRATIVE, "", analysis)
    assert picked["article_rows"] == [spotted_row] and picked["spotted"] == [spotted_row]
