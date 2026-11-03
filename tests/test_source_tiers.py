"""Source tiers (src/rag/source_tiers.py): primary-only article channels, rulings only as support for the
articles found, the narrative map's hypotheses confirmed by a primary channel or by 3+ similar rulings.
Invented data only."""
import os

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")

import pytest  # noqa: E402

from src.rag import source_tiers as st  # noqa: E402


class _Result:
    def __init__(self, rows):
        self._rows = rows

    def data(self):
        return self._rows


def _row(doc_id, name, doc_type=None, article=None, **extra):
    d = {"id": doc_id, "name": name}
    if doc_type is not None:
        d["document_type"] = doc_type
    row = {"d": d, "s": {"name": f"{article}.0.0" if article else "x", "plain_text": "t"}, **extra}
    return row


class TypeSession:
    def __init__(self, types):
        self.types, self.asked = types, []

    def run(self, query, **params):
        self.asked.append(params.get("ids"))
        return _Result([{"id": i, "type": self.types.get(i)} for i in params["ids"]])


@pytest.fixture
def logged(monkeypatch):
    events = []
    monkeypatch.setattr(st, "vlog", lambda event, data=None, **k: events.append((event, data)))
    return events


def test_keep_types_keeps_primary_and_logs_the_hits_by_type(logged):
    rows = [_row("cc", "Codice Civile", "primary"), _row("cass1", "Cass. civ. 1/2024", "interpretation"),
            _row("manuale", "Manuale", "special"), {"x": 1}, _row("nodoc", "Senza tipo")]
    session = TypeSession({"nodoc": None})
    kept = st.keep_types(session, rows, st.DIRECT_TYPES, "context_retrieval")
    assert [((r.get("d") or {}).get("id")) for r in kept] == ["cc", "manuale", None, "nodoc"]
    event, data = logged[-1]
    assert event == "source_tier_hits" and data["stage"] == "context_retrieval"
    assert data["by_type"] == {"primary": 1, "interpretation": 1, "special": 1, "unknown": 2}
    assert data["dropped"] == 1 and data["kept"] == 4 and data["non_primary_after"] == 0


def test_the_filter_fails_open_on_a_missing_type_and_looks_it_up_from_the_document(logged):
    """No document_type on the row: read from its Document; not there either, or no such Document: kept."""
    rows = [{"d": {"id": "LEGAL_DOC::cp", "name": "Codice Penale"}, "s": {"name": "52.0.0"}},
            {"s": {"id": "DOCUMENT_SECTION::cass7::4", "name": "4"}},
            {"d": {"id": "LEGAL_DOC::ghost", "name": "?"}, "s": {"name": "1"}},
            {"d": {"id": "LEGAL_DOC::bare", "name": "Senza tipo"}, "s": {"name": "2"}}]

    class Missing(TypeSession):          # only the Documents that exist come back
        def run(self, query, **params):
            self.asked.append(params.get("ids"))
            return _Result([{"id": i, "type": self.types[i]} for i in params["ids"] if i in self.types])
    session = Missing({"LEGAL_DOC::cp": "primary", "LEGAL_DOC::cass7": "interpretation", "LEGAL_DOC::bare": None})
    kept = st.keep_types(session, rows, st.DIRECT_TYPES, "article_router")
    assert [(r.get("d") or {}).get("id") for r in kept] == ["LEGAL_DOC::cp", "LEGAL_DOC::ghost", "LEGAL_DOC::bare"]
    decisions = st._decisions(session, rows, set(st.DIRECT_TYPES))
    assert decisions[0]["type_from"].startswith("Document lookup") and decisions[1]["why"].startswith("interpretation")
    assert "no Document" in decisions[2]["why"] and "has no document_type" in decisions[3]["why"]


def test_a_failed_lookup_keeps_every_untyped_row(logged):
    class Broken:
        def run(self, *a, **k):
            raise RuntimeError("down")
    rows = [{"d": {"id": "a", "name": "A"}}, {"d": {"id": "b", "name": "B", "document_type": "interpretation"}}]
    assert [r["d"]["id"] for r in st.keep_types(Broken(), rows, st.DIRECT_TYPES, "x")] == ["a"]


def test_the_debug_hook_records_each_nodes_rows_and_decisions(monkeypatch, logged):
    monkeypatch.setattr(st, "DEBUG_ROWS", [])
    st.keep_types(TypeSession({}), [_row("cc", "Codice Civile", "primary", article="1490")], st.DIRECT_TYPES, "n")
    record = st.DEBUG_ROWS[-1]
    assert record["stage"] == "n" and record["decisions"][0]["type_from"] == "d.document_type"
    assert record["decisions"][0]["kept"] and record["decisions"][0]["section"] == "1490.0.0"


def test_a_row_without_its_document_type_is_looked_up_once():
    session = TypeSession({"LEGAL_DOC::cass9": "interpretation", "cass9": "interpretation"})
    rows = [{"s": {"id": "DOCUMENT_SECTION::cass9::3", "name": "3"}}, _row("cass9", "Cass.")]
    assert st.doc_types(session, rows) == ["interpretation", "interpretation"]
    assert session.asked == [["LEGAL_DOC::cass9", "cass9"]]


def test_refs_of_reads_tags_and_code_names():
    rows = [{"_article": ("1490", "c.c."), "d": {}, "s": {}},
            {"d": {"name": "Codice Penale 2026"}, "s": {"name": "52.0.0"}},
            {"d": {"name": "Costituzione"}, "s": {"name": "24.0.0"}},
            {"d": {"name": "Codice del Consumo"}, "s": {"name": "130.0.0"}}]     # no CITES edges for it
    assert st.refs_of(rows) == [("1490", "c.c."), ("52", "c.p."), ("24", "Cost.")]


class SupportSession:
    """CITES edges and ruling similarities for support_rulings."""
    CITES = {"cassA": {("1490", "c.c."), ("1495", "c.c.")}, "cassB": {("1490", "c.c.")},
             "cassC": {("360", "c.p.c.")}}
    SIMS = {"cassA": 0.5, "cassB": 0.8, "cassC": 0.99}

    def __init__(self):
        self.refs = None

    def run(self, query, **params):
        if "UNWIND $refs AS ref" in query:
            self.refs = params["refs"]
            wanted = {(r["article"], r["code"]) for r in params["refs"]}
            return _Result([{"ruling": rid, "article": a, "code": c}
                            for rid, refs in self.CITES.items() for a, c in refs if (a, c) in wanted])
        if "vector.similarity.cosine" in query:
            return _Result([{"d": {"id": i, "name": f"Cass. {i}", "document_type": "interpretation"},
                             "s": {"name": "1"}, "sim": self.SIMS[i]} for i in params["ids"]])
        raise AssertionError(query)


@pytest.fixture
def no_embedding(monkeypatch):
    import src.rag.ai_chat as ai_chat

    class _Model:
        def embed_query(self, text):
            return [0.1, 0.2]
    monkeypatch.setattr(ai_chat, "embedding_model", _Model())


def test_support_rulings_cite_an_answer_article_without_boilerplate_ranked_by_similarity(no_embedding, logged):
    session = SupportSession()
    rows = st.support_rulings(session, [("1490", "c.c."), ("1495", "c.c."), ("360", "c.p.c.")], "vizi", 4)
    assert [r["d"]["id"] for r in rows] == ["cassB", "cassA"]          # cassC only shares boilerplate
    assert {(r["article"], r["code"]) for r in session.refs} == {("1490", "c.c."), ("1495", "c.c.")}
    assert rows[0]["_source"] == st.SUPPORT_SOURCE and rows[1]["_cites"] == ["1490 c.c.", "1495 c.c."]


def test_on_the_narrative_path_shared_articles_weigh(no_embedding, logged):
    rows = st.support_rulings(SupportSession(), [("1490", "c.c."), ("1495", "c.c.")], "vizi", 4,
                              weight_shared=True)
    assert [r["d"]["id"] for r in rows] == ["cassA", "cassB"]          # 0.5 x 2 beats 0.8 x 1


def test_a_failure_gives_no_support(no_embedding, logged):
    class Broken:
        def run(self, *a, **k):
            raise RuntimeError("down")
    assert st.support_rulings(Broken(), [("1490", "c.c.")], "vizi", 4) == []
    assert "error" in logged[-1][1]


def test_a_hypothesis_needs_a_primary_channel_or_three_similar_rulings():
    hypotheses = [("1490", "c.c."), ("2043", "c.c."), ("1218", "c.c.")]
    stats = {("2043", "c.c."): {"rulings": 3}, ("1218", "c.c."): {"rulings": 2}}
    kept, why = st.confirmed_hypotheses(hypotheses, [("1490", "c.c.")], stats)
    assert kept == [("1490", "c.c."), ("2043", "c.c.")]
    assert why == {"channel": ["1490 c.c."], "rulings": ["2043 c.c."], "rejected": ["1218 c.c."]}


def test_the_flags_act_on_their_route(monkeypatch):
    monkeypatch.setenv("HYBRID_NARRATIVE_ENABLED", "true")
    monkeypatch.setenv("PRIMARY_FIRST_ENABLED", "true")
    assert st.hybrid_enabled("narrative") and not st.hybrid_enabled("direct")
    assert st.primary_first_enabled("direct") and not st.primary_first_enabled("ccnl")
    monkeypatch.setenv("HYBRID_NARRATIVE_ENABLED", "false")
    assert not st.hybrid_enabled("narrative")


class _Driver:
    def __init__(self, session):
        self._session = session

    def session(self, **config):
        driver = self

        class _Ctx:
            def __enter__(self_):
                return driver._session

            def __exit__(self_, *exc):
                return False
        return _Ctx()


def test_the_graph_node_keeps_primary_rows_on_a_direct_turn_only(monkeypatch, logged):
    rows = [_row("cc", "Codice Civile", "primary"), _row("cass1", "Cass.", "interpretation")]
    node = st.primary_node(lambda state: {"raw_result": list(rows)}, "execute_cypher", _Driver(TypeSession({})), "db")
    monkeypatch.setenv("PRIMARY_FIRST_ENABLED", "true")
    monkeypatch.delenv("ROUTER_ENABLED", raising=False)
    assert [r["d"]["id"] for r in node({"query": "q"})["raw_result"]] == ["cc"]
    assert [r["d"]["id"] for r in node({"query": "q", "route": "ccnl"})["raw_result"]] == ["cc", "cass1"]
    monkeypatch.setenv("PRIMARY_FIRST_ENABLED", "false")
    assert len(node({"query": "q"})["raw_result"]) == 2


def test_the_synthesis_gets_the_support_rulings(monkeypatch, no_embedding, logged):
    monkeypatch.setenv("PRIMARY_FIRST_ENABLED", "true")
    monkeypatch.delenv("ROUTER_ENABLED", raising=False)

    class Session(SupportSession, TypeSession):
        def __init__(self):
            SupportSession.__init__(self)
            TypeSession.__init__(self, {})

        def run(self, query, **params):
            if "RETURN d.id AS id, d.document_type AS type" in query:
                return TypeSession.run(self, query, **params)
            return SupportSession.run(self, query, **params)

    seen = {}

    def synth(state):
        seen["rows"] = state["raw_result"]
        return {"answer": "ok"}

    node = st.support_node(synth, _Driver(Session()), "db")
    article = {"d": {"id": "cc", "name": "Codice Civile", "document_type": "primary"},
               "s": {"name": "1490.0.0", "plain_text": "Il venditore è tenuto"}}
    out = node({"query": "garanzia per vizi", "raw_result": [article]})
    assert [r["d"]["id"] for r in seen["rows"]] == ["cc", "cassB", "cassA"]
    assert out["answer"] == "ok" and len(out["raw_result"]) == 3


def test_the_narrative_writer_gets_rulings_citing_the_kept_articles(monkeypatch, logged):
    os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
    os.environ.setdefault("NEO4J_USER", "neo4j")
    os.environ.setdefault("NEO4J_PASSWORD", "test-password")
    import src.rag.defensive_generation as dg
    import src.rag.main as main
    monkeypatch.setattr(main, "driver", _Driver(object()))
    seen = {}

    def fake_support(session, refs, text, k, weight_shared=False, stage=""):
        seen.update(refs=refs, weight_shared=weight_shared, stage=stage)
        return [{"d": {"id": "cassA", "name": "Cass. A"}, "s": {"name": "1"}}]
    monkeypatch.setattr(st, "support_rulings", fake_support)
    monkeypatch.setattr(dg, "_ruling_passages", lambda session, rows, sel, analysis: [{**r, "_passage": "p"} for r in rows])
    rows = dg._support_rulings("fatti", "", [{"_article": ("1490", "c.c."), "d": {}, "s": {}},
                                             {"_article": ("130", "Codice del Consumo"), "d": {}, "s": {}}], {})
    assert seen == {"refs": [("1490", "c.c.")], "weight_shared": True, "stage": "narrative"}
    assert [r["d"]["id"] for r in rows] == ["cassA"] and rows[0]["_passage"] == "p"


class RerunSession(TypeSession):
    """Document types, plus the primary-restricted fulltext/vector passes (their Cypher recorded)."""
    def __init__(self, types, fulltext, vector):
        super().__init__(types)
        self.fulltext, self.vector, self.queries = fulltext, vector, []

    def run(self, query, **params):
        if "RETURN d.id AS id, d.document_type AS type" in query:
            return super().run(query, **params)
        self.queries.append(query)
        return _Result(self.fulltext if "fulltext" in query else self.vector)


def _hit(doc, name, article, score):
    return {"d": {"id": doc, "name": name, "document_type": "primary"}, "s": {"name": article}, "score": score}


@pytest.fixture
def cached_embedding(monkeypatch):
    import src.rag.lookups as lookups
    monkeypatch.setattr(lookups, "_cached_embed_query", lambda text: (0.1, 0.2))


def test_a_channel_left_with_nothing_runs_again_on_primary_documents_only(monkeypatch, cached_embedding, logged):
    monkeypatch.setenv("PRIMARY_FIRST_ENABLED", "true")
    monkeypatch.delenv("ROUTER_ENABLED", raising=False)
    rulings = [_row(f"cass{i}", f"Cass. pen. {i}/2024", "interpretation") for i in range(15)]
    session = RerunSession({}, fulltext=[_hit("LEGAL_DOC::cpp", "Codice di procedura penale", "585.0.0", 9.1)],
                           vector=[_hit("LEGAL_DOC::cpp", "Codice di procedura penale", "585.0.0", 0.8),
                                   _hit("LEGAL_DOC::cpp", "Codice di procedura penale", "593.0.0", 0.7)])
    node = st.primary_node(lambda state: {"raw_result": rulings}, "context_retrieval", _Driver(session), "db")
    out = node({"query": "termine appello penale", "retrieval_keywords": ["termine appello", "sentenza penale"]})
    assert [(r["s"]["name"], r["_source"]) for r in out["raw_result"]] == [
        ("585.0.0", "primary_rerun_fulltext"), ("593.0.0", "primary_rerun_vector")]
    # the document_type filter is inside each search, before its LIMIT -- not applied to the results after
    assert len(session.queries) == 2 and all("d.document_type = 'primary'" in q and q.index("document_type") <
                                             q.index("LIMIT $k") for q in session.queries)
    events = dict(logged)
    assert events["source_tier_hits"]["kept"] == 0 and events["source_tier_hits"]["dropped"] == 15
    rerun = events["source_tier_rerun"]
    assert rerun["stage"] == "context_retrieval" and rerun["first_pass"] == {"rows": 15, "kept": 0}
    assert rerun["second_pass"]["fulltext"] == 1 and rerun["second_pass"]["vector"] == 2
    assert rerun["second_pass"]["fulltext_query"] == "termine appello sentenza penale"


def test_a_channel_with_some_primary_rows_is_not_run_again(monkeypatch, cached_embedding, logged):
    monkeypatch.setenv("PRIMARY_FIRST_ENABLED", "true")
    monkeypatch.delenv("ROUTER_ENABLED", raising=False)
    session = RerunSession({}, fulltext=[], vector=[])
    rows = [_row("cpp", "Codice di procedura penale", "primary", "586"), _row("cass1", "Cass.", "interpretation")]
    out = st.primary_node(lambda state: {"raw_result": rows}, "execute_cypher", _Driver(session), "db")({"query": "q"})
    assert [r["d"]["id"] for r in out["raw_result"]] == ["cpp"] and session.queries == []
    assert "source_tier_rerun" not in dict(logged)


def test_a_failed_primary_pass_gives_nothing_and_says_why(cached_embedding):
    class Broken:
        def run(self, *a, **k):
            raise RuntimeError("index missing")
    rows, info = st.primary_search(Broken(), "appello", "appello")
    assert rows == [] and "index missing" in info["fulltext_error"] and "index missing" in info["vector_error"]


def test_the_primary_pass_reads_the_configured_index_property_and_query_mode(monkeypatch):
    """EMBEDDING_INDEX_SUFFIX=_v2, EMBEDDING_PROPERTY=embedding_v2, EMBEDDING_QUERY_MODE=text: the live
    search's index, query instruction and text-mode threshold -- never the v1 "section_embeddings"."""
    import sys
    import src.rag.lookups as lookups
    # The modules the live embedding actually reads (other tests swap src.rag.ai_chat in sys.modules).
    ai_globals = lookups._embed_query_with_prefix.__globals__
    configs = {id(m): m for m in (ai_globals["embedding_config"], lookups.embedding_config,
                                  sys.modules["src.rag.embedding_config"])}.values()
    for config in configs:
        monkeypatch.setattr(config, "INDEX_SUFFIX", "_v2")
        monkeypatch.setattr(config, "VECTOR_PROPERTY", "embedding_v2")
        monkeypatch.setattr(config, "QUERY_MODE", "text")
    ec = lookups.embedding_config
    embedded = []

    class _Model:
        def embed_query(self, text):
            embedded.append(text)
            return [0.1, 0.2]
    monkeypatch.setitem(ai_globals, "embedding_model", _Model())
    lookups._cached_embed_query.cache_clear()

    class Session:
        def __init__(self):
            self.vector_params = None

        def run(self, query, **params):
            if "db.index.vector.queryNodes" in query:
                self.vector_params = params
            return _Result([])
    session = Session()
    rows, info = st.primary_search(session, "appello", "termine per l'appello penale")
    assert session.vector_params, info.get("vector_error")
    assert session.vector_params["index"] == "section_embeddings_v2"
    assert session.vector_params["min_score"] == ec.TEXT_MODE_INDEX_SETTINGS["section_embeddings"]["min_score"]
    assert embedded == [ec.QUERY_INSTRUCTION + "termine per l'appello penale"]
    assert (info["vector_index"], info["vector_property"], info["query_mode"]) == (
        "section_embeddings_v2", "embedding_v2", "text")
    lookups._cached_embed_query.cache_clear()
