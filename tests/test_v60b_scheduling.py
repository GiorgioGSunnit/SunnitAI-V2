"""v60b: round 1's collection rescheduled -- the same reads, at once. Each test runs a step the old way (no driver: one
session, one query after another) and the new way (a driver: one session per concurrent read) and asserts the same
result; the batched topic_step reads are checked to answer exactly what the per-item reads answered."""

import os
import threading
import time

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import topic_step  # noqa: E402


class Driver:
    """Sessions that answer like the given session, counted, with the threads that used them."""

    def __init__(self, session):
        self.inner, self.opened, self.threads = session, 0, set()
        self.lock = threading.Lock()

    def session(self, **kwargs):
        driver = self

        class Own:
            def __enter__(self_inner):
                with driver.lock:
                    driver.opened += 1
                    driver.threads.add(threading.get_ident())
                return driver.inner

            def __exit__(self_inner, *exc):
                return False
        return Own()


def test_graph_parallel_keeps_the_order_hands_back_errors_and_needs_a_driver():
    driver = Driver(object())

    def job(i):
        def run(own):
            time.sleep(0.01 * (5 - i))
            if i == 2:
                raise ValueError("boom")
            return i
        return run
    out = dg.graph_parallel(driver, "db", [job(i) for i in range(5)])
    assert out[:2] + out[3:] == [0, 1, 3, 4] and isinstance(out[2], ValueError) and driver.opened == 5
    assert dg.graph_parallel(None, "db", [job(0)]) == [] and dg.graph_parallel(object(), "db", [job(0)]) == []


class CodeSearchSession:
    def run(self, query, **params):
        terms = params["t"]
        n = sum(map(ord, terms)) % 50
        rows = [{"doc": "Codice Civile 2026", "sec": f"{n + i}.0.0", "score": 1.0} for i in range(6)]

        class Result:
            def data(self_inner):
                time.sleep(0.02)
                return rows
        return Result()


def test_the_code_searches_on_a_pool_find_what_the_serial_ones_found():
    analysis = {"questioni": [{"titolo": f"questione {i}", "parole_chiave": [f"parola{i}", "danno"]} for i in range(8)]}
    serial = dg._search_code_articles(CodeSearchSession(), analysis)
    driver = Driver(CodeSearchSession())
    t0 = time.time()
    pooled = dg._search_code_articles(CodeSearchSession(), analysis, driver, "db")
    assert pooled == serial and len(serial) > 8 and driver.opened == 8
    assert time.time() - t0 < 0.15                       # eight 20 ms queries at once, not 160 ms


@pytest.fixture
def fusion_env(monkeypatch):
    calls = {"distill": 0}

    def v2(*a, **k):
        return {"first_rulings": [], "similar": [], "keep": [], "named": [], "candidates": [], "chosen": [],
                "selected": [("1", "c.c.")], "article_rows": [], "spotted": [], "institutions_resolved": [],
                "heading_matched": []}
    monkeypatch.setattr(dg, "_select_articles_v2", v2)

    def bullets(case):
        calls["distill"] += 1
        return [{"testo": "uno"}, {"testo": "due"}]
    monkeypatch.setattr(dg, "_fusion_distill_bullets", bullets)

    def search(session, text, index, k):
        time.sleep(0.01)
        n = sum(map(ord, text)) % 40
        return [(str(n + i), "c.c.") for i in range(3)]
    monkeypatch.setattr(dg, "_fusion_text_search", search)
    monkeypatch.setattr(dg, "_frame_queries", lambda frame: ["domanda uno", "domanda due", "domanda tre"])
    import src.rag.doc_lookup as doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs, store=None: [
        {"d": {"id": "doc"}, "s": {"name": f"{n}.0.0", "plain_text": f"testo {n}"}, "_article": (n, c)} for n, c in refs])
    import src.rag.nodes.retrieval as retrieval_module
    monkeypatch.setattr(retrieval_module, "broaden_article_context",
                        lambda state, driver, database: {"raw_result": state["raw_result"]})
    return calls


def test_the_fusion_branches_at_once_select_what_they_selected_one_after_another(fusion_env):
    analysis = {"questioni": [], "frame": {"x": 1}}
    serial = dg._select_articles_fusion(object(), object(), "db", "fatti", "", dict(analysis))
    driver = Driver(object())
    pooled = dg._select_articles_fusion(object(), driver, "db", "fatti", "", dict(analysis))
    assert pooled["selected"] == serial["selected"] and len(serial["selected"]) > 5 and driver.opened >= 6
    assert [r.get("_signals") for r in pooled["article_rows"]] == [r.get("_signals") for r in serial["article_rows"]]


def test_the_case_branches_started_beside_the_analysis_are_taken_not_run_again(fusion_env):
    analysis = {"questioni": [], "frame": {"x": 1}}
    serial = dg._select_articles_fusion(object(), object(), "db", "fatti", "", dict(analysis))
    fusion_env["distill"] = 0
    driver = Driver(object())
    with dg.prefetch_case_branches(driver, "db", "fatti", ""):
        time.sleep(0.05)                                   # "the analysis"
        early = dg._select_articles_fusion(object(), driver, "db", "fatti", "", dict(analysis))
    assert early["selected"] == serial["selected"] and fusion_env["distill"] == 1
    with dg.prefetch_case_branches(driver, "db", "altro caso", ""):        # another case: not taken
        dg._select_articles_fusion(object(), driver, "db", "fatti", "", dict(analysis))
    assert fusion_env["distill"] == 3


def test_the_institutions_resolved_at_once_are_read_in_the_questions_order(monkeypatch):
    def resolve(session, documents, istituto, parole):
        time.sleep(0.03 if istituto["nome"].endswith("1") else 0.0)
        n = istituto["nome"][-1]
        return [{"d": {"id": "doc"}, "s": {"name": f"{n}0.0.0"}, "_article": (f"{n}0", "c.c.")},
                {"d": {"id": "doc"}, "s": {"name": f"{n}1.0.0"}, "_article": (f"{n}1", "c.c.")}]
    monkeypatch.setattr(dg, "_resolve_institution", resolve)
    monkeypatch.setattr(dg, "_select_articles", lambda *a, **k: {
        "first_rulings": [], "similar": [], "keep": [], "named": [], "candidates": [], "chosen": [], "selected": [],
        "article_rows": [], "spotted": []})
    import src.rag.doc_lookup as doc_lookup
    monkeypatch.setattr(doc_lookup, "_fetch_doc_names", lambda driver, database: [])
    monkeypatch.setattr(dg, "_required_rows", lambda session, analysis, refs, store=None:
                        analysis.setdefault("rule_required", []) and [])

    def analysis():
        return {"questioni": [{"titolo": f"q{i}", "istituti": [{"nome": f"istituto {i}{j}"} for j in (1, 2)]}
                              for i in range(4)]}
    serial = dg._select_articles_v2(object(), object(), "db", "fatti", "", analysis())
    driver = Driver(object())
    pooled = dg._select_articles_v2(object(), driver, "db", "fatti", "", analysis())
    assert [r["_article"] for r in pooled["article_rows"]] == [r["_article"] for r in serial["article_rows"]]
    assert pooled["selected"] == serial["selected"] and driver.opened == 8


# --- topic_step: the batched reads ------------------------------------------------------------------------------------

DOCS = [{"id": "cc", "name": "Codice Civile 2026"}, {"id": "cp", "name": "Codice Penale"}]
RUBRICHE = {"cc": {"1051": "Passaggio coattivo", "1052": "Passaggio coattivo a favore di fondo non intercluso",
                   "1062": "Destinazione del padre di famiglia", "1061": "Servitù non apparenti",
                   "1158": "Usucapione dei beni immobili"}}
UNITS = {("cc", "1051"): {"SEZIONE": ["1051", "1052", "1053"], "CAPO": ["1051", "1052", "1062"],
                          "LIBRO": "LIBRO III"},
         ("cc", "1062"): {"SEZIONE": ["1061", "1062"], "CAPO": ["1051", "1062"], "LIBRO": "LIBRO III"}}
REFERS = {("cc", "1051"): [("1052", 2)], ("cc", "1062"): [("1061", 1), ("1051", 9)]}


class GraphSession:
    """Answers both the per-item queries and the batched ones from the same tables; counts the queries."""

    def __init__(self):
        self.queries = 0

    def run(self, query, **p):
        self.queries += 1
        rows = []
        if "UNWIND $words" in query:
            for w in p["words"]:
                rows += [{"key": w, **r} for r in self._rubrica(w, p["docs"])]
        elif "toLower(s.titolo_articolo) CONTAINS $word" in query:
            rows = self._rubrica(p["word"], p["docs"])
        elif "UNWIND $firsts" in query or "STARTS WITH $first" in query:
            for f in p.get("firsts") or [p.get("first")]:
                rows += [{"key": f, "doc": d, "section": f"{n}.0.0", "rubrica": r} for d, by in RUBRICHE.items()
                         for n, r in by.items() if r.lower().startswith(f)]
        elif "UNWIND $items" in query and "u.name AS name" in query:
            rows = [{"qdoc": i["doc"], "qnum": i["num"], "qlevel": i["level"],
                     "name": UNITS.get((i["doc"], i["num"]), {}).get(i["level"])} for i in p["items"]
                    if UNITS.get((i["doc"], i["num"]), {}).get(i["level"])]
        elif "u.name AS name" in query:
            name = UNITS.get((p["doc"], p["num"]), {}).get(p["level"])
            rows = [{"name": name}] if name else []
        elif "UNWIND $items" in query and "HAS_ARTICLE]->(t:Section)" in query:
            for i in p["items"]:
                rows += [{"qdoc": i["doc"], "qnum": i["num"], "qlevel": i["level"], "doc": "Codice Civile 2026",
                          "section": f"{n}.0.0"} for n in UNITS.get((i["doc"], i["num"]), {}).get(i["level"], [])]
        elif "HAS_ARTICLE]->(t:Section)" in query:
            rows = [{"doc": "Codice Civile 2026", "section": f"{n}.0.0"}
                    for n in UNITS.get((p["doc"], p["num"]), {}).get(p["level"], [])]
        elif "UNWIND $items" in query and "REFERS_TO]->(t" in query:
            for i in p["items"]:
                rows += [{"qdoc": i["doc"], "qnum": i["num"], "doc": "Codice Civile 2026", "section": f"{n}.0.0",
                          "inbound": c} for n, c in REFERS.get((i["doc"], i["num"]), [])]
        elif "REFERS_TO]->(t" in query:
            rows = [{"doc": "Codice Civile 2026", "section": f"{n}.0.0", "inbound": c}
                    for n, c in REFERS.get((p["doc"], p["num"]), [])]
        elif "UNWIND $refs" in query:
            rows = [{"doc": a["doc"], "num": a["num"], "rubrica": RUBRICHE.get(a["doc"], {}).get(a["num"], "")}
                    for a in p["refs"]]

        class Result:
            def data(self_inner):
                return rows
        return Result()

    @staticmethod
    def _rubrica(word, docs):
        return [{"doc": "Codice Civile 2026", "section": f"{n}.0.0", "rubrica": r} for d, by in RUBRICHE.items()
                if d in docs for n, r in by.items() if word in r.lower()]


def _reads(links):
    refs = [("1051", "c.c."), ("1062", "c.c.")]
    return ([links.rubrica_matches(n, None) for n in ("passaggio coattivo", "padre di famiglia")],
            [links.unit(r, lv) for r in refs for lv in ("SEZIONE", "CAPO")],
            [links.refers_out(r) for r in refs], [links.unit_name(r, "LIBRO") for r in refs],
            links.rubrica_count("Passaggio coattivo"))


def test_the_batched_topic_reads_answer_what_the_per_item_reads_answered():
    one = topic_step.GraphLinks(GraphSession(), DOCS)
    expected = _reads(one)
    session = GraphSession()
    batched = topic_step.GraphLinks(session, DOCS)
    refs = [("1051", "c.c."), ("1062", "c.c.")]
    batched.prefetch_rubriche_matches(["passaggio coattivo", "padre di famiglia"], None)
    batched.prefetch_counts(["Passaggio coattivo"])
    batched.prefetch_units(refs, ("SEZIONE", "CAPO"))
    batched.prefetch_refers_out(refs)
    batched.prefetch_unit_names(refs, "LIBRO")
    before = session.queries
    assert before == 5
    assert _reads(batched) == expected and session.queries == before          # every read from the caches
