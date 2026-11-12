"""v60n (the seven-case sim): the rubrica search for single-word and prefix names -- exact on the rubrica stripped of
markers and punctuation, then prefix, then root (1); the institutions index with article-level entries (2); area-code
priority at anchoring -- a procedural code's article anchors only a procedural institution (3); the fired rules' and
tables' anchors as candidates, a topic only when graded 2 or more (4); single shot covering at most five topics, the
rest one line each under «Altri profili» (5). The model and the graph are stubbed."""

import importlib.util
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import fact_resolver as fr  # noqa: E402
from src.rag import issue_spotting  # noqa: E402

DOCS = [{"id": "cp", "name": "Codice Penale 2026"}]

# c.p. Libro II: its single-word rubriche (as the store may write them: footnote markers, a stop) and, beside them, the
# rubriche that begin with the same word or its root -- the names a search must tell apart
LIBRO_II_SINGLE = {
    "314": "Peculato", "317": "Concussione (1)", "368": "Calunnia", "369": "Autocalunnia", "385": "Evasione.",
    "422": "Strage", "423": "Incendio", "438": "Epidemia", "556": "Bigamia", "564": "Incesto", "575": "Omicidio",
    "581": "Percosse", "588": "Rissa", "595": "Diffamazione", "612": "Minaccia (1) (2)", "613-bis": "Tortura",
    "624": "Furto", "628": "Rapina", "629": "Estorsione", "631": "Usurpazione", "635": "Danneggiamento [12]",
    "640": "Truffa (1) (2) (3)", "644": "Usura", "648": "Ricettazione", "648-bis": "Riciclaggio"}
LIBRO_II_OTHERS = {
    "318": "Corruzione per l'esercizio della funzione (1)", "319": "Corruzione per un atto contrario ai doveri d'ufficio",
    "319-ter": "Corruzione in atti giudiziari", "320": "Corruzione di persona incaricata di un pubblico servizio",
    "321": "Pene per il corruttore", "322": "Istigazione alla corruzione", "423-bis": "Incendio boschivo",
    "584": "Omicidio preterintenzionale", "589": "Omicidio colposo", "612-bis": "Atti persecutori",
    "624-bis": "Furto in abitazione e furto con strappo", "626": "Furti punibili a querela dell'offeso",
    "640-bis": "Truffa aggravata per il conseguimento di erogazioni pubbliche", "640-ter": "Frode informatica",
    "646": "Appropriazione indebita", "648-ter": "Impiego di denaro, beni o utilità di provenienza illecita",
    "157": "Prescrizione. Tempo necessario a prescrivere"}
RUBRICHE = {**LIBRO_II_SINGLE, **LIBRO_II_OTHERS}


class RubricaSession:
    """The code's rubriche (RUBRICHE_QUERY); every index query finds nothing -- the rubrica step alone decides."""

    def __init__(self, rubriche=RUBRICHE):
        self.rubriche = rubriche
        self.queries = []

    def run(self, query, **params):
        self.queries.append(query)
        rows = [{"name": f"{n}.0.0", "rubrica": r} for n, r in self.rubriche.items()] if query == fr.RUBRICHE_QUERY else []
        return type("R", (), {"data": lambda self_: rows})()


@pytest.fixture
def store(monkeypatch):
    monkeypatch.setattr(fr, "_RUBRICHE", {})
    monkeypatch.setattr(dg, "embedding_model", type("E", (), {"embed_query": lambda self, t: [0.1]})())
    monkeypatch.setattr(issue_spotting, "fetch_article", lambda session, doc_id, number, *a, **k: [
        {"s": {"name": f"{number}.0.0", "titolo_articolo": RUBRICHE[number], "plain_text": f"Testo dell'art. {number}."},
         "d": {"id": doc_id}}] if number in RUBRICHE else [])


def _found(name, session=None):
    rows, log = fr.resolve(session or RubricaSession(), DOCS, {"nome": name, "codice": "c.p."})
    return [r["_article"][0] for r in rows], log


# --- 1. the rubrica search ---------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("number,rubrica", sorted(LIBRO_II_SINGLE.items()))
def test_every_single_word_rubrica_of_libro_ii_finds_its_article(store, number, rubrica):
    name = fr.rubrica_key(rubrica).capitalize()
    found, log = _found(name)
    assert found == [number], (name, found, log)
    assert log["rubrica"]["how"] == "esatta" and log["kept"][0]["as"] == "core"


def test_a_name_finds_the_rubriche_it_begins_then_its_root(store):
    found, log = _found("Corruzione")
    assert found == ["318", "319", "319-ter", "320"] and log["rubrica"]["how"] == "prefisso"     # not 321, 322
    assert _found("Appropriazione indebita")[0] == ["646"]
    assert _found("Furti")[0] == ["626"]               # «Furti punibili a querela dell'offeso» begins with it
    found, log = _found("Truffe")                     # no rubrica is «Truffe» or begins with it: the root «truff»
    assert found == ["640", "640-bis"] and log["rubrica"]["how"] == "radice"
    assert _found("il Furto")[0] == ["624"] and _found("TRUFFA.")[0] == ["640"]   # an article, case, a stop
    assert _found("Omicidio")[0] == ["575"]           # exact: never the colposo nor the preterintenzionale


def test_a_generic_single_word_is_still_never_resolved(store):
    found, log = _found("Prescrizione")
    assert found == [] and log["refused"][0]["why"] == "generic institution name: no specific word"


def test_rubrica_key_strips_markers_punctuation_and_the_leading_article():
    assert fr.rubrica_key("Truffa (1) (2)") == fr.rubrica_key("Art. 640 – Truffa.") == fr.rubrica_key("La truffa") == "truffa"
    assert fr.rubrica_key("Concussione [12]") == "concussione"
    assert fr.rubrica_key("Corruzione per l'esercizio della funzione") == "corruzione per l esercizio della funzione"
    assert fr.rubrica_matches([("1", "Furto"), ("2", "Furto in abitazione")], "furto") == ("esatta", ["1"])


def test_the_rubriche_are_read_once_per_document(store):
    session = RubricaSession()
    _found("Truffa", session)
    _found("Furto", session)
    assert session.queries.count(fr.RUBRICHE_QUERY) == 1


# --- 2. the institutions index: article-level entries ------------------------------------------------------------------------

def _build():
    spec = importlib.util.spec_from_file_location("build_institutions", HERE.parent / "scripts" / "build_institutions.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from src.rag import article_text, topic_groups
    from src.rag.rule_quote import clean_rubrica
    return module, (article_text.article_number, topic_groups.unit_title, topic_groups.generic, clean_rubrica)


UNIT_ROWS = [
    {"uid": "capo2", "level": "CAPO", "unit": "CAPO II – Dei delitti contro il patrimonio mediante frode", "name": "640",
     "rubrica": "Truffa (1)"},
    {"uid": "capo2", "level": "CAPO", "unit": "CAPO II – Dei delitti contro il patrimonio mediante frode",
     "name": "640-bis", "rubrica": "Truffa aggravata per il conseguimento di erogazioni pubbliche"},
    {"uid": "capo2", "level": "CAPO", "unit": "CAPO II – Dei delitti contro il patrimonio mediante frode", "name": "646",
     "rubrica": "Appropriazione indebita"},
    {"uid": "tit13", "level": "TITOLO", "unit": "TITOLO XIII – Dei delitti contro il patrimonio", "name": "640",
     "rubrica": "Truffa (1)"},
    {"uid": "tit13", "level": "TITOLO", "unit": "TITOLO XIII – Dei delitti contro il patrimonio", "name": "625",
     "rubrica": "Circostanze aggravanti"},
    {"uid": "tit13", "level": "TITOLO", "unit": "TITOLO XIII – Dei delitti contro il patrimonio", "name": "624",
     "rubrica": "Furto"}]


def test_the_index_holds_each_article_as_a_leaf_under_its_deepest_unit():
    build, helpers = _build()
    out = build.entries(UNIT_ROWS, *helpers)
    leaves = {e["articles"][0]: e for e in out if e["level"] == "ARTICOLO"}
    assert leaves["640"] == {"name": "Truffa", "rubricated": "Truffa", "level": "ARTICOLO", "articles": ["640"],
                             "rubriche": {"640": "Truffa"}, "parent": "Delitti contro il patrimonio mediante frode",
                             "parent_level": "CAPO"}                       # the Capo, not the Titolo above it
    assert leaves["624"]["parent_level"] == "TITOLO" and "646" in leaves
    units = [(e["level"], e["articles"]) for e in out if e["level"] != "ARTICOLO"]
    assert units == [("TITOLO", ["624", "625", "640"]), ("CAPO", ["640", "640-bis", "646"])]
    single = build.entries(build.rubrica_rows([{"name": "29", "rubrica": "Azione di condanna"}]),
                           helpers[0], lambda u: helpers[3](u), lambda n: False, helpers[3])
    assert [e["level"] for e in single] == ["ARTICOLO"]          # the rubriche source: no second leaf per article


def test_the_index_channel_resolves_a_single_article_institution(monkeypatch):
    build, helpers = _build()
    index = build.entries(UNIT_ROWS, *helpers)
    monkeypatch.setattr(fr, "_rows_for", lambda session, doc, code, numbers: [{"_article": (n, code)} for n in numbers])
    rows, log = fr.index_channel(None, DOCS, "Truffa", "c.p.", index=index)
    assert [r["_article"][0] for r in rows] == ["640"] and log["level"] == "ARTICOLO"
    rows, log = fr.index_channel(None, DOCS, "delitti contro il patrimonio mediante frode", "c.p.", index=index)
    assert log["level"] == "CAPO" and [r["_article"][0] for r in rows] == ["640", "640-bis", "646"]


def test_record_check_reads_the_units_never_the_leaves():
    from src.rag import record_check
    build, helpers = _build()
    index = build.entries(UNIT_ROWS, *helpers)
    units = record_check._index_units("624 c.p.", lambda code: index)
    assert "Dei delitti contro il patrimonio" in units and "Furto" not in units


# --- 3. area-code priority at anchoring ------------------------------------------------------------------------------------

def test_a_procedural_institution_is_named_as_one():
    from src.rag import topic_groups
    for name in ("arresto in flagranza", "Sequestro preventivo", "querela", "Giudizio abbreviato", "Decreto ingiuntivo",
                 "misure cautelari personali", "competenza per territorio"):
        assert topic_groups.procedural(name), name
    for name in ("Appropriazione indebita", "complicità", "Sequestro di persona", "Truffa", "Concorso di persone nel reato",
                 "Servitù per destinazione del padre di famiglia", "Corruzione per un atto contrario ai doveri d'ufficio"):
        assert not topic_groups.procedural(name), name


def _search(found, procedural_rubrica=()):
    """first_read_search with each (name, code) answering found[(name, code)] -- (channel, [numbers])."""
    from src.rag import interactive_flow as flow

    def channel(kind):
        def run(own, documents, query, code, nouns=""):
            ch, numbers = found.get((query, code), ("", []))
            return ([{"_article": (n, code)} for n in numbers], {}) if ch == kind else ([], {})
        return run

    def resolve(own, documents, spec):
        ch, numbers = found.get((spec["nome"], spec["codice"]), ("", []))
        return ([{"_article": (n, spec["codice"])} for n in numbers], {}) if ch == "rubrica" else ([], {})
    names = sorted({name for name, _ in found})
    return flow.first_read_search(None, "db", [], names, ["c.p.", "c.p.p."], session=object(), resolve=resolve,
                                  text_search=channel("testo"), index_search=channel("indice"))


def test_a_substantive_institution_never_takes_a_procedural_codes_hit():
    rows, log = _search({("Appropriazione indebita", "c.p."): ("rubrica", ["646"]),
                         ("Appropriazione indebita", "c.p.p."): ("testo", ["381"]),          # the arrest article
                         ("Arresto in flagranza", "c.p.p."): ("rubrica", ["381"]),
                         ("Incompatibilità del giudice", "c.p.p."): ("rubrica", ["34"])})
    assert log["hits"]["Appropriazione indebita"] == ["646 c.p."]
    assert log["area_code"]["Appropriazione indebita"] == {"procedural": False, "dropped": ["381 c.p.p."]}
    assert log["hits"]["Arresto in flagranza"] == ["381 c.p.p."]                     # procedural by its name
    assert log["hits"]["Incompatibilità del giudice"] == ["34 c.p.p."]               # the store rubricates it there only
    assert log["area_code"]["Incompatibilità del giudice"] == {"procedural": True, "why": "rubrica solo in c.p.p."}
    tagged = {r["_first_read"] for r in rows if r["_article"] == ("381", "c.p.p.")}
    assert tagged == {"Arresto in flagranza"}                   # never the substantive one's anchor


def _topic(title, code, label, grade=2):
    return {"title": title, "code": code, "grades": {label: grade}, "articles": [label], "unit": "", "hypotheses": []}


def test_a_procedural_topic_anchors_only_a_procedural_institution():
    from src.rag import topic_groups
    rubriche = {"381 c.p.p.": "Arresto facoltativo in flagranza per appropriazione indebita",
                "646 c.p.": "Appropriazione indebita", "640 c.p.": "Truffa"}
    topics = [_topic("Arresto in flagranza", "c.p.p.", "381 c.p.p."), _topic("Truffa", "c.p.", "640 c.p.")]
    hyp = {"line": "principale", "claim": "Mevio si appropria del bene", "institutions": ["Appropriazione indebita"]}
    unanchored, log = topic_groups.anchor_by_similarity(topics, {"topics": [[], []]}, [hyp], embed=lambda t: [[1.0]] * len(t),
                                                        rubriche=rubriche, codes=["c.p.", "c.p.p."])
    assert unanchored and not topics[0]["hypotheses"]          # the c.p.p. rubrica shares its words: still never there
    topics.append(_topic("Appropriazione indebita", "c.p.", "646 c.p."))
    unanchored, _ = topic_groups.anchor_by_similarity(topics, {"topics": [[], [], []]}, [hyp], rubriche=rubriche,
                                                      codes=["c.p.", "c.p.p."])
    assert not unanchored and topics[2]["hypotheses"] and not topics[0]["hypotheses"]
    arrest = {"line": "subordinata", "claim": "x", "institutions": ["arresto facoltativo in flagranza"]}
    for t in topics:
        t["hypotheses"] = []
    topic_groups.anchor_by_similarity(topics, {"topics": [[], [], []]}, [arrest], rubriche=rubriche, codes=["c.p."])
    assert topics[0]["hypotheses"]                               # a procedural institution: its code's topic
    assert topic_groups.attach_hypotheses(topics[:2], [hyp], rubriche) == [hyp]      # the words path: the same rule


def test_on_a_tie_the_areas_substantive_code_anchors():
    from src.rag import topic_groups
    rubriche = {"57 c.c.": "Sequestro della cosa", "58 c.p.": "Sequestro della cosa"}       # same words, two codes
    topics = [_topic("A", "c.c.", "57 c.c."), _topic("B", "c.p.", "58 c.p.")]
    hyp = {"line": "principale", "claim": "x", "institutions": ["sequestro della cosa"]}
    topic_groups.anchor_by_similarity(topics, {"topics": [[], []]}, [hyp], rubriche=rubriche, codes=["c.p.", "c.p.p."])
    assert topics[1]["hypotheses"] and not topics[0]["hypotheses"]


# --- 4. the fired rules' and tables' anchors: candidates, graded like any other ---------------------------------------------

def test_the_rules_anchors_become_candidates(monkeypatch):
    from src.rag import article_text, interactive_flow as flow
    rule_topics = [{"title": "Destinazione del padre di famiglia", "anchor": "1062 c.c.", "source": "regime dpf"},
                   {"title": "Passaggio coattivo", "anchor": "1051 c.c.", "source": "regime pc"},
                   {"title": "Usucapione", "anchor": "1158 c.c.", "source": "regime u"},
                   {"title": "Azione confessoria", "anchor": "1079 c.c.", "source": "rule ac"}]
    article_rows = [{"_article": ("1062", "c.c."), "s": {}}]
    regime_rows = [{"_article": ("1051", "c.c."), "s": {"plain_text": "Il proprietario..."}, "_procedural_tool": True}]
    monkeypatch.setattr(dg, "_fetch_any", lambda session, documents, refs, store=None: [
        {"_article": ("1158", "c.c."), "s": {"plain_text": "La proprietà..."}}] if ("1158", "c.c.") in refs else [])
    monkeypatch.setattr(article_text, "fill_texts", lambda *a, **k: None)
    added, log = flow.rule_anchor_rows(None, [], rule_topics, article_rows, regime_rows)
    assert [r["_article"] for r in added] == [("1051", "c.c."), ("1158", "c.c.")]
    assert all(r["_rule_anchor"] for r in added) and "_procedural_tool" not in added[0]
    assert article_rows[0]["_rule_anchor"] == "regime dpf"
    assert (log["already"], log["from_tools"], log["read"], log["unread"]) == (
        ["1062 c.c."], ["1051 c.c."], ["1158 c.c."], ["1079 c.c."])
    assert sorted(log["keep"]) == ["1051 c.c.", "1062 c.c.", "1158 c.c."]


def test_a_rules_anchor_makes_a_topic_only_when_graded_two_or_more():
    from src.rag import topic_groups
    rows = [{"_article": ("1051", "c.c."), "_grade": 2, "s": {"titolo_articolo": "Passaggio coattivo"}},
            {"_article": ("1079", "c.c."), "_grade": 1, "s": {"titolo_articolo": "Accertamento della servitù"}}]
    rule_topics = [{"title": "Passaggio coattivo", "anchor": "1051 c.c.", "matched": [], "questions": ["accesso"]},
                   {"title": "Azione confessoria", "anchor": "1079 c.c.", "matched": []}]
    built = topic_groups.build(rows, {}, (), rule_topics)
    assert [t["anchor"] for t in built["topics"]] == ["1051 c.c."] and built["topics"][0]["questions"] == ["accesso"]
    assert built["lost_rules"] == ["Azione confessoria"] and built["others"][0]["articles"] == ["1079 c.c."]


# --- 5. single shot: at most five topics, the rest under «Altri profili» --------------------------------------------------

def test_single_shot_covers_at_most_five_topics_by_their_reasons(monkeypatch):
    from src.rag import interactive_flow as flow
    logged = []
    monkeypatch.setattr(flow, "_log", lambda event, data, *a: logged.append((event, data)))

    def topic(n, grade=2, line=None):
        return {"title": f"T{n}", "articles": [f"{n} c.c."], "grades": {f"{n} c.c.": grade},
                "hypotheses": [{"line": line}] if line else []}
    topics = [topic(1), topic(2, 3), topic(3, 3), topic(4, line="principale"), topic(5, 3), topic(6, 3), topic(7, 3),
              topic(8)]
    state = {"topics": topics, "text": "Il cliente chiede dell'art. 8 c.c.", "request": "", "analysis": {}}
    assert flow.covered_topics(state) == [8, 4, 2, 3, 5]      # v60q: the named one, the main line, then grade 3, in
    entry = next(d for e, d in logged if e == "single_shot_topics")                            # that order
    assert entry["why"]["8"] == ["articolo nominato"] and entry["why"]["4"] == ["linea principale"]
    assert entry["beyond_cap"] == [6, 7] and entry["rest"] == [1, 6, 7] and entry["order"] == [8, 4, 2, 3, 5, 6, 7, 1]
    state["topics"] = [topic(1), topic(2)]                    # v60q: none qualifies -- every topic, in its order
    state["text"] = "Il caso."
    assert flow.covered_topics(state) == [1, 2]


def test_a_single_word_with_no_rubrica_match_needs_its_root_in_a_heading(store, monkeypatch):
    rubriche = {"110": "Pena per coloro che concorrono nel reato", "200": "Completamento della procedura",
                "378": "Favoreggiamento del complice"}
    monkeypatch.setitem(RUBRICHE, "110", rubriche["110"])
    monkeypatch.setitem(RUBRICHE, "200", rubriche["200"])
    monkeypatch.setitem(RUBRICHE, "378", rubriche["378"])

    class Indexes(RubricaSession):              # the heading index offers both
        def run(self, query, **params):
            if query == fr.RUBRICHE_QUERY:
                return super().run(query, **params)
            rows = [{"name": "200.0.0", "heading": rubriche["200"]}, {"name": "378.0.0", "heading": rubriche["378"]}] \
                if "vector.queryNodes" in query else []
            return type("R", (), {"data": lambda self_: rows})()
    found, log = _found("Complicità", Indexes(rubriche))     # no rubrica is it, begins with it or has its root first
    assert not log.get("rubrica") and found == ["378"]       # «complice» has its root: a core; «Completamento» never
    assert {"article": "200 c.p.", "why": "heading and text share no specific word"} in log["refused"]
