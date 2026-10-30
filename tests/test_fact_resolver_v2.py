"""FACT_ISSUES_RESOLVER_V2_ENABLED: precise naming (fact_issues), the resolver by headings (fact_resolver), T1 and the
plan placement (evidence_tiers, plan_then_write), and fact_loss_report's v2 outcomes. Article texts here are shortened
or invented; no model, no database."""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

import pytest  # noqa: E402

from src.rag import defensive_generation as dg  # noqa: E402
from src.rag import evidence_tiers  # noqa: E402
from src.rag import fact_issues as fi  # noqa: E402
from src.rag import fact_resolver as fr  # noqa: E402
from src.rag import issue_spotting  # noqa: E402
from src.rag import plan_then_write as ptw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DOCS = [{"id": "cc", "name": "Codice Civile 2026"}]
ARTICLES = {
    "1490": ("Garanzia per i vizi della cosa venduta", "Il venditore è tenuto a garantire che la cosa venduta sia "
             "immune da vizi che la rendano inidonea all'uso a cui è destinata."),
    "1492": ("Effetti della garanzia", "Il compratore può domandare a sua scelta la risoluzione del contratto ovvero "
             "la riduzione del prezzo."),
    "1495": ("Termini e condizioni per l'azione", "Il compratore decade dal diritto alla garanzia, se non denunzia i "
             "vizi al venditore entro otto giorni dalla scoperta."),
    "2462": ("Responsabilità", "Nella società a responsabilità limitata per le obbligazioni sociali risponde soltanto "
             "la società con il suo patrimonio."),
    "2689": ("Usucapione", "Deve essere trascritta la sentenza che accerta l'usucapione."),
    "1158": ("Usucapione dei beni immobili e dei diritti reali immobiliari", "La proprietà dei beni immobili si "
             "acquista in virtù del possesso continuato per venti anni."),
}


class FakeSession:
    """Answers the resolver's three candidate queries from ARTICLES: the heading index by `vector`, the heading full
    text by `title`, the text full text by `text` (lists of article numbers)."""

    def __init__(self, vector=(), title=(), text=()):
        self.lists = {"vector": vector, "title": title, "text": text}
        self.queries = []

    def run(self, query, **params):
        kind = "vector" if "vector.queryNodes" in query else "title" if "$index, $lucene" in query else "text"
        self.queries.append((kind, params))
        rows = [{"name": f"{n}.0.0", "heading": ARTICLES[n][0]} for n in self.lists[kind]]
        return type("R", (), {"data": lambda self_: rows})()


@pytest.fixture
def v2(monkeypatch):
    monkeypatch.setenv(fi.FLAG, "true")
    monkeypatch.setenv(fi.RESOLVER_V2_FLAG, "true")
    monkeypatch.setattr(dg, "embedding_model", type("E", (), {"embed_query": lambda self, t: [0.1, 0.2]})())
    monkeypatch.setattr(issue_spotting, "fetch_article", lambda session, doc_id, number, *a, **k: [
        {"s": {"name": f"{number}.0.0", "titolo_articolo": ARTICLES[number][0], "plain_text": ARTICLES[number][1]},
         "d": {"id": doc_id}}] if number in ARTICLES else [])


# --- 1. naming ------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("name,why", [
    ("Risarcimento danni", "generic"), ("Responsabilità medica", "generic"), ("Decisione amministrativa", "generic"),
    ("Dolus eventualis", "latin"), ("Diligentia", "latin"), ("responsabilità contrattuale", "generic"),
    ("inadempimento del conduttore", None), ("risoluzione della locazione per morosità", None),
    ("responsabilità contrattuale della struttura sanitaria", None), ("giudizio di ottemperanza", None),
    ("garanzia per i vizi della cosa venduta", None), ("responsabilità del precettore", None),
])
def test_generic_and_latin_names(name, why):
    assert fi.generic_name(name) == why


def test_the_prompt_names_precisely_and_a_generic_name_is_renamed_once(monkeypatch, v2):
    assert fi.NAMING_RULE in fi.system_for(8) and "giudizio di ottemperanza" in fi.system_for(8)
    calls = []

    def chat(messages, max_tokens, **k):
        calls.append(messages[1].content)
        if len(calls) == 1:
            return {"istituti": [
                {"fatto": "il conduttore non paga da sei mesi", "nome": "Risarcimento danni", "codice": "c.c."},
                {"fatto": "la lavatrice si rompe", "nome": "garanzia per i vizi della cosa venduta", "codice": "c.c."},
                {"fatto": "l'alunno cade in cortile", "nome": "Diligentia", "codice": "c.c."}]}
        assert fi.RETRY_TITLE in messages[1].content and "Risarcimento danni" in messages[1].content
        return {"istituti": [{"fatto": "il conduttore non paga da sei mesi",
                              "nome": "risoluzione della locazione per morosità", "codice": "c.c."},
                             {"fatto": "l'alunno cade in cortile", "nome": "Culpa in vigilando", "codice": "c.c."}]}
    monkeypatch.setattr(dg, "_chat_json", chat)
    events = []
    monkeypatch.setattr(fi, "vlog", lambda e, d=None, **_: events.append((e, d)))
    names = [i["nome"] for i in fi.fact_issues("caso", {"facts": ["a"]}, None)]
    assert names == ["risoluzione della locazione per morosità", "garanzia per i vizi della cosa venduta"]
    log = [d for e, d in events if e == "case_analysis_fact_issues_names"][-1]
    assert log["dropped"] == [{"nome": "Diligentia", "fatto": "l'alunno cade in cortile"}]
    assert log["still_generic"] == ["Culpa in vigilando"] and len(calls) == 2


# --- 2. resolution ----------------------------------------------------------------------------------------------------

def test_core_and_related_articles_within_the_code(v2):
    session = FakeSession(vector=["1490", "1492", "2462"], title=["1490"], text=["1495", "1490"])
    rows, log = fr.resolve(session, DOCS, {"nome": "garanzia per i vizi della cosa venduta", "codice": "c.c."})
    assert [r["_article"] for r in rows] == [("1490", "c.c."), ("1495", "c.c.")]
    assert [(k["article"], k["as"]) for k in log["kept"]] == [("1490 c.c.", "core"), ("1495 c.c.", "related")]
    refused = {x["article"]: x["why"] for x in log["refused"]}
    assert refused["2462 c.c."] == "outside the institution's chapter"          # the sale's chapter is known
    assert refused["1492 c.c."] == "heading and text share no specific word"
    assert rows[0]["_heading_match"] and rows[0]["_fact_institution"] == "garanzia per i vizi della cosa venduta"
    assert [k for k, _ in session.queries] == ["vector", "title", "text"]
    assert all(p.get("doc_id") == "cc" for _, p in session.queries)          # always within the named code


def test_refused_outside_the_chapter_and_never_through_a_generic_word(v2):
    rows, log = fr.resolve(FakeSession(vector=["2689", "1158"]), DOCS,
                           {"nome": "usucapione dei beni immobili", "codice": "c.c."})
    assert [r["_article"] for r in rows] == [("1158", "c.c.")]
    assert {"article": "2689 c.c.", "why": "outside the institution's chapter"} in log["refused"]
    session = FakeSession(vector=["2462"])
    rows, log = fr.resolve(session, DOCS, {"nome": "Risarcimento danni", "codice": "c.c."})
    assert rows == [] and session.queries == [] and "generic institution name" in log["refused"][0]["why"]
    rows, log = fr.resolve(FakeSession(vector=["1490"]), DOCS, {"nome": "garanzia per i vizi", "codice": "c.zz."})
    assert rows == [] and "not resolved to a document" in log["refused"][0]["why"]


# --- 4. tier and plan -------------------------------------------------------------------------------------------------

def test_a_fact_institution_row_is_t1_and_takes_no_forced_slot(monkeypatch, v2):
    row = {"_article": ("1495", "c.c."), "_fact_institution": "garanzia per i vizi della cosa venduta", "_grade": 3}
    assert evidence_tiers.tier_of(row, set(), set(), set()) == (evidence_tiers.T1, "fact institution")
    other = {"_article": ("1218", "c.c."), "_grade": 3}
    monkeypatch.setattr(dg, "vlog", lambda *a, **k: None)
    assert list(ptw._forced([row, other])) == [("1218", "c.c.")]
    monkeypatch.delenv(fi.RESOLVER_V2_FLAG)
    assert set(ptw._forced([row, other])) == {("1218", "c.c."), ("1495", "c.c.")}


def test_placed_under_its_institution_s_issue(v2):
    plan = {"questioni": [{"titolo": "Vizi della lavatrice", "norme": [("1490", "c.c.")], "pronunce": [],
                           "argomento": "", "obiezione": "", "replica": ""}]}
    analysis = {"fact_institutions": [
        {"nome": "garanzia per i vizi della cosa venduta", "fatto": "la lavatrice si rompe",
         "refs": [("1490", "c.c."), ("1495", "c.c.")]},
        {"nome": "risoluzione della locazione per morosità", "fatto": "il conduttore non paga",
         "refs": [("1453", "c.c."), ("9999", "c.c.")]}]}
    rows = [{"_article": ref} for ref in [("1490", "c.c."), ("1495", "c.c."), ("1453", "c.c.")]]
    out, log = ptw.place_fact_institutions(plan, analysis, rows)
    assert out["questioni"][0]["norme"] == [("1490", "c.c."), ("1495", "c.c.")]           # merged: it held 1490
    assert out["questioni"][1]["titolo"] == "Risoluzione della locazione per morosità"
    assert out["questioni"][1]["norme"] == [("1453", "c.c.")] and "il conduttore non paga" in out["questioni"][1]["argomento"]
    assert log["reached"] == ["1490 c.c.", "1495 c.c.", "1453 c.c.", "9999 c.c."]
    assert log["placed"] == ["1490 c.c.", "1495 c.c.", "1453 c.c."]                     # 9999: no row, not placed
    assert log["merged_into"] == {"garanzia per i vizi della cosa venduta": "Vizi della lavatrice"}
    assert plan["questioni"][0]["norme"] == [("1490", "c.c.")]                             # the plan given unchanged


# --- 3. the loss report's v2 split ----------------------------------------------------------------------------------

def test_the_loss_report_splits_v2_outcomes():
    root = tempfile.mkdtemp()
    run = os.path.join(root, "cand")
    os.makedirs(os.path.join(run, "cases", "c1"))
    events = [["case_analysis_fact_issues", {"institutions": [
        {"nome": "garanzia per i vizi della cosa venduta", "fatto": "la lavatrice si rompe dopo tre mesi"},
        {"nome": "responsabilità del precettore", "fatto": "l'alunno cade nel cortile della scuola"}]}],
        ["case_analysis_fact_institution", {"institution": "garanzia per i vizi della cosa venduta", "fonte": "fatto",
                                            "resolver": "v2", "kept": ["1490 c.c."], "refused": [
                                                {"article": "1495 c.c.", "why": "beyond the 3 kept"}]}],
        ["case_analysis_fact_institution", {"institution": "responsabilità del precettore", "fonte": "fatto",
                                            "resolver": "v2", "kept": [], "refused": []}]]
    json.dump({"id": "c1", "events": events}, open(os.path.join(run, "cases", "c1.json"), "w"))
    lost = [{"article": "art. 1495 cc", "fact": "la lavatrice si rompe dopo tre mesi", "captured": ["x"],
             "funnel": {"lost_at": "resolved"}, "capture_kind": "fact"},
            {"article": "art. 2048 cc", "fact": "l'alunno cade nel cortile della scuola", "captured": ["x"],
             "funnel": {"lost_at": "resolved"}, "capture_kind": "fact"}]
    audit = os.path.join(root, "audit.json")
    json.dump({"cases": [{"run": "cand", "id": "c1", "understanding": {"lost": lost}}]}, open(audit, "w"))
    out = subprocess.run([sys.executable, str(ROOT / "scripts" / "fact_loss_report.py"), audit, run], cwd=ROOT,
                         capture_output=True, text=True, env={**os.environ}).stdout
    assert "| v2: refused | 1 | 50% |" in out and "| v2: not a candidate, resolved to nothing | 1 | 50% |" in out
    assert "v2 refusals by reason: beyond the 3 kept 1." in out
