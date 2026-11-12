"""v60s, general rules: an offence issue keeps an offence article or goes with all its articles (1); the anchor comma is
co. 1 unless the topic names one (2: tests/test_v60r.py); the element grader reads the narrative, never asks for proof
(3); §1 and §4 assert only what the record allows, the rest «da valutare» (4); no bare article line under a §2 heading
(5); the procedural store free of the corpus' markers (6); the single-shot client line, the client known too (7). The
model and the graph are stubbed."""

import os
import sys
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import answer_ledger, issue_records, procedural_facts, writer_rules  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402
import test_interactive_flow as base  # noqa: E402
from test_v60k import WRITER, _single  # noqa: E402

NORMS = flow.ANSWER_TITLES[1]
LEAD, PROC = flow.ANSWER_TITLES[0], flow.ANSWER_TITLES[3]
PROVATO, DA_PROVARE, NON_RISULTA = C.STATUS_PROVATO, C.STATUS_DA_PROVARE, C.STATUS_NON_RISULTA
PENA = {"pena": "reclusione da sei mesi a tre anni"}


def _req(comma, status, text="elemento"):
    return {"kind": "requisito", "comma": comma, "text": text, "status": status, "supporting": [],
            "contradicting": [], "question": ""}


# --- 1. an offence issue keeps an offence article, or goes -------------------------------------------------------------------

def _offence_state(verdict_640=DA_PROVARE):
    return {"records": {"640 c.p.": {"verdict": verdict_640}, "61 c.p.": {"verdict": DA_PROVARE},
                        "336 c.p.p.": {"verdict": DA_PROVARE}}}


TRUFFA = {"title": "Dei delitti contro il patrimonio mediante frode", "anchor": "640 c.p.",
          "articles": ["640 c.p.", "61 c.p.", "336 c.p.p."], "planned": 3}
SERVITU = {"title": "Servitù", "anchor": "1062 c.c.", "articles": ["1062 c.c."], "planned": 1}
ROWS = [{"_article": ("640", "c.p."), "_proc": PENA}, {"_article": ("61", "c.p.")}, {"_article": ("336", "c.p.p.")}]


def test_an_offence_issue_without_its_offence_goes_with_all_its_articles(monkeypatch):
    logged = []
    monkeypatch.setattr(flow, "_log", lambda event, data, *a: logged.append((event, data)))
    assert flow.offence_issue(TRUFFA) and not flow.offence_issue(SERVITU)
    assert not flow.offence_issue({"anchor": "61 c.p."})                     # Libro I: no offence unit
    kept = flow.surviving_issues(_offence_state(), [TRUFFA, SERVITU], ROWS, set())
    assert kept == [TRUFFA, SERVITU] and not logged                          # 640: a pena, «da provare»: it stays
    kept = flow.surviving_issues(_offence_state(issue_records.NOT_APPLICABLE), [TRUFFA, SERVITU], ROWS, set())
    assert kept == [SERVITU]                                                 # 61 and 336 go with it
    assert dict(logged)["issue_dropped"] == {"issue": TRUFFA["title"], "anchor": "640 c.p.",
                                             "offences": {"640 c.p.": "non applicabile"},
                                             "articles": ["640 c.p.", "61 c.p.", "336 c.p.p."]}
    assert flow.surviving_issues(_offence_state(), [TRUFFA], ROWS, {("640", "c.p.")}) == []      # abrogated
    no_pena = [{"_article": ("640", "c.p."), "_proc": {"procedibilita": "a querela"}}] + ROWS[1:]
    assert flow.surviving_issues(_offence_state(), [TRUFFA], no_pena, set()) == []
    assert dict(logged)["issue_dropped"]["offences"] == {"640 c.p.": "nessuna pena nel testo"}


def test_a_second_offence_with_a_pena_keeps_the_issue():
    both = {**TRUFFA, "articles": ["640 c.p.", "646 c.p.", "336 c.p.p."]}
    rows = ROWS + [{"_article": ("646", "c.p."), "_proc": PENA}]
    state = _offence_state(issue_records.NOT_APPLICABLE)
    state["records"]["646 c.p."] = {"verdict": DA_PROVARE}
    assert flow.surviving_issues(state, [both], rows, set()) == [both]


# --- 3. the element grader reads the narrative --------------------------------------------------------------------------------

def test_the_link_prompt_never_asks_for_proof():
    prompt = issue_records._LINK_SYSTEM
    assert "non serve una prova" in prompt and "racconti il fatto come accaduto" in prompt
    assert "per provarlo" not in prompt


# --- 4. §1 and §4 assert only what the record allows ----------------------------------------------------------------------------

def _records():
    return {"1062 c.c.": {"verdict": issue_records.APPLICABLE, "requisites": [_req(1, PROVATO)]},
            "1158 c.c.": {"verdict": DA_PROVARE, "requisites": [_req(1, PROVATO), _req(1, DA_PROVARE)]},
            "1061 c.c.": {"verdict": DA_PROVARE, "requisites": [_req(1, DA_PROVARE)]},
            "1051 c.c.": {"verdict": issue_records.NOT_APPLICABLE, "requisites": [_req(1, NON_RISULTA)]},
            "1079 c.c.": {"verdict": issue_records.CLIENT_ARGUMENT, "requisites": [_req(1, NON_RISULTA)],
                          "relied_by": "Domitilla"}}


def test_only_applicable_or_partly_proven_articles_are_asserted():
    records = _records()
    assert [k for k, r in records.items() if issue_records.assertable(r)] == ["1062 c.c.", "1158 c.c."]
    draft = (f"**1. {LEAD}**\n\nLeonardo agisce ex art. 1062 c.c. e art. 1158 c.c. Può invocare l'art. 1061 c.c. "
             "Il passaggio coattivo (art. 1051 c.c.) è escluso. L'art. 1079 c.c. invocato da Domitilla non regge. "
             "L'art. 2043 c.c. non c'entra. L'art. 1051 c.c. è da valutare.\n\n"
             f"**2. {NORMS}**\n\nL'art. 1061 c.c. resta.\n\n"
             f"**4. {PROC}**\n\n- Usucapione (art. 1158 c.c.): 20 anni.\n- Passaggio (art. 1051 c.c.): indennità.")
    log = []
    out = flow.asserted_only(draft, {"records": records}, log)
    sections = dict(answer_ledger.sections_of(out))
    lead = sections[LEAD]
    assert "Leonardo agisce ex art. 1062 c.c. e art. 1158 c.c. Può" in lead                        # assertable
    assert "Può invocare l'art. 1061 c.c. (da valutare: art. 1061 c.c.)." in lead
    assert "è escluso (da valutare: art. 1051 c.c.)." in lead
    assert "non regge. L'art. 2043 c.c. non c'entra." in lead                  # «argomento del cliente»; no record
    assert lead.count("da valutare") == 3                                     # one already said so: as it was
    assert sections[NORMS].strip() == "L'art. 1061 c.c. resta."                 # §2: not this rule's
    assert sections[PROC].strip() == "- Usucapione (art. 1158 c.c.): 20 anni.\n- Passaggio (art. 1051 c.c.): indennità."
    assert [x["articles"] for x in log] == [["1061 c.c."], ["1051 c.c."]]    # v60t (3): never the procedure's lines


def test_the_writer_is_told_which_norms_it_may_assert(monkeypatch):
    state, seen = _single(monkeypatch)
    state["records"]["1062 c.c."]["requisites"][0]["status"] = PROVATO
    flow.single_shot("Il caso.")
    lead_job = next(j for j in seen["jobs"][-1] if j[0] == LEAD)
    assert "Fonda la sezione solo su queste norme: art. 1062 c.c.;" in lead_job[1]
    assert "«da valutare»" in lead_job[1]


# --- 5. no bare article line under a §2 heading ------------------------------------------------------------------------------

def test_a_line_that_is_only_the_articles_label_leaves_the_block():
    heading = "**Art. 1062 c.c. — Destinazione del padre di famiglia**"
    records = {"1062 c.c.": {"rubrica": "Destinazione del padre di famiglia"}}
    for line in ("art. 1062 c.c.", "[1] Art. 1062 c.c. — Destinazione del padre di famiglia", "**Art. 1062 c.c.**",
                 "[2]", "Art. 1062 del codice civile:"):
        assert flow._bare_article_line(line, heading, records), line
    for line in ("Art. 1062 c.c.: i fondi erano di Mevio.", "La destinazione del padre di famiglia richiede opere.", ""):
        assert not flow._bare_article_line(line, heading, records), line
    draft = (f"**2. {NORMS}**\n\n{heading}\n- testo, requisito «fondi»: provato.\n\nart. 1062 c.c.\n"
             "Mevio possedeva i fondi.\n\n**Art. 1061 c.c.**\n\n[2] Art. 1061 c.c.\n\n**5. Conclusioni**\n\nFine.")
    log = {}
    text, rows = flow.ledger_out(draft, {"records": {"1062 c.c.": {"verdict": PROVATO, "requisites": []}}}, log)
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert norms.strip() == f"{heading}\n\nMevio possedeva i fondi."
    assert log["bare_lines"] == {"1062 c.c.": ["art. 1062 c.c."], "1061 c.c.": ["[2] Art. 1061 c.c."]}
    assert log["bare"] == ["1061 c.c."]


# --- 6. the procedural store free of the corpus' markers ---------------------------------------------------------------------

def test_the_store_facts_lose_the_corpus_markers():
    facts = {"procedibilita": "a querela della persona offesa (3)", "pena": "reclusione da sei mesi a tre anni [196]",
             "ufficio_condizioni": [{"comma": 2, "text": "(abrogato)"}, {"comma": 3, "text": "se ricorre (abrogata) una "
                                                                                          "circostanza aggravante"}],
             "competenza": "tribunale monocratico", "querela": True}
    clean = procedural_facts.clean_facts(facts)
    assert clean["procedibilita"] == "a querela della persona offesa" and clean["pena"] == "reclusione da sei mesi a tre anni"
    assert clean["ufficio_condizioni"] == [{"comma": 3, "text": "se ricorre una circostanza aggravante"}]
    assert clean["querela"] is True and clean["competenza"] == "tribunale monocratico"
    script = (HERE.parent / "scripts" / "fill_procedural_store.py").read_text(encoding="utf-8")
    assert "procedural_facts.clean_facts(procedural_facts.from_row(row))" in script


# --- 7. the single-shot client line ---------------------------------------------------------------------------------------------

def test_a_client_the_facts_name_gets_its_line_with_the_counterpartys_label(monkeypatch):
    state = base._state(monkeypatch)                                          # Leonardo cliente, Domitilla controparte
    state["mode"] = "single_shot"
    assert not any(q["id"] == "parte_assistita" for q in state["questions"])
    client, line = flow.settle_client(state, flow._restored_analysis(state["analysis"]))
    assert client == "" and line == "Il parere è reso nell'interesse di Leonardo; controparte: Domitilla (convenuto)."
    state["mode"] = None
    assert flow.settle_client(state, flow._restored_analysis(state["analysis"])) == ("", "")      # the rounds: none


def test_a_stated_client_a_carico_gets_its_own_label(monkeypatch):
    facts = {**base.FACTS, "parties": [{"name": "Leonardo", "role": "proprietario"},
                                       {"name": "Domitilla", "role": "convenuta, proprietaria"}]}
    state = base._state(monkeypatch, facts)
    state["mode"] = "single_shot"
    state.setdefault("answered", {})["parte_assistita"] = "Domitilla"
    analysis = flow._restored_analysis(state["analysis"])
    analysis["parte_assistita"] = "Domitilla"
    client, line = flow.settle_client(state, analysis)
    assert client == "Domitilla" and line == "Il parere è reso nell'interesse di Domitilla (convenuta)."


def test_the_assumed_client_may_be_the_extractions_controparte(monkeypatch):
    facts = {**base.FACTS, "parties": [{"name": "Leonardo", "role": "proprietario"},
                                       {"name": "Domitilla", "role": "controparte, proprietaria"}]}
    state = base._state(monkeypatch, facts)
    assert any(q["id"] == "parte_assistita" for q in state["questions"])
    state["mode"] = "single_shot"
    client, line = flow.settle_client(state, flow._restored_analysis(state["analysis"]))
    assert client == "Domitilla" and line == ("Il parere è reso nell'interesse di Domitilla (convenuto); se assisti "
                                              "un'altra parte, dimmelo.")


def test_the_line_opens_the_single_shot_parere(monkeypatch):
    state, seen = _single(monkeypatch)
    text = flow.single_shot("Il caso.")["answer"]
    assert text.startswith("Il parere è reso nell'interesse di Leonardo; controparte: Domitilla (convenuto).\n\n**1.")
