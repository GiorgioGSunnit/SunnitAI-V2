"""CCNL_PATH_ENABLED (src/rag/ccnl_path.py): the contract, the version in force, the clauses rebuilt from their
Sections, the framework, the answer's checks, the hook in src/rag/main.run; and the scoring of tests/eval_ccnl.py.
Invented sectors, documents and texts only."""
import os
import sys
from datetime import date
from pathlib import Path

os.environ.setdefault("EMBEDDING_PROVIDER", "openai")
os.environ.setdefault("EMBEDDING_API_KEY", "test-key")
os.environ.setdefault("LLM_API_KEY", "test-key")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")
os.environ.setdefault("NEO4J_USER", "neo4j")
os.environ.setdefault("NEO4J_PASSWORD", "test-password")

import pytest  # noqa: E402

from src.rag import ccnl_path as cp  # noqa: E402

SECTORS = {
    "Metalmeccanica - Industria": 28, "Metalmeccanica P.I. - Confapi": 26, "Metalmeccanica - Artigianato": 12,
    "Metalmeccanica - Cooperative": 9, "Edilizia - Industria": 25, "Edilizia - Artigianato": 32,
    "Colf - Personale Domestico Convivente": 21, "Colf - Personale Domestico Non Convivente": 21,
    "Studi Professionali (dal 01.07.04)": 18, "Studi Professionali (Cisal - Anpit)": 7,
    "Turismo - Pubblici Esercizi": 15, "Agricoltura - Contoterzisti": 8,
}


# --- cleaning and names ------------------------------------------------------------------------------------

def test_the_watermark_is_stripped():
    text = "- IlCCNL | S.I.A. Srl | Riproduzione Riservata - Livello 1 € 2.089,63\nIl CCNL|SIA Srl|Riproduzione riservata"
    assert cp.clean(text) == "Livello 1 € 2.089,63"


@pytest.mark.parametrize("subtype, kind", [
    ("Testo Consolidato - Metalmeccanica - Industria", cp.TC), ("Accordo di rinnovo", cp.RINNOVO),
    ("Ipotesi di Accordo", cp.IPOTESI), ("CCNL", cp.CCNL), ("CCNL - Vettori", cp.CCNL),
    ("Verbale Integrativo", cp.VERBALE), ("Accordo Previdenza Integrativa", cp.ACCORDO),
    ("Protocollo di Intesa", cp.PROTOCOLLO), ("Regolamento Cassa", cp.OTHER)])
def test_the_kind_is_read_from_the_subtype(subtype, kind):
    assert cp.doc_kind({"document_subtype": subtype}) == kind


def test_the_kind_falls_back_to_the_name_after_the_dash():
    assert cp.doc_kind({"name": "[CCNL] Edilizia - Industria — Accordo di rinnovo del 18072018"}) == cp.RINNOVO
    assert cp.name_date("[CCNL] Edilizia — Accordo di rinnovo del 18072018") == date(2018, 7, 18)


# --- step 1 ---------------------------------------------------------------------------------------------------

def test_a_named_contract_settles_the_variant():
    got = cp.identify_contract("Secondo il CCNL Metalmeccanica Industria quante ferie spettano?", SECTORS, llm=False)
    assert got["sector"] == "Metalmeccanica - Industria"
    assert not got["ambiguous"] and not got["assumption"]


def test_an_inferred_family_without_its_variant_is_an_assumption():
    got = cp.identify_contract("Lavoro come operaio in una fabbrica metalmeccanica: quanto dura la prova?",
                               SECTORS, llm=False)
    assert got["sector"] == "Metalmeccanica - Industria"          # the industry's own, first by preference
    assert got["ambiguous"] and got["assumption"]
    assert "Metalmeccanica P.I. - Confapi" in got["alternatives"]


def test_the_text_settles_artigianato_and_convivente():
    edile = cp.identify_contract("Sono un muratore di un'impresa edile artigiana: qual è la paga?", SECTORS, llm=False)
    assert edile["sector"] == "Edilizia - Artigianato" and not edile["ambiguous"]
    colf = cp.identify_contract("Una badante convivente quante ferie ha?", SECTORS, llm=False)
    assert colf["sector"] == "Colf - Personale Domestico Convivente" and not colf["ambiguous"]
    both = cp.identify_contract("Qual è il preavviso per una colf?", SECTORS, llm=False)
    assert both["ambiguous"]


def test_a_union_qualifier_loses_to_the_plain_name():
    got = cp.identify_contract("Segretaria in uno studio legale: quanti mesi di comporto?", SECTORS, llm=False)
    assert got["sector"] == "Studi Professionali (dal 01.07.04)" and got["ambiguous"]
    named = cp.identify_contract("Nel CCNL Studi Professionali quanto dura la prova?", SECTORS, llm=False)
    assert named["sector"] == "Studi Professionali (dal 01.07.04)" and not named["ambiguous"]


def test_no_match_without_the_llm_is_no_contract():
    assert cp.identify_contract("Quanto dura la prova?", SECTORS, llm=False)["sector"] is None


# --- step 2 ---------------------------------------------------------------------------------------------------

def _doc(id_, subtype, valid_from, valid_until=None, name=None):
    return {"id": id_, "document_subtype": subtype, "valid_from": valid_from, "valid_until": valid_until,
            "name": name or f"[CCNL] Settore — {subtype}"}


DOCS = [
    _doc("tc", "Testo Consolidato - Settore", "2021-02-05", "2028-06-30"),
    _doc("c16", "CCNL", "2016-11-26", "2019-12-31"),
    _doc("r18", "Accordo di rinnovo", "2018-03-01"),
    _doc("r24", "Accordo di rinnovo", "2024-07-01", name="[CCNL] Settore — Accordo di rinnovo del 01072024"),
    _doc("i24", "Ipotesi di Accordo", "2024-07-01", name="[CCNL] Settore — Ipotesi di Accordo del 01072024"),
    _doc("i25", "Ipotesi di Accordo", "2025-05-10"),
    _doc("a27", "Accordo", "2027-01-01"),
]


def test_the_testo_consolidato_covering_the_date_is_the_base_with_later_agreements():
    v = cp.choose_version(DOCS, date(2026, 10, 28))
    assert v["base"]["id"] == "tc" and not v["expired"]
    assert [d["id"] for d in v["amendments"]] == ["r24", "i25"]       # the 2027 accordo is after the date
    assert v["dropped"] and "ipotesi" in v["dropped"][0]["why"]        # i24: a signed accordo has its date


def test_before_the_testo_consolidato_the_latest_ccnl_in_force():
    v = cp.choose_version(DOCS, date(2019, 6, 1))
    assert v["base"]["id"] == "c16"
    assert [d["id"] for d in v["amendments"]] == ["r18"]


def test_an_expired_testo_consolidato_stays_the_base_and_is_marked():
    v = cp.choose_version(DOCS, date(2029, 1, 1))
    assert v["base"]["id"] == "tc" and v["expired"]


def test_before_every_ccnl_the_latest_renewal_in_force():
    docs = [_doc("c21", "CCNL", "2021-02-05"), _doc("r17", "Accordo di rinnovo", "2017-01-01"),
            _doc("r19", "Accordo di rinnovo", "2019-03-01")]
    assert cp.choose_version(docs, date(2019, 12, 31))["base"]["id"] == "r19"


def test_the_relevant_date():
    today = date(2026, 10, 28)
    assert cp.relevant_date("Quante ferie spettano?", today) == (today, "today (no date in the text)")
    assert cp.relevant_date("Licenziato il 3 maggio 2024, assunto nel 2019", today)[0] == date(2024, 5, 3)
    assert cp.relevant_date("Nel 2019 ha ricevuto una contestazione", today)[0] == date(2019, 12, 31)
    assert cp.relevant_date("a marzo 2023", today)[0] == date(2023, 3, 31)
    assert cp.relevant_date("art. 7 l. 300/1970", today)[0] == today
    assert cp.relevant_date("nel 2026", today)[0] == today                 # not past today


# --- step 3 ---------------------------------------------------------------------------------------------------

def test_article_headings_and_titles():
    text = "TITOLO V Art. 16 - Diritti e rapporti sindacali Al fine di regolamentare l'esercizio dei diritti"
    assert cp.article_headings(text) == [(9, "Art. 16", "Diritti e rapporti sindacali")]
    assert cp.article_headings("Articolo 24 -")[0][1:] == ("Art. 24", "")
    assert cp.article_headings("Articolo 24 -\nPeriodo di prova\nIl periodo")[0][2] == "Periodo di prova"
    assert cp.article_headings("ai sensi dell'art. 7 - comma 2") == []
    assert cp.article_headings("Art. 5-bis. Ferie solidali") == [(0, "Art. 5-bis", "Ferie solidali")]


def test_the_order_key_prefers_a_real_position_then_the_id_then_the_element_id():
    rows = [{"id": f"x::{n}", "element_id": f"4:u:{100 + n}", "rel": {}, "props": {}} for n in (3, 1, 2)]
    assert cp.order_key(rows) == "id_suffix"
    assert [r["id"] for r in cp.ordered(rows)[0]] == ["x::1", "x::2", "x::3"]
    with_pos = [{**r, "rel": {"position": 10 - i}} for i, r in enumerate(rows)]
    assert cp.order_key(with_pos) == "rel.position"
    named = [{"id": f"DOCUMENT_SECTION::h::{w}", "element_id": f"4:u:{n}", "rel": {}, "props": {}}
             for w, n in (("ferie", 7), ("prova", 5))]
    assert cp.order_key(named) == "element_id"
    assert cp.order_key(named, forced="s.order") is None


def _rows(texts, doc="tc"):
    return [{"doc": doc, "id": f"s{i}", "name": f"n{i}", "text": t, "element_id": f"4:u:{i}", "rel": {},
             "props": {}, "sim": None} for i, t in enumerate(texts)]


def test_a_clause_is_rebuilt_from_its_heading_to_the_next_article():
    rows = _rows(["Art. 9 - Orario Le ore sono 40.", "Art. 10 - Ferie Il lavoratore ha diritto a",
                  "un periodo di ferie di quattro settimane", "retribuite per ogni anno.", "Art. 11 - Malattia ..."])
    clause = cp.rebuild_clause(rows, 2)
    assert (clause["start"], clause["end"]) == (1, 3)
    assert clause["ref"] == "Art. 10" and clause["title"] == "Ferie"
    assert cp.clause_reference(clause) == "Art. 10 - Ferie"


def test_a_clause_without_an_article_is_cited_by_its_title():
    rows = _rows(["introduzione", "il periodo di prova dura tre mesi", "per gli impiegati"])
    rows[1]["name"] = "Periodo di prova degli impiegati"
    clause = cp.rebuild_clause(rows, 1)
    assert clause["ref"] is None and cp.clause_reference(clause) == "clausola «Periodo di prova degli impiegati»"


def test_select_clauses_takes_base_and_amendment_hits():
    base = _rows(["Art. 10 - Ferie Il lavoratore ha diritto a ferie", "di quattro settimane", "Art. 11 - Altro"])
    amend = _rows(["Le parti modificano le ferie: cinque settimane"], doc="r24")
    chosen = [{"id": "tc", "kind": cp.TC, "name": "TC"}, {"id": "r24", "kind": cp.RINNOVO, "name": "R"}]
    clauses, order = cp.select_clauses(base + amend, chosen, ["ferie", "settimane"])
    assert [c["doc"]["id"] for c in clauses] == ["tc", "r24"]
    assert clauses[0]["ref"] == "Art. 10" and order["TC"] == "element_id"


def test_pay_rows_are_read_not_computed():
    text = ("- IlCCNL | S.I.A. Srl | Riproduzione Riservata - Livello Retribuzione Tabellare al 31/08/2022 "
            "Retribuzione tabellare dal 01/09/2022 1 € 2.089,63 € 46,14 € 2.135,77 2 € 1.959,16 € 43,26 € 2.002,42 "
            "AS € 2.013,13 € 35,37 € 2.048,50. L'aumento è pari a € 59,68.")
    rows = cp.pay_rows(text)
    assert rows["rows"][:3] == [("1", ["€2.089,63", "€46,14", "€2.135,77"]),
                                ("2", ["€1.959,16", "€43,26", "€2.002,42"]),
                                ("AS", ["€2.013,13", "€35,37", "€2.048,50"])]
    assert len(rows["rows"]) == 3                     # "pari a € 59,68" is no row
    assert "01/09/2022" in rows["dates"]


# --- the answer's checks -----------------------------------------------------------------------------------

def test_an_amount_not_in_the_material_goes_with_its_sentence():
    answer = "Il minimo del 1° livello è di € 2.135,77 dal 1/9/2022. Il superminimo è di 150,00 euro. Fine."
    kept, removed = cp.check_amounts(answer, "1 € 2.089,63 € 46,14 € 2.135,77")
    assert kept == "Il minimo del 1° livello è di € 2.135,77 dal 1/9/2022. Fine." and removed == ["150,00"]


def test_copied_sentences_are_found():
    clause = "Il lavoratore ha diritto ad un periodo di ferie retribuito di quattro settimane per ogni anno di servizio"
    answer = ("Il CCNL prevede quattro settimane di ferie. Il lavoratore ha diritto ad un periodo di ferie "
              "retribuito di quattro settimane per ogni anno.")
    assert cp.verbatim_sentences(answer, [clause]) == [answer.split(". ", 1)[1]]
    assert cp.verbatim_sentences("Spettano quattro settimane di ferie.", [clause]) == []


def test_the_header_states_version_and_assumption():
    version = cp.choose_version(DOCS, date(2026, 10, 28))
    text = cp.header({"sector": "Settore", "assumption": True, "alternatives": ["Settore - Artigianato"]},
                     version, date(2026, 10, 28), "today (no date in the text)")
    assert "vigenza dal 05/02/2021 al 30/06/2028" in text and "**Ipotesi**" in text
    assert "Settore - Artigianato" in text and "Integrato da" in text


# --- the session, end to end ---------------------------------------------------------------------------------

class _Result:
    def __init__(self, rows):
        self.rows = rows

    def data(self):
        return self.rows


class _Session:
    def __init__(self, sections):
        self.sections = sections
        self.queries = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def run(self, query, **params):
        self.queries.append(query)
        if "count(*) AS n" in query and "d.sector AS sector" in query:
            return _Result([{"sector": s, "n": n} for s, n in SECTORS.items()])
        if "d.sector = $sector" in query:
            return _Result([{**d, "sector": params["sector"]} for d in DOCS])
        if "UNWIND $ids AS doc_id" in query:
            return _Result([r for r in self.sections if r["doc"] in params["ids"]])
        if "STARTS WITH $prefix" in query:
            return _Result([{"id": "LEGAL_DOC::cc", "name": "Codice Civile"}])
        if "CONTAINS $number" in query:
            return _Result([{"id": "LEGAL_DOC::dlgs66", "name": "Decreto Legislativo 8 aprile 2003, n. 66"}])
        if "UNWIND $items AS it" in query:
            out = []
            for it in params["items"]:
                out.append({"doc": it["doc"], "num": it["num"], "d": {"id": it["doc"], "name": "x",
                                                                      "document_type": "primary"},
                            "s": {"name": f"{it['num']}.0.0", "plain_text": f"Testo dell'articolo {it['num']}."}})
            return _Result(out)
        raise AssertionError(f"unexpected query: {query[:80]}")


class _Driver:
    def __init__(self, session):
        self._session = session

    def session(self, **_):
        return self._session


@pytest.fixture
def path(monkeypatch):
    from src.rag import ai_chat, lookups, source_tiers
    calls = []

    def fake_chat(messages, max_tokens=None, **_):
        calls.append(messages[-1].content)
        return ("**Cosa prevede il CCNL**\nSpettano quattro settimane di ferie (Art. 10 - Ferie). Il superminimo "
                "è di 99,99 euro.\n**Quadro normativo**\nL'art. 2109 c.c. e l'art. 10 d.lgs. 66/2003 fissano il "
                "minimo.\n**Da verificare**\n- il livello.")
    monkeypatch.setattr(ai_chat, "_call_chat", fake_chat)
    monkeypatch.setattr(lookups, "_cached_embed_query", lambda text: (0.1, 0.2))
    monkeypatch.setattr(source_tiers, "support_rulings", lambda *a, **k: [])
    with cp._SECTOR_LOCK:
        cp._SECTOR_CACHE.update(sectors=None, ts=0.0)
    sections = _rows(["Art. 10 - Ferie - IlCCNL | S.I.A. Srl | Riproduzione Riservata - Il lavoratore ha diritto a",
                      "un periodo di ferie di quattro settimane.", "Art. 11 - Malattia"])
    for r in sections:
        r["sim"] = 0.5
    return _Session(sections), calls


def test_answer_end_to_end(path):
    session, calls = path
    out = cp.answer("Secondo il CCNL Metalmeccanica Industria quante ferie spettano?", _Driver(session), "neo4j",
                    today=date(2026, 10, 28))
    assert out["route"] == "ccnl" and out["ccnl_sector"] == "Metalmeccanica - Industria"
    assert out["answer"].startswith("**Contratto applicato**: CCNL Metalmeccanica - Industria")
    assert "99,99" not in out["answer"] and "€" not in out["answer"].split("**Cosa")[0]
    assert "Riproduzione" not in calls[0] and "[CLAUSOLA 1]" in calls[0] and "Art. 10 - Ferie" in calls[0]
    assert "[LEGGE] art. 2109 c.c." in calls[0] and "[LEGGE] art. 10 d.lgs. 66/2003" in calls[0]
    info = out["ccnl"]
    assert info["version"]["base"]["id"] == "tc" and info["clauses"]["found"][0]["ref"] == "Art. 10 - Ferie"
    assert info["answer"]["amounts_removed"] == ["99,99"]
    ccnl_cite = next(c for c in out["citations"] if c["document_type"] == "ccnl")
    assert ccnl_cite["sections"][0]["plain_text"] == ""            # no CCNL text handed out
    assert info["framework"]["found"] == ["2109 c.c.", "10 d.lgs. 66/2003"]


def test_answer_gives_up_without_a_contract(path):
    session, calls = path
    assert cp.answer("Quanto dura la prova?", _Driver(session), "neo4j", llm=False) is None


def test_should_run_and_the_hook(monkeypatch):
    monkeypatch.delenv(cp.FLAG, raising=False)
    assert cp.should_run("CCNL Metalmeccanica: quante ferie?") == (False, "flag off")
    monkeypatch.setenv(cp.FLAG, "true")
    assert cp.should_run("CCNL Metalmeccanica: quante ferie?")[0]
    assert not cp.should_run("CCNL Metalmeccanica: quante ferie?", pending_calculation={"x": 1})[0]
    assert not cp.should_run("Calcola il TFR secondo il CCNL commercio")[0]
    assert cp.should_run("Calcola il TFR secondo il CCNL commercio", skip_calculation=True)[0]
    assert not cp.should_run("Quali sono i requisiti della legittima difesa?")[0]
    assert cp.should_run("Faccio il cameriere in un ristorante: quanto dura il periodo di prova?")[0]
    assert cp.should_run("Qual è il preavviso di licenziamento per una colf?")[0]
    assert not cp.should_run("Quanto dura il periodo di prova nel patto di non concorrenza?")[0]

    from src.rag import main
    monkeypatch.setattr(cp, "answer", lambda text, driver, db: {"answer": "A", "route": "ccnl", "citations": []})
    assert main._ccnl_turn("CCNL Metalmeccanica: quante ferie?", {"query": "q"})["answer"] == "A"

    def boom(*a):
        raise RuntimeError("down")
    monkeypatch.setattr(cp, "answer", boom)
    assert main._ccnl_turn("CCNL Metalmeccanica: quante ferie?", {"query": "q"}) is None
    monkeypatch.setenv(cp.FLAG, "false")
    assert main._ccnl_turn("CCNL Metalmeccanica: quante ferie?", {"query": "q"}) is None


# --- tests/eval_ccnl.py ---------------------------------------------------------------------------------------

sys.path.insert(0, str(Path(__file__).resolve().parent))
import eval_ccnl  # noqa: E402


def test_framework_mentions():
    answer = ("Gli artt. 2118 e 2119 c.c.; l'art. 7 l. 300/1970; l'art. 10 del d.lgs. n. 66/2003; l'art. 18 dello "
              "Statuto dei lavoratori; l'art. 36 Cost.; l'art. 16 del CCNL.")
    assert eval_ccnl.framework_mentions(answer) == {("2118", "c.c."), ("2119", "c.c."), ("7", "l. 300/1970"),
                                                   ("10", "d.lgs. 66/2003"), ("18", "l. 300/1970"), ("36", "Cost.")}


def _version_log(base, amendments, docs, when="2026-10-28"):
    return {"date": when, "base": base, "amendments": amendments, "sector_documents": docs}


def test_version_scoring():
    tc = {"id": "tc", "kind": "testo consolidato", "valid_from": "2021-02-05"}
    r24 = {"id": "r24", "kind": "accordo di rinnovo", "valid_from": "2024-07-01"}
    c16 = {"id": "c16", "kind": "ccnl", "valid_from": "2016-11-26"}
    docs = [tc, r24, c16]
    assert eval_ccnl._version_right("testo consolidato", _version_log(tc, [r24], docs)) is True
    assert eval_ccnl._version_right("testo consolidato", _version_log(tc, [], docs)) is False   # renewal missed
    assert eval_ccnl._version_right("latest renewal", _version_log(c16, [], docs, "2019-12-31")) is True
    assert eval_ccnl._version_right("latest renewal", _version_log(tc, [], docs, "2019-12-31")) is False
    assert eval_ccnl._version_right("testo consolidato", {}) is None


def test_score_case():
    case = {"sector": "^metalmeccanica industria$", "version": "testo consolidato", "clause": "ferie",
            "framework": [{"article": "2109", "law": "c.c."}], "ambiguous": False}
    tc = {"id": "tc", "kind": "testo consolidato", "valid_from": "2021-02-05"}
    result = {"answer": "**Contratto applicato**: ... art. 2109 c.c.",
              "ccnl": {"contract": {"sector": "Metalmeccanica - Industria"},
                       "version": _version_log(tc, [], [tc]),
                       "clauses": {"found": [{"ref": "Art. 10 - Ferie", "text": "..."}]}}}
    s = eval_ccnl.score_case(case, result)
    assert (s["sector_right"], s["version_right"], s["clause_found"], s["framework_cited"], s["ambiguity_right"]) \
        == (True, True, True, 1, True)
    assert eval_ccnl.score_case({**case, "ambiguous": True}, result)["ambiguity_right"] is False


# --- second round: headings, wider window, pay tables in force, case law, framework --------------------------

def test_the_heading_names_the_topic():
    phrases = cp.topic_headings(["prova"])
    assert cp.heading_score({"text": "Art. 23 - Periodo di prova L'assunzione", "name": "x"}, phrases) == 4
    assert cp.heading_score({"text": "La durata è di tre mesi", "name": "di prova"}, phrases) == 4    # the tail
    assert cp.heading_score({"text": "Art. 50 - Periodo di prova - periodo di comporto e preavviso Il",
                             "name": "x"}, phrases) == 3
    assert cp.heading_score({"text": "Periodo di prova: tre mesi", "name": "x"}, phrases) == 1
    # a generic tail is no heading: "del rapporto" of "risoluzione del rapporto"
    assert cp.heading_score({"text": "Il rapporto si sospende", "name": "del rapporto"},
                            cp.topic_headings(["preavviso"])) == 0
    # the apprentices' own clause, for a question that is not about them
    apprentice = {"text": "Art. 7 - Periodo di prova Per l'assunzione in prova dell'apprendista è richiesto", "name": "x"}
    assert cp.heading_score(apprentice, phrases, "Quanto dura la prova per un operaio?") == 1
    assert cp.heading_score(apprentice, phrases, "Quanto dura la prova di un apprendista?") == 4
    assert cp.heading_score({"text": "", "name": "di prova"}, phrases) == 0
    assert cp.heading_score({"text": "Il lavoratore durante il periodo di prova", "name": "x"}, phrases) == 0
    toc = "Art. 21 - Assunzione Art. 22 - Documenti Art. 23 - Periodo di prova Art. 24 - Orario"
    assert cp.heading_score({"text": toc, "name": "Indice"}, phrases) == 0


def test_a_heading_hit_wins_over_a_wordy_section():
    rows = _rows(["Art. 23 - Periodo di prova", "La durata è di sei mesi per i quadri, tre per gli altri.",
                  "Art. 24 - Orario", "Durante il periodo di prova, il patto di prova e la prova " * 3])
    chosen = [{"id": "tc", "kind": cp.TC, "name": "TC"}]
    clauses, _ = cp.select_clauses(rows, chosen, cp.topic_keywords("", ["prova"]),
                                   headings=cp.topic_headings(["prova"]))
    assert clauses[0]["ref"] == "Art. 23" and clauses[0]["title"] == "Periodo di prova"
    assert "sei mesi" in clauses[0]["text"]


def test_a_heading_only_hit_takes_more_sections():
    rows = _rows(["Art. 23 - Periodo di prova", "a) quadri: sei mesi;", "b) impiegati: tre mesi;",
                  "c) operai: un mese;", "d) apprendisti: un mese.", "Art. 24 - Orario"])
    clause = cp.rebuild_clause(rows, 0)
    assert (clause["start"], clause["end"]) == (0, 4)
    wordy = _rows(["Art. 23 - Periodo di prova " + "testo della clausola " * 20, "a", "b", "c", "d"])
    assert cp.rebuild_clause(wordy, 0)["end"] == 2


def _pay_doc(id_, kind, frm):
    return {"id": id_, "kind": kind, "from": date.fromisoformat(frm), "until": None,
            "name": f"[CCNL] Settore — {kind} {id_}"}


TABLE_2022 = "Minimi tabellari dal 01/09/2022 1 € 2.000,00 2 € 1.900,00 3 € 1.800,00"
TABLE_2024 = "Minimi tabellari dal 01/06/2024 1 € 2.100,00 2 € 2.000,00 3 € 1.900,00"
TABLE_2027 = "Minimi tabellari dal 01/06/2027 1 € 2.300,00 2 € 2.200,00 3 € 2.100,00"


def test_the_pay_table_in_force_is_the_latest_decorrenza_across_all_documents():
    tc, r24 = _pay_doc("tc", cp.TC, "2021-02-05"), _pay_doc("r24", cp.RINNOVO, "2024-05-01")
    by_doc = {"tc": _rows(["Art. 30 - Minimi", TABLE_2022]),
              "r24": _rows(["Le parti concordano", TABLE_2024 + " " + TABLE_2027], doc="r24")}
    pay = cp.pay_tables(by_doc, [tc, r24], date(2026, 10, 28))
    assert pay["source"]["id"] == "r24" and pay["decorrenza"] == date(2024, 6, 1)
    assert pay["superseded_by"] is None and pay["tables"][0]["rows"]["rows"][0][0] == "1"
    old = cp.pay_tables(by_doc, [tc, r24], date(2023, 1, 1))
    assert old["source"]["id"] == "tc" and old["decorrenza"] == date(2022, 9, 1)
    assert "dalla tabella di" in cp.pay_line(pay) and "01/06/2024" in cp.pay_line(pay)


def test_an_unreadable_later_table_is_named_as_possibly_superseding():
    tc, r25 = _pay_doc("tc", cp.TC, "2021-02-05"), _pay_doc("r25", cp.RINNOVO, "2025-03-01")
    by_doc = {"tc": _rows([TABLE_2022]),
              "r25": _rows(["Minimi tabellari dal 01/03/2025: importi 2.150,00 / 2.050,00 / 1.950,00 per livello"],
                           doc="r25")}
    pay = cp.pay_tables(by_doc, [tc, r25], date(2026, 10, 28))
    assert pay["source"]["id"] == "tc" and pay["superseded_by"]["doc"]["id"] == "r25"
    line = cp.pay_line(pay)
    assert "Potrebbe essere stata superata da" in line and "01/03/2025" in line
    none = cp.pay_tables({"tc": _rows(["Art. 1 - Oggetto"])}, [tc], date(2026, 10, 28))
    assert none["source"] is None and "non indico importi" in cp.pay_line(none)


def test_an_empty_case_law_section_is_omitted():
    body = ("**Quadro normativo**\nL'art. 2109 c.c.\n**Giurisprudenza**\nLe pronunce fornite non sono pertinenti "
            "alla domanda.\n**Risposta pratica**\nQuattro settimane.")
    out, done = cp.drop_empty_caselaw(body)
    assert done == "section" and "Giurisprudenza" not in out and "**Risposta pratica**" in out
    mixed = ("**Giurisprudenza**\nCass. civ., sez. lav., n. 123/2020 conferma il principio. Le altre non sono "
             "pertinenti.\n**Risposta pratica**\nX.")
    out, done = cp.drop_empty_caselaw(mixed)
    assert done == "sentences" and "n. 123/2020" in out and "non sono pertinenti" not in out
    assert cp.drop_empty_caselaw("**Giurisprudenza**\nCass. n. 1/2020 afferma X.\n")[1] == ""
    assert "Giurisprudenza" not in cp.system_prompt(pay=False, rulings=False)
    assert "TABELLE IMPORTI" in cp.system_prompt(pay=True, rulings=True)


def test_a_law_found_by_its_title(monkeypatch):
    from src.rag import article_text

    class S:
        def run(self, query, **params):
            if "CONTAINS $number" in query:
                return _Result([])
            if "any(a IN $aliases" in query:
                assert params["aliases"] == ["statuto dei lavoratori"]
                return _Result([{"id": "LEGAL_DOC::st", "name": "Statuto dei Lavoratori"}])
            raise AssertionError(query)
    monkeypatch.setattr(article_text, "fetch_article",
                        lambda session, doc, n: [{"d": {"id": doc, "name": "Statuto dei Lavoratori"},
                                                  "s": {"name": f"{n}.0.0", "plain_text": "Sanzioni disciplinari"}}])
    rows, missing = cp.fetch_framework(S(), [("7", "l. 300/1970")])
    assert rows[0]["_article"] == ("7", "l. 300/1970") and missing == []


def test_framework_mentions_with_the_law_dated():
    answer = "l'art. 7 della legge 20 maggio 1970, n. 300 e l'art. 10 del d.lgs. 8 aprile 2003, n. 66"
    assert eval_ccnl.framework_mentions(answer) == {("7", "l. 300/1970"), ("10", "d.lgs. 66/2003")}


def test_an_anchored_clause_runs_from_its_heading_to_the_next():
    rows = _rows(["Art. 9 - Orario Le ore sono 40. Art. 10 - Ferie Il lavoratore ha diritto a quattro settimane. "
                  "Art. 11 - Malattia La malattia ...", "altro"])
    match = cp.heading_match(rows[0], cp.topic_headings(["ferie"]))
    assert match["via"] == "heading" and match["score"] == 4
    clause = cp.rebuild_clause(rows, 0, anchor=match["anchor"])
    assert clause["ref"] == "Art. 10" and clause["title"] == "Ferie"
    assert clause["text"].startswith("Art. 10 - Ferie") and "Malattia" not in clause["text"]


def test_a_name_headed_clause_starts_at_its_section():
    rows = _rows(["Art. 38 - Variazione dei minimi Le retribuzioni sono variate ogni anno.",
                  "1. Il rapporto può essere risolto con i seguenti termini di preavviso: 15 giorni.",
                  "2. In caso di licenziamento", "Art. 40 - Altro"])
    rows[1]["name"] = "del rapporto di lavoro e preavviso"
    clause = cp.rebuild_clause(rows, 1, name_headed=True)
    assert clause["start"] == 1 and clause["ref"] is None              # not Art. 38
    assert clause["title"] == "Del rapporto di lavoro e preavviso" and "2. In caso" in clause["text"]
    rows[0]["text"] += " Art. 39 - Risoluzione"
    clause = cp.rebuild_clause(rows, 1, name_headed=True)
    assert (clause["ref"], clause["title"]) == ("Art. 39", "Risoluzione del rapporto di lavoro e preavviso")


TOURISM = {"Turismo Confcommercio - Pubblici esercizi": 19, "Turismo Confcommercio - Pubblici esercizi minori": 19,
           "Turismo Confesercenti - Pubblici esercizi": 8, "Turismo Confindustria - Pubblici Esercizi": 4,
           "Turismo Confcommercio - Aziende alberghiere": 10, "Turismo Confcommercio - Aziende alberghiere minori": 10,
           "Dirigenti - Aziende Alberghiere": 9, "Pubblici Esercizi - Personale Artistico": 1,
           "Turismo, Pubblici esercizi (Confsal - Cifa Fedarcom)": 8, "Commercio - Confcommercio": 20,
           "Commercio - Confesercenti": 11, "Terziario (FID - FederPartiteIva)": 3, "Dirigenti - Commercio": 12,
           "Distribuzione moderna organizzata - Federdistribuzione": 8, "Poste S.P.A. - Ente Poste Italiane": 56,
           "Agenti e Rappresentanti - Commercio": 4, "Terziario (Sinalp - Aniac)": 1}


def test_the_eval_key_cases_of_commercio_and_turismo():
    c1 = cp.identify_contract("Nel CCNL Terziario Distribuzione e Servizi Confcommercio, qual è il preavviso di "
                              "dimissioni per un impiegato di 4° livello?", TOURISM, llm=False)
    assert c1["sector"] == "Commercio - Confcommercio" and not c1["ambiguous"]
    c2 = cp.identify_contract("Sono commessa in un supermercato: per quanti giorni di malattia mi viene conservato "
                              "il posto (periodo di comporto)?", TOURISM, llm=False)
    assert c2["sector"] == "Commercio - Confcommercio" and c2["ambiguous"]          # not Poste ("posto")
    assert c2["alternatives"][:2] == ["Commercio - Confesercenti",
                                      "Distribuzione moderna organizzata - Federdistribuzione"]
    t1 = cp.identify_contract("Faccio il cameriere in un ristorante: quanto dura il periodo di prova?", TOURISM,
                              llm=False)
    assert t1["sector"] == "Turismo Confcommercio - Pubblici esercizi" and t1["ambiguous"]
    t2 = cp.identify_contract("CCNL Turismo Confcommercio: quante settimane di ferie spettano ai dipendenti di un "
                              "albergo?", TOURISM, llm=False)
    assert t2["sector"] == "Turismo Confcommercio - Aziende alberghiere"
    assert t2["alternatives"] == ["Turismo Confcommercio - Aziende alberghiere minori"]
    same = [{"document_subtype": "Testo Consolidato", "valid_from": "2010-01-01", "valid_until": "2027-12-31"}]
    settled = cp.settle_by_documents(t2, same, lambda sector: same)
    assert not settled["ambiguous"] and not settled["assumption"] and settled["same_documents"]
    other = [{"document_subtype": "CCNL", "valid_from": "2024-06-01", "valid_until": "2027-12-31"}]
    kept = cp.settle_by_documents(t1, same, lambda sector: other)
    assert kept["ambiguous"] and kept["alternatives"]


def test_pay_tables_need_pay_words_and_skip_contribution_agreements():
    tc = _pay_doc("tc", cp.TC, "2021-01-01")
    fund = {**_pay_doc("prev", cp.ACCORDO, "2025-07-04"), "document_subtype": "Accordo Previdenza Integrativa"}
    by_doc = {"tc": _rows(["Allegato A: Tabella dei minimi di paga base mensile dal 01/03/2024",
                           "Operaio specializzato 1.800,00 Operaio qualificato 1.650,00 Operaio comune 1.480,00"]),
              "prev": _rows(["TABELLA A contributi", "5 12,00 4 10,00 3 9,00 1 8,00"], doc="prev")}
    pay = cp.pay_tables(by_doc, [tc, fund], date(2026, 10, 28))
    assert pay["source"]["id"] == "tc" and pay["decorrenza"] == date(2024, 3, 1)
    assert [r[0] for r in pay["tables"][0]["rows"]["rows"]] == ["Operaio specializzato", "Operaio qualificato",
                                                                "Operaio comune"]
    quota = cp.pay_tables({"tc": _rows(["Quote di adesione al fondo dal 01/01/2025 1 € 10,00 2 € 12,00 3 € 14,00"])},
                          [tc], date(2026, 10, 28))
    assert quota["source"] is None and quota["candidates"] == 0


def test_the_later_full_text_for_a_renewal_base():
    docs = [_doc("r19", "Accordo di rinnovo", "2019-05-30"), _doc("tc", "Testo Consolidato - X", "2021-02-05"),
            _doc("c21", "CCNL", "2021-02-05"), _doc("c16", "CCNL", "2016-11-26", name="x")]
    assert cp.later_full_text(docs, date(2019, 12, 31))["id"] == "c21"      # the CCNL before the consolidated
    assert cp.later_full_text(docs, date(2026, 1, 1)) is None


def test_a_law_missing_from_the_corpus_is_cited_by_reference(path):
    session, calls = path
    real = session.run

    def run(query, **params):
        if "CONTAINS $number" in query or "any(a IN $aliases" in query:
            return _Result([])
        return real(query, **params)
    session.run = run
    out = cp.answer("Secondo il CCNL Metalmeccanica Industria quante ferie spettano?", _Driver(session), "neo4j",
                    today=date(2026, 10, 28))
    assert "[LEGGE - testo non disponibile] art. 10 d.lgs. 66/2003 - Ferie annuali" in calls[0]
    assert out["ccnl"]["framework"]["by_reference"] == ["10 d.lgs. 66/2003"]


def test_twin_alternatives_are_listed_once():
    contract = {"sector": "A", "how": "concept", "ambiguous": True, "assumption": True,
                "alternatives": ["B", "B minori", "C"]}
    mine = [{"document_subtype": "CCNL", "valid_from": "2020-01-01"}]
    other = [{"document_subtype": "CCNL", "valid_from": "2024-06-01"}]
    docs = {"B": other, "B minori": other, "C": mine}
    out = cp.settle_by_documents(contract, mine, docs.get)
    assert out["alternatives"] == ["B"] and out["same_documents"] == ["B minori", "C"] and out["ambiguous"]
