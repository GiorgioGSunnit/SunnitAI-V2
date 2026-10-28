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
