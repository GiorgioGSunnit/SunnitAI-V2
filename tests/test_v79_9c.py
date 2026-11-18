"""v79.9 block C -- the gate's side doors and the corpus noise, on synthetic sentences and rows only:

C1. the document-level allowlist per matter family on every candidate source: the codes, the defaults (strada,
    stupefacenti, mediazione, consumo with a professional counterparty), an active module's laws; a banking, finance,
    antiriciclaggio, data-protection, insurance, DORA, Banca d'Italia, schools, nautica or public-contracts document
    only with its family's signal or the user naming the law
C2. the counter candidates_out_of_family (0); the nightly lists every document admitted outside the codes and why
C3. a civil contract case never grades a banking circular's, the GDPR's, the insurance code's or the schools TU's
    article; a case naming «banca» or «trattamento dei dati» does
"""

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def _env(monkeypatch, tmp_path):
    for name, value in (("LLM_API_KEY", "k"), ("OPENAI_API_KEY", "k"), ("EMBEDDING_PROVIDER", "openai"),
                        ("EMBEDDING_API_KEY", "k"), ("NEO4J_URI", "bolt://localhost:7687"), ("NEO4J_USER", "n"),
                        ("NEO4J_PASSWORD", "x")):
        monkeypatch.setenv(name, value)
    monkeypatch.setenv("THESIS_LOG_DIR", str(tmp_path / "logs"))


CONTRACT = ("Alfa s.r.l. ha venduto a Beta s.r.l. un macchinario industriale per 50.000 euro; Beta non ha pagato il "
            "saldo alla scadenza.")


def _row(number, code, doc=""):
    return {"_article": (number, code), "d": {"name": doc or code}}


NOISE = [_row("1", "Circolare n. 285 della Banca d'Italia", "Circolare n. 285 della Banca d'Italia del 17 dicembre 2013"),
         _row("6", "Regolamento (UE) 2016/679"), _row("1", "Codice delle assicurazioni private",
                                                     "Decreto legislativo 7 settembre 2005, n. 209"),
         _row("1", "D.lgs. 297/1994", "Decreto legislativo 16 aprile 1994, n. 297"),
         _row("117", "D.lgs. 385/1993", "Decreto legislativo 1 settembre 1993, n. 385"),
         _row("4", "Regolamento (UE) 2022/2554"), _row("21", "D.lgs. 231/2007", "Decreto legislativo 21 novembre 2007, n. 231"),
         _row("1", "Codice della nautica da diporto", "Decreto legislativo 18 luglio 2005, n. 171"),
         _row("1", "Codice dei contratti pubblici", "Decreto legislativo 31 marzo 2023, n. 36"),
         _row("3", "Legge 4 maggio 1983, n. 184")]
MEDIAZIONE_DOC = "Mediazione civile (Decreto Legislativo n. 28/2010) Edizione 2025"
CODES = [_row("1453", "c.c."), _row("1218", "c.c."), _row("640", "c.p."), _row("163", "c.p.c."),
         _row("5", "D.lgs. 28/2010", MEDIAZIONE_DOC)]          # v79.10 (A4): the archive's name


def test_a_civil_contract_case_admits_the_codes_and_mediazione_only():
    from src.rag import family_gate
    ctx = family_gate.context(CONTRACT)
    kept, dropped, outside = family_gate.gate(CODES + NOISE, ctx, set())
    assert [r["_article"] for r in kept] == [r["_article"] for r in CODES]
    assert {x["family"] for x in dropped} == {"Banca d'Italia", "protezione dei dati", "assicurativo", "istruzione",
                                              "bancario", "DORA", "antiriciclaggio", "nautica", "appalti pubblici",
                                              family_gate.OTHER}
    assert outside == [{"doc": MEDIAZIONE_DOC, "family": "mediazione", "by": "default"}]


@pytest.mark.parametrize("case, doc, family", [
    ("Tizio ha un conto corrente presso la banca Alfa, che gli ha addebitato commissioni non pattuite.",
     _row("117", "D.lgs. 385/1993", "Decreto legislativo 1 settembre 1993, n. 385"), "bancario"),
    ("Il datore di lavoro ha pubblicato in bacheca i dati sanitari di Laura: un trattamento dei dati senza consenso.",
     _row("6", "Regolamento (UE) 2016/679"), "protezione dei dati"),
    ("La compagnia di assicurazione rifiuta l'indennizzo del sinistro previsto dalla polizza.",
     _row("1", "Codice delle assicurazioni private", "Decreto legislativo 7 settembre 2005, n. 209"), "assicurativo"),
    ("Il docente impugna la sanzione disciplinare del dirigente scolastico.",
     _row("1", "D.lgs. 297/1994", "Decreto legislativo 16 aprile 1994, n. 297"), "istruzione"),
])
def test_a_family_signal_admits_its_documents(case, doc, family):
    from src.rag import family_gate
    kept, dropped, outside = family_gate.gate([doc], family_gate.context(case), set())
    assert kept == [doc] and not dropped
    assert outside[0]["family"] == family and outside[0]["by"].startswith("segnale «")


def test_the_user_naming_the_law_admits_it():
    from src.rag import family_gate
    case = CONTRACT + " Si applica la legge 184/1983?"
    kept, _, outside = family_gate.gate([_row("3", "Legge 4 maggio 1983, n. 184")], family_gate.context(case), set())
    assert kept and outside[0]["by"] == "nominata dall'utente"
    kept, _, _ = family_gate.gate([_row("3", "Legge 4 maggio 1983, n. 184")], family_gate.context(CONTRACT),
                                  {("3", "Legge 4 maggio 1983, n. 184")})
    assert kept                                              # an article the user names, by its key


def test_an_active_modules_laws_and_the_consumer_code_with_a_professional():
    from src.rag import family_gate
    dismissal = "Laura, dipendente della società Alfa, è stata licenziata senza motivazione."
    row = _row("1", "Riforma del Lavoro Fornero 2026")                  # v79.10: by its archive name only
    assert family_gate.gate([row], family_gate.context(dismissal), set())[0] == [row]
    assert family_gate.gate([row], family_gate.context(CONTRACT), set())[0] == []
    numbered = _row("18", "Legge 20 maggio 1970, n. 300")                 # L. 300/1970: unmapped, never by number
    assert family_gate.gate([numbered], family_gate.context(dismissal), set())[0] == []
    consumer = _row("128", "Codice del Consumo 2026")
    shop = "Il consumatore Tizio ha comprato una lavatrice nel negozio Alfa s.r.l.; si è guastata dopo un mese."
    private = "Tizio ha comprato una lavatrice usata da un venditore privato; si è guastata dopo un mese."
    assert family_gate.gate([consumer], family_gate.context(shop), set())[0] == [consumer]
    assert family_gate.gate([consumer], family_gate.context(private), set())[0] == []


@pytest.mark.parametrize("code, family", [("c.c.", "codice"), ("c.p.p.", "codice"), ("Costituzione", "codice"),
                                          ("c.d.s.", "strada"), ("D.P.R. 309/1990", "stupefacenti"),
                                          ("Regolamento (UE) 2016/679", "protezione dei dati"),
                                          ("Riforma del Lavoro Fornero 2026", "modulo"),
                                          ("Legge 15 luglio 1966, n. 604", "altra legge speciale")])
def test_classify(code, family):
    from src.rag import family_gate
    assert family_gate.classify(code)[0] == family


def _req(i, text, status="provato"):
    return {"id": f"R{i}", "comma": 1, "kind": "requisito", "text": text, "status": status}


def test_no_candidate_from_outside_the_family_and_the_counter():
    from src.rag import lawyer_method as M, posture
    p = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing",
         "entry": posture.entry_for("civile", "attore", "pre_filing")}
    recs = {"1453 c.c.": {"rubrica": "Risolubilità del contratto per inadempimento", "side": "weapon",
                          "requisites": [_req(1, "l'altro contraente non adempie")]},
            "117 D.lgs. 385/1993": {"rubrica": "Contratti", "side": "weapon", "ranked_candidate": True,
                                    "requisites": [_req(2, "i contratti sono redatti per iscritto")]}}
    ctx = {"records": recs, "circumstances": {}, "posture": p, "case": CONTRACT, "assessment": {}, "institutions": []}
    ctx["candidates"] = M.thesis_candidates(ctx)
    assert not any("385/1993" in str(c.get("article")) for c in ctx["candidates"])
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "medio"} for c in ctx["candidates"]}}, ctx["candidates"])
    counts = M.selection_counts(ctx, plan, {})
    assert counts["candidates_out_of_family"] == 0


def test_the_counters_and_the_nightly_list():
    from src.rag import interactive_flow as flow
    state = {"family_gate": {"dropped": [{"article": "6 Regolamento (UE) 2016/679", "family": "protezione dei dati",
                                          "source": "retrieval"}],
                             "admitted": [{"doc": "Decreto legislativo 4 marzo 2010, n. 28", "family": "mediazione",
                                           "by": "default"}]}}
    c = flow.case_counters(state)
    assert c["family_gate_dropped"] == 1 and c["family_admitted_outside_codes"] == 1
    assert c["candidates_out_of_family"] == 0
    spec = importlib.util.spec_from_file_location("nightly_sim", ROOT / "scripts" / "nightly_sim.py")
    nightly = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(nightly)
    text = "\n".join(nightly.family_lines([{"id": "c1", **state}]))
    assert "| c1 | Decreto legislativo 4 marzo 2010, n. 28 | mediazione | default |" in text
    assert "c1: 6 Regolamento (UE) 2016/679 (protezione dei dati, retrieval)" in text
