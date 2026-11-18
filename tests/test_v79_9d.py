"""v79.9 block D -- allegations, fragments, penal flag, citations, scriminante commi; synthetic sentences only:

D1. an allegazione only on an attribution verb or «secondo <parte>» -- never the conditional, the tense, «presunto» or
    a neighbouring attributed sentence
D2. a modal permission, a bare object, a scope or saving clause dropped at decomposition; the nightly lists a survivor
D3. the penal route on the fraud indicators (a false description, a non-delivery, a vanished counterparty) and on a thing
    lent and not returned
D4. a ruling's reference whole or not at all in every written section; a criminal ruling on a civil record never printed
    with its estremi alone
D5. a scriminante's thesis lists its base comma's elements; a later comma is a thesis of its own only when a fact
    touches it
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
    from src.rag import plan_then_write
    monkeypatch.setattr(plan_then_write, "negation_review", lambda items, known, call=None: [])


def _nightly():
    spec = importlib.util.spec_from_file_location("nightly_sim", ROOT / "scripts" / "nightly_sim.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --- D1 ---------------------------------------------------------------------------------------------------------------------

LATE = "I lavori avrebbero dovuto concludersi entro tre mesi, ma dopo sei mesi risultano incompleti."


@pytest.mark.parametrize("sentence", [LATE, "Tizio avrebbe dovuto pagare entro marzo.", "Il denaro sarebbe stato "
                                      "restituito a maggio.", "Il presunto autore del furto è un vicino.",
                                      "La merce era stata consegnata il 3 marzo."])
def test_a_sentence_with_no_attribution_is_certain_whatever_its_grammar(sentence):
    from src.rag import issue_records
    assert issue_records.certainty(sentence) == "accertato"


@pytest.mark.parametrize("sentence", ["Secondo Caio, i lavori avrebbero dovuto concludersi a marzo.",
                                      "Alfa sostiene che i lavori sarebbero stati conclusi.",
                                      "Secondo la controparte la consegna era avvenuta."])
def test_an_attribution_still_makes_an_allegazione(sentence):
    from src.rag import issue_records
    assert issue_records.certainty(sentence) == "allegato"


def test_a_neighbouring_attributed_sentence_never_makes_one():
    from src.rag import issue_records
    sents = issue_records.sentences("Alfa sostiene che il ritardo dipende dal maltempo. " + LATE)
    assert [s["certainty"] for s in sents] == ["allegato", "accertato"]


def test_the_late_works_sentence_puts_the_appalto_in_corso():
    from src.rag import lawyer_method as M
    mod = next(m for m in M.institution_modules()["modules"] if m["id"] == "appalto")
    assert M.module_stage(mod, {"case": "Tizio ha affidato all'impresa Alfa il rifacimento del tetto. " + LATE}) == \
        "in_corso"


# --- D2 ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, why", [
    ("il compratore può chiedere la risoluzione del contratto", "facoltà, non condizione"),
    ("può domandare la riduzione del prezzo", "facoltà, non condizione"),
    ("il danno", "oggetto senza condizione"), ("la cosa", "oggetto senza condizione"),
    ("riguardo ai quali il venditore ha garantito la qualità", "clausola di ambito"),
    ("nei limiti del valore della cosa", "clausola di ambito"),
    ("salvo che il fatto costituisca più grave reato", "clausola di salvezza"),
    ("la cosa mobile altrui", ""), ("il danno ingiusto", ""),
    ("se il compratore può provare il vizio", ""),
    ("il debitore non esegue esattamente la prestazione dovuta", ""),
])
def test_the_fragments(text, why):
    from src.rag import rule_quote
    assert rule_quote.fragment_why(text) == why


def test_a_fragment_is_dropped_at_decomposition_and_a_survivor_is_listed(monkeypatch):
    from src.rag import interactive_flow as flow, issue_records, rule_quote
    rule = {"how": "civil", "commi": [(1, "x")], "selected": [1],
            "requisites": {1: ["la cosa venduta ha vizi", "il compratore può chiedere la risoluzione del contratto",
                               "il danno", "riguardo ai quali il venditore ha garantito la qualità"]}}
    monkeypatch.setattr(rule_quote, "article_rules", lambda *a, **k: [rule])
    monkeypatch.setattr(flow, "facts_status", lambda text, analysis: "da provare")
    frags = []
    reqs = issue_records._requisites({"_article": ("1492", "c.c."), "s": {"plain_text": "x"}}, {}, frags)
    assert [r["text"] for r in reqs] == ["la cosa venduta ha vizi"]
    assert [f["why"] for f in frags] == ["facoltà, non condizione", "oggetto senza condizione", "clausola di ambito"]
    state = {"records": {"1492 c.c.": {"requisites": [{"kind": "requisito", "text": "il danno"},
                                                      {"kind": "requisito", "text": "la cosa venduta ha vizi"}]}}}
    assert flow.surviving_fragments(state) == ["1492 c.c.: il danno (oggetto senza condizione)"]
    assert flow.grading_counters(state)["fragments_in_tables"] == 1
    text = "\n".join(_nightly().fragment_lines([{"id": "c1", "fragments_surviving": flow.surviving_fragments(state)}]))
    assert "- c1: 1492 c.c.: il danno (oggetto senza condizione)" in text


# --- D3 ---------------------------------------------------------------------------------------------------------------------

def _route(case):
    from src.rag import issue_records, lawyer_method as M
    return M.penal_route(issue_records.sentences(case), {"rito": "civile", "family": "civile", "role": "attore"})


def test_the_online_sale_yields_the_640_line():
    from src.rag import lawyer_method as M
    case = ("Tizio ha comprato online da Caio uno smartphone descritto nell'annuncio come nuovo e sigillato, pagando "
            "300 euro con bonifico. Lo smartphone non è mai arrivato e il profilo di Caio risulta disattivato.")
    assert _route(case) == ("640 c.p.", "truffa")
    line = M.penal_route_line("640 c.p.", "truffa", "procedibile a querela")
    assert "truffa (art. 640 c.p.)" in line["summary"] and "truffa (art. 640 c.p.)" in line["conclusion"]


@pytest.mark.parametrize("case", [
    "Tizio ha pagato a Caio un divano usato; il divano non è mai stato consegnato e Caio è irreperibile.",
    "Alfa ha venduto a Tizio un'auto presentata come mai incidentata; in realtà era incidentata e l'annuncio è stato "
    "cancellato.",
])
def test_two_fraud_indicators_make_the_640_line(case):
    assert _route(case) == ("640 c.p.", "truffa")


def test_one_indicator_alone_is_no_truffa():
    assert _route("Tizio ha pagato a Caio un divano; il divano non è stato consegnato nei tempi.") is None


def test_the_comodato_case_yields_the_646_line():
    case = "Tizio ha prestato la sua bicicletta a Caio per una settimana; Caio non l'ha più restituita."
    assert _route(case) == ("646 c.p.", "appropriazione indebita")
    assert _route("Tizio ha dato in comodato la sua auto a Caio, che si rifiuta di restituirla.")[0] == "646 c.p."


def test_no_penal_route_in_a_criminal_matter():
    from src.rag import issue_records, lawyer_method as M
    case = "Tizio ha prestato la sua bicicletta a Caio; Caio non l'ha più restituita."
    assert M.penal_route(issue_records.sentences(case), {"rito": "penale", "family": "penale"}) is None


# --- D4 ---------------------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("text, out", [
    ("Il nesso causale va provato (Sez. 4, n. 29558/2025).", "Il nesso causale va provato."),
    ("Il nesso causale va provato: 4, n. 29558/2025.", "Il nesso causale va provato."),
    ("Lo afferma Sez. 5, n. 12/2025, Rv. 288001-01, sul punto.", "Lo afferma sul punto."),
    ("Lo afferma Cass. pen., Sez. 4, n. 29558/2025 (01.07.2025).", None),
    ("Si applica il d.lgs. n. 23/2015.", None), ("Lo ha detto Corte cost., n. 242/2019.", None),
])
def test_a_ruling_reference_is_whole_or_not_at_all(text, out):
    from src.rag import lawyer_method as M, rulings
    log = []
    assert M.whole_citations(text, log) == (out if out is not None else text)
    assert bool(log) is (out is not None)
    assert bool(rulings.printed_problems(text)) is (out is not None)


def test_compose_removes_a_headless_reference_from_the_conclusions_and_the_norms():
    from src.rag import lawyer_method as M, posture
    p = {"rito": "civile", "family": "civile", "role": "attore", "stage": "pre_filing",
         "entry": posture.entry_for("civile", "attore", "pre_filing")}
    recs = {"1453 c.c.": {"rubrica": "Risolubilità del contratto per inadempimento", "side": "weapon",
                          "requisites": [{"id": "R1", "comma": 1, "kind": "requisito", "status": "provato",
                                          "text": "l'altro contraente non adempie"}]}}
    ctx = {"records": recs, "circumstances": {}, "posture": p, "case": "Beta non paga Alfa.", "assessment": {},
           "institutions": []}
    ctx["candidates"] = M.thesis_candidates(ctx)
    plan = M.clean_method({"valutazioni": {c["id"]: {"realismo": "alto"} for c in ctx["candidates"]}}, ctx["candidates"])
    log = {}
    out = M.compose([("Norme applicate ai fatti", "Il quadro è quello dell'art. 1453 c.c. e il cliente ha già inviato "
                      "la diffida (Sez. 2, n. 1234/2024).")], ctx, plan, (), log)
    assert "il cliente ha già inviato la diffida." in out and "1234/2024" not in out and "Sez. 2" not in out
    assert log["headless_citations_out"] == ["(Sez. 2, n. 1234/2024)"]


def _ruling(i, archive="snpen", abstract="Effetti della risoluzione"):
    return {"ruling": f"X{i}", "file_name": f"{archive}@s20@a2025@n{i:05d}@tS.clean.pdf", "name": "",
            "document_number": "", "date_enacted": "2025-03-01", "linked": True, "section": f"S{i}", "abstract": abstract}


class Session:
    def __init__(self, rows, texts):
        self.rows, self.text_of = rows, texts

    def run(self, query, **kw):
        from src.rag import rulings
        rows = self.rows if query == rulings.FETCH_QUERY else [
            {"section": i, "plain_text": self.text_of.get(i, "")} for i in kw["ids"]]

        class R:
            def data(self_inner):
                return rows
        return R()


def test_a_criminal_ruling_on_1458_is_never_printed_with_its_estremi_alone():
    from src.rag import rulings
    el = {"id": "R1", "text": "la risoluzione del contratto ha effetto retroattivo tra le parti", "status": "da provare",
          "kind": "requisito"}
    ctx = {"records": {"1458 c.c.": {"side": "weapon", "rubrica": "Effetti della risoluzione", "requisites": [el]}},
           "candidates": [{"id": "T1", "contesta": ["R1"]}], "case": "Alfa chiede la risoluzione."}
    off = "Il reato di appropriazione indebita richiede la interversione del possesso con effetti definitivi."
    civil = "La risoluzione del contratto ha effetto retroattivo e obbliga le parti alle restituzioni."
    rows = [_ruling(1), _ruling(2, archive="snciv")]
    record = rulings.build(ctx, [{"_article": ("1458", "c.c."), "d": {"id": "CC"}, "s": {"name": "1458"}}],
                           Session(rows, {"S1": off, "S2": civil}), call=lambda m, t: {})
    assert [r["id"] for r in record["kept"]] == ["X2"]
    assert not any(r["estremi"].startswith("Cass. pen.") for r in record["kept"])
    assert record["log"]["articles"]["1458 c.c."]["filters"]["penal_on_civil"] >= 1


# --- D5 ---------------------------------------------------------------------------------------------------------------------

def _scriminante(touch=False):
    from src.rag import lawyer_method as M
    reqs = [{"id": "R1", "comma": 1, "kind": "requisito", "status": "provato", "text": "la necessità di difendere un "
             "diritto proprio contro il pericolo attuale di un'offesa ingiusta", "supporting": ["S1"]},
            {"id": "R2", "comma": 1, "kind": "requisito", "status": "da provare",
             "text": "la difesa sia proporzionata all'offesa"},
            {"id": "R3", "comma": 2, "kind": "requisito", "status": "da provare",
             "text": "il fatto avviene nel domicilio altrui", "supporting": ["S2"] if touch else []},
            {"id": "R4", "comma": 3, "kind": "requisito", "status": "da provare",
             "text": "il fatto avviene all'interno di ogni altro luogo ove venga esercitata un'attività commerciale"}]
    recs = {"52 c.p.": {"rubrica": "Difesa legittima", "decomposed_for": "difesa di parte generale", "requisites": reqs},
            "582 c.p.": {"rubrica": "Lesione personale", "side": "threat", "decomposed_for": "x",
                         "requisites": [{"id": "R9", "comma": 1, "kind": "requisito", "status": "provato",
                                         "text": "cagiona ad alcuno una lesione personale", "supporting": ["S1"]}]}}
    return M.method_records(recs)


def _imputato_ctx(records):
    from src.rag import posture
    p = {"rito": "penale", "family": "penale", "role": "imputato", "stage": "indagini",
         "entry": posture.entry_for("penale", "imputato", "indagini")}
    return {"records": records, "circumstances": {}, "posture": p, "case": "Tizio colpisce Caio.",
            "assessment": {"article": "582 c.p."}, "institutions": []}


def test_the_scriminante_lists_its_base_comma_and_an_untouched_comma_is_no_thesis():
    from src.rag import lawyer_method as M
    recs = _scriminante()
    assert [r["id"] for r in recs["52 c.p."]["requisites"]] == ["R1", "R2"]
    assert "52 c.p. co. 2" not in recs and "52 c.p. co. 3" not in recs          # fact-free: v78's noise
    kept = {**recs, "52 c.p. co. 3": {"rubrica": "Difesa legittima (co. 3)", "comma_of": "52 c.p.", "requisites": [
        {"id": "R4", "comma": 3, "kind": "requisito", "status": "da provare", "text": "un luogo di attività commerciale"}]}}
    sc = [c for c in M.thesis_candidates(_imputato_ctx(kept)) if c["kind"] == "scriminante"]
    assert [(c["article"], c["poggia_su"]) for c in sc] == [("52 c.p.", ["R1"])]


def test_a_later_comma_a_fact_touches_is_a_thesis_of_its_own():
    from src.rag import lawyer_method as M
    recs = _scriminante(touch=True)
    recs["52 c.p. co. 2"]["requisites"][0]["status"] = "provato"
    sc = [c for c in M.thesis_candidates(_imputato_ctx(recs)) if c["kind"] == "scriminante"]
    assert [c["article"] for c in sc] == ["52 c.p.", "52 c.p. co. 2"]
    assert sc[1]["poggia_su"] == ["R3"]
