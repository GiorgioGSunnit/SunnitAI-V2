"""v69: six general rules from the review -- synthetic texts, no case of the sim.

1. two or more question sentences, no list: one quesito block each, in order
2. the labour rito only with an employment relationship («lavori», «impresa edile» never); c.p.c. 409-447 gated by it
3. the norms section's cap is on blocks: at most seven, the strongest first; the others rows of the citations
4. the plain cenni: inside the area its subject names in the institutions index (code + Titolo), the nota dottrinale
   from the same code family, never closing with a question
5. a calculation only with a date to count from; otherwise a quesito, answered with the store's term and its article
   (166 c.p.c.: «almeno settanta giorni prima dell'udienza», a form the store now reads)
6. the rubrica rule on «istituto (art. N)» too -- the record's rubrica when no row holds the article, the requested
   sections too
"""

import inspect
import json
import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import (answer_ledger, calculation, intake, issue_records, plain_grounding, procedural_facts as pf,  # noqa: E402
                     record_check, special_laws)
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

PROVATO, DA_PROVARE, NON_RISULTA = flow.PROVATO, flow.DA_PROVARE, flow.NON_RISULTA
LEAD, NORMS, OBJECTIONS, PROCEDURE, CLOSING = flow.ANSWER_TITLES
TEXTS = json.loads((HERE / "data" / "store_articles.json").read_text(encoding="utf-8"))["articles"]


def _rec(rubrica, *statuses, **extra):
    reqs = [{"kind": "requisito", "comma": 1, "text": f"elemento {i}", "status": st} for i, st in enumerate(statuses, 1)]
    rec = {"rubrica": rubrica, "requisites": reqs, **extra}
    rec["verdict"] = issue_records.verdict(rec)
    return rec


# --- 1. question sentences without a list ---------------------------------------------------------------------------------

def test_two_or_more_question_sentences_are_one_quesito_block_each_in_order():
    text = ("Tizio ha affidato all'impresa Alfa la ristrutturazione del bagno. Le opere presentano vizi gravi e Alfa "
            "pretende il saldo. Tizio può rifiutare il pagamento? Entro quale termine deve denunciare i vizi? Può "
            "chiedere la riduzione del prezzo?")
    blocks = intake.requested_blocks(text)
    assert [(b["kind"], b["subject"]) for b in blocks] == [
        ("quesito", "Tizio può rifiutare il pagamento?"), ("quesito", "Entro quale termine deve denunciare i vizi?"),
        ("quesito", "Può chiedere la riduzione del prezzo?")]
    assert intake.requested_blocks("Il cane di Tizio ha morso Caio nel giardino. Chi risponde del danno?") == []
    listed = "Tizio ha venduto un'auto a Caio. Si chiede: 1) chi paga i vizi; 2) entro quando agire?"
    assert [b["subject"] for b in intake.requested_blocks(listed)] == ["chi paga i vizi", "entro quando agire?"]


# --- 2. the labour rito, and its articles -----------------------------------------------------------------------------------

@pytest.mark.parametrize("text, labour", [
    ("Tizio ha affidato all'impresa edile Alfa i lavori di ristrutturazione con un contratto d'appalto. I lavori sono "
     "viziati e Alfa chiede il pagamento delle fatture.", False),
    ("Il condominio ha deliberato lavori straordinari e il condomino Tizio non paga la sua quota.", False),
    ("Laura, lavoratrice della società Alfa, è stata licenziata senza contestazione scritta.", True),
    ("Il datore di lavoro non versa la retribuzione a Marco da tre mesi.", True),
])
def test_the_labour_rito_needs_an_employment_relationship(text, labour):
    assert (intake.rito_rule(text)[0] == "lavoro") is labour


def test_the_models_labour_rito_needs_one_too(monkeypatch):
    monkeypatch.setattr(intake, "ask_model", lambda *a, **k: {"rito": "lavoro", "role": "attore", "stage": "pre_filing",
                                                              "deliverable": "parere"})
    appalto = intake.detect("Tizio ha affidato all'impresa edile Alfa i lavori di ristrutturazione. I lavori sono "
                            "viziati. Si chiede quali azioni.", model=True)
    assert appalto["rito"] != "lavoro"
    sacked = intake.detect("Laura è stata licenziata dal suo datore di lavoro dopo dodici anni. Si chiede quali azioni.",
                           model=True)
    assert sacked["rito"] == "lavoro"


def test_the_labour_articles_need_the_labour_rito():
    rows = [{"_article": (n, "c.p.c."), "s": {}} for n in ("409", "414", "420-bis", "447", "448", "163")]
    rows.append({"_article": ("1667", "c.c."), "s": {}})
    kept, dropped = special_laws.labour_gate(rows, False, set())
    assert dropped == ["409 c.p.c.", "414 c.p.c.", "420-bis c.p.c.", "447 c.p.c."]
    assert [flow._label(flow._key(r["_article"])) for r in kept] == ["448 c.p.c.", "163 c.p.c.", "1667 c.c."]
    kept, dropped = special_laws.labour_gate(rows, False, {flow._key(("414", "c.p.c."))})     # the user's own: kept
    assert "414 c.p.c." not in dropped
    assert special_laws.labour_gate(rows, True, set())[1] == []


# --- 3. the cap on blocks ----------------------------------------------------------------------------------------------------

def _eleven():
    records = {f"{n} c.c.": _rec(f"Vendita {n}", NON_RISULTA, DA_PROVARE) for n in range(1470, 1478)}   # 0/N
    records["1453 c.c."] = _rec("Risoluzione", PROVATO, PROVATO)                                       # applicabile
    records["1490 c.c."] = _rec("Garanzia per vizi", PROVATO, DA_PROVARE)                              # one proven
    records["1385 c.c."] = {**_rec("Caparra", DA_PROVARE), "named_due": True}                          # named
    return records


def test_the_body_holds_seven_blocks_the_strongest_first():
    records = _eleven()
    ranked = sorted(records, key=lambda k: flow.block_rank(k, records[k], "1490 c.c."))
    assert ranked[:3] == ["1490 c.c.", "1453 c.c.", "1385 c.c."]          # main anchor, applicabile, named
    blocks = "\n\n".join(f"**Art. {k} — {r['rubrica']}**\n\nParagrafo su {k}." for k, r in records.items())
    log = {}
    text, rows = flow.ledger_out(f"**2. {NORMS}**\n\n{blocks}\n\n**5. {CLOSING}**\n\nX.",
                                 {"records": records, "main_anchor": "1490 c.c."}, log)
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert norms.count("**Art.") == flow.BODY_BLOCKS_MAX == 7
    assert norms.index("Art. 1490") < norms.index("Art. 1453") < norms.index("Art. 1385") < norms.index("Art. 1470")
    assert log["capped"] == ["1474 c.c.", "1475 c.c.", "1476 c.c.", "1477 c.c."]
    capped = [r for r in rows if r["article"] in log["capped"] and r.get("requisite") is None]
    assert capped and all(r["reason"] == C.LINE_REASON_CAPPED and not r["in_body"] for r in capped)


def test_the_writer_is_given_seven_articles_at_most():
    records = _eleven()
    for rec in records.values():
        rec["named_due"] = True                                          # every one due
    planned = [flow._parse_label(k) for k in records]
    got = flow.glossed_refs({"records": records, "main_anchor": "1490 c.c."}, [], {}, "", planned)
    assert len(got) == 7 and flow._label(flow._key(got[0])) == "1490 c.c."


# --- 4. the plain cenni: its area, its notes, no closing question ---------------------------------------------------------------

INDEX = {"c.c.": [{"name": "Della prescrizione e della decadenza", "level": "TITOLO",
                   "articles": [str(n) for n in range(2934, 2970)]},
                  {"name": "Della prescrizione", "level": "CAPO", "articles": [str(n) for n in range(2934, 2964)]}],
         "c.p.": [{"name": "Della estinzione del reato", "rubricated": "prescrizione", "level": "CAPO",
                   "articles": ["150", "157", "158"]}],
         "c.p.c.": [], "c.p.p.": []}


def _row(doc, number, kind="primary", text="…"):
    return {"d": {"id": doc, "name": doc, "document_type": kind}, "s": {"name": f"{number}.0.0", "plain_text": text}}


def test_a_plain_cenni_stays_inside_the_area_its_subject_names():
    area = plain_grounding.cenni_area("Cenni sulla prescrizione civile", INDEX.get)
    assert (area["code"], area["level"], area["articles"][0]) == ("c.c.", "TITOLO", "2934")
    assert plain_grounding.cenni_area("Cenni sulla prescrizione", INDEX.get) is None          # c.c. or c.p.: no guess
    assert plain_grounding.cenni_area("Cenni sulla prescrizione del reato", INDEX.get)["code"] == "c.p."
    rows = [_row("Codice di procedura civile", "152"), _row("Statuto del contribuente", "3"),
            _row("Codice Civile", "2946")]
    notes = [_row("Codice Penale Commentato", "157", "special"), _row("Codice Civile Commentato", "2934", "special")]
    asked = []

    def fetch(refs):
        asked.extend(refs)
        return [{**_row("Codice Civile", n), "_article": (n, c)} for n, c in refs]
    log = {}
    kept, same = plain_grounding.cenni_restrict("Cenni sulla prescrizione civile", rows, notes, log, INDEX.get, fetch)
    assert [plain_grounding._ref_of(r) for r in kept] == [("2946", "c.c."), ("2934", "c.c."), ("2935", "c.c."),
                                                          ("2936", "c.c."), ("2937", "c.c.")]
    assert asked == [("2934", "c.c."), ("2935", "c.c."), ("2936", "c.c."), ("2937", "c.c.")]
    assert [n["d"]["name"] for n in same] == ["Codice Civile Commentato"] and log["notes_dropped"] == 1


def test_a_plain_cenni_never_closes_with_a_question():
    assert plain_grounding.no_closing_question("La prescrizione estingue il diritto.\n\nVuoi approfondire i termini "
                                               "brevi?") == "La prescrizione estingue il diritto."
    assert plain_grounding.no_closing_question("Si estingue il diritto. Ti serve altro?") == "Si estingue il diritto."
    from src.rag.nodes import synthesis
    assert synthesis.generate_clarifying_question({"deliverable": "cenni", "citations": [
        {"document_name": "Codice Civile", "document_type": "primary"},
        {"document_name": "Commentario", "document_type": "special"}]}) == {}
    source = inspect.getsource(synthesis.synthesize_answer)
    assert "plain_grounding.cenni_restrict(" in source and "plain_grounding.no_closing_question(answer)" in source


# --- 5. a calculation needs a date; else the term and its article ------------------------------------------------------------

def test_a_calculation_needs_a_date_to_count_from():
    assert intake.deliverable_rule("Calcola il termine per la costituzione del convenuto.")[0] == "quesito"
    assert intake.deliverable_rule("Calcola il termine per la costituzione del convenuto: la citazione è stata "
                                   "notificata il 12/03/2026.")[0] == "calcolo"
    assert intake.decorrenza_date("notificato ieri") and not intake.decorrenza_date("entro quale termine")


def test_the_calculator_never_answers_a_quesito(monkeypatch):
    class Never:
        def match(self, query):
            raise AssertionError("the platform asked")
    monkeypatch.setattr(calculation, "PlatformClient", Never)
    out = calculation.calculation_gate({"query": "Entro quale termine deve costituirsi il convenuto?",
                                        "deliverable": "quesito"})
    assert out == {"calc_route": "normal"}


def test_the_quesito_gets_the_stores_term_with_its_article():
    text = TEXTS["166 c.p.c."]["commi"][0][1]
    (term,) = pf.extract("166", "c.p.c.", [(1, text)])["termini"]
    assert (term["subject"], term["qty"], term["text"], term["since"]) == (
        "costituzione", 70, "almeno settanta giorni", "prima dell'udienza di comparizione")
    rows = [_row("Codice di procedura civile", "166", text=text), _row("Codice Civile", "2934")]
    assert plain_grounding.store_lines(rows) == [
        "- art. 166 c.p.c.: Costituzione: almeno settanta giorni prima dell'udienza di comparizione (art. 166, testo, "
        "c.p.c.)"]
    from src.rag.nodes import synthesis
    assert "plain_grounding.STORE_TERMS_NOTE" in inspect.getsource(synthesis.synthesize_answer)


# --- 6. the rubrica rule on «istituto (art. N)» ----------------------------------------------------------------------------------

RECORDS = {"696-bis c.p.c.": {"rubrica": "Consulenza tecnica preventiva ai fini della composizione della lite",
                              "requisites": []},
           "166 c.p.c.": {"rubrica": "Costituzione del convenuto", "requisites": []}}


def test_a_parenthesised_citation_with_another_institutions_number_takes_the_records_article():
    sentence = "Il convenuto cura la costituzione in giudizio (art. 696-bis c.p.c.) nei termini."
    out, log = record_check.check([(LEAD, sentence)], {"records": RECORDS, "rubriche": {}, "texts": {},
                                                       "allowed": set(RECORDS), "parties": []})
    assert out[0][1] == "Il convenuto cura la costituzione in giudizio (art. 166 c.p.c.) nei termini."   # no row read
    miscited = []
    text = flow.rubrica_out(f"Primo. {sentence}", {"records": RECORDS}, [], miscited)
    assert text == "Primo. Il convenuto cura la costituzione in giudizio (art. 166 c.p.c.) nei termini."
    assert miscited[0]["now_article"] == "166 c.p.c."                         # a requested section too


def test_the_requested_sections_are_checked_for_the_rubrica():
    def write(jobs):
        return [(t, "La costituzione in giudizio (art. 696-bis c.p.c.) va fatta per tempo.") for t, *_ in jobs]
    blocks = [{"kind": "quesito", "subject": "entro quando costituirsi"}]
    out = flow.requested_sections({"records": RECORDS}, f"**1. {LEAD}**\n\nX.", blocks, write, {}, "Il caso.")
    assert "(art. 166 c.p.c.)" in out and "696-bis" not in out
