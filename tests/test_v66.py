"""v66 (store v3): the term forms «il termine per <atto> è di N …», «è anche di N … il termine per <atto>», a lettered
list «è: a) di N …, <condition>; b) …» and «decorsi N …», each term with its sentence, condition and «dal …»; two numbers
for one act in one comma ambiguous -- termine null, the sentence quoted (v67: 641 c.p.c. is no such comma -- its two
numbers own two acts, tests/test_v67.py; the ambiguity's tests use a synthetic comma). Every reader of a stored term
calls one accessor (procedural_facts.term_lines): the quesito's material, §4's «Termini», the impugnazione closed list
(no number of its own). The texts of 325, 327, 641 c.p.c. and 585 c.p.p. are brocardi.it's (tests/data/
store_articles.json), not the graph's: the server's fill_procedural_store.py --show is the check against the store."""

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

from src.rag import answer_ledger, posture, procedural_facts as pf, render_computed  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

TEXTS = json.loads((HERE / "data" / "store_articles.json").read_text(encoding="utf-8"))["articles"]


def _facts(label):
    number, code = label.split(" ", 1)
    return pf.extract(number, code, [tuple(c) for c in TEXTS[label]["commi"]])


def _row(label, commi=None):
    number, code = label.split(" ", 1)
    text = "\n".join(f"{k}. {t}" for k, t in commi or TEXTS[label]["commi"])      # v67: «1-bis.» as the graph writes it
    return {"_article": (number, code), "s": {"plain_text": text}}


# v67: two numbers for one act in one comma, no condition between them -- synthetic (641 c.p.c. no longer is one)
AMBIGUOUS = ("999 c.p.c.", [[1, "L'opposizione è proposta entro trenta giorni dalla notificazione del decreto. "
                                 "L'opposizione è proposta entro quaranta giorni dalla notificazione del decreto."]])
STORE = {label: _facts(label) for label in TEXTS}
STORE[AMBIGUOUS[0]] = pf.extract("999", "c.p.c.", [tuple(c) for c in AMBIGUOUS[1]])


# --- (a) the forms ---------------------------------------------------------------------------------------------------------

def test_325_cpc_the_term_for_its_acts_either_order_of_number_and_unit():
    terms = STORE["325 c.p.c."]["termini"]
    assert [(t["comma"], t["subject"], t["qty"], t["unit"]) for t in terms] == [
        ("co. 1", "appello, revocazione, opposizione di terzo", 30, "giorni"),
        ("co. 1", "revocazione, opposizione di terzo", 30, "giorni"),         # «È anche di trenta giorni il termine per …»
        ("co. 2", "ricorso per cassazione", 60, "giorni")]                    # «è di giorni sessanta»
    assert not any(t.get("ambiguous") for t in terms) and STORE["325 c.p.c."]["termine"] == "trenta giorni (co. 1)"
    assert terms[2]["sentence"] == "Il termine per proporre il ricorso per cassazione è di giorni sessanta."


def test_585_cpp_a_lettered_list_one_term_per_letter_with_its_condition():
    facts = STORE["585 c.p.p."]
    terms = facts["termini"]
    assert [(t["letter"], t["subject"], t["qty"]) for t in terms] == [("a", "impugnazione", 15), ("b", "impugnazione", 30),
                                                                      ("c", "impugnazione", 45)]
    assert terms[0]["when"] == ("per i provvedimenti emessi in seguito a procedimento in camera di consiglio e nel caso "
                                "previsto dall'articolo 544 comma 1")
    assert terms[1]["when"] == "nel caso previsto dall'articolo 544 comma 2" and not facts.get("termine_ambiguo")
    assert terms[2]["sentence"].startswith("Il termine per proporre impugnazione, per ciascuna delle parti, è: c) di "
                                           "quarantacinque giorni")
    assert [d.get("letter") for d in facts["decorrenze"]] == [None, "a", "b", "c", "d"]   # co. 2, the law's words
    assert facts["decorrenze"][1]["text"].startswith("dalla notificazione o comunicazione dell'avviso di deposito")
    assert not any(t["qty"] == 15 and t["comma"] == "co. 1-bis" for t in terms)          # «aumentati di»: no term


def test_327_cpc_a_term_after_decorsi_with_its_decorrenza():
    (term,) = STORE["327 c.p.c."]["termini"]
    assert (term["subject"], term["qty"], term["unit"], term["since"]) == (
        "appello, ricorso per cassazione, revocazione", 6, "mesi", "dalla pubblicazione della sentenza")   # v67: owner()


# --- (b) two numbers for one act in one comma: ambiguous ----------------------------------------------------------------------

def test_two_numbers_for_one_act_in_one_comma_are_ambiguous():
    facts = STORE[AMBIGUOUS[0]]
    assert facts["termine"] is None and facts["termine_ambiguo"] is True
    first, second = facts["termini"]
    assert (first["qty"], second["qty"]) == (30, 40) and first["ambiguous"] and second["ambiguous"]
    assert first["subject"] == second["subject"] == "opposizione"
    assert first["sentence"] == "L'opposizione è proposta entro trenta giorni dalla notificazione del decreto."
    lines, info = pf.term_lines(facts, AMBIGUOUS[0])
    assert info["ambiguous"] and info["numbers"] == set()
    assert len(lines) == 2 and all(x.endswith("— il comma dà più termini per lo stesso atto: il termine va letto "
                                              "nel testo e la decorrenza è da verificare") for x in lines)
    assert not any(x.startswith(("Opposizione: trenta", "Opposizione: quaranta")) for x in lines)


# --- the readers: one accessor ------------------------------------------------------------------------------------------------

def test_the_quesito_quotes_an_ambiguous_term_and_states_a_clear_one():
    text = flow.quesito_material({"records": {}}, [], [_row(*AMBIGUOUS), _row("327 c.p.c.")], "")
    assert "- art. 999 c.p.c.: «L'opposizione è proposta entro trenta" in text and "Opposizione: trenta" not in text
    assert ("Appello, ricorso per cassazione, revocazione: sei mesi dalla pubblicazione della sentenza (art. 327, co. 1, "
            "c.p.c.)") in text


def _term(id_, label, ref, length, text):
    return {"id": id_, "label": label, "ref": ref, "length": length, "text": text, "status": "NON DECORSO"}


def test_section_4s_terms_take_the_stores_number_and_quote_an_ambiguous_one():
    log = []
    same = _term("impugnazione_penale", "impugnazione della sentenza penale", "art. 585 c.p.p.", "15, 30 o 45 giorni",
                 "impugnazione: termine di 15, 30 o 45 giorni NON DECORSO (sentenza 2024, oggi 2024)")
    assert render_computed.store_term_line(same, STORE, log) == ""                  # the store says the same: computed
    other = _term("appello_civile", "appello", "art. 325 c.p.c.", "20 giorni", "appello: termine di 20 giorni DECORSO")
    line = render_computed.store_term_line(other, STORE, log)
    assert line.startswith("- Appello: Appello, revocazione, opposizione di terzo: trenta giorni (art. 325, co. 1")
    assert line.endswith("— decorrenza da verificare") and "DECORSO" not in line and "20 giorni" not in line
    ambiguous = _term("opposizione_di", "opposizione al decreto", "art. 999 c.p.c.", "40 giorni",
                      "opposizione: termine di 40 giorni DECORSO (notifica 2023, oggi 2024)")
    line = render_computed.store_term_line(ambiguous, STORE, log)
    assert "«L'opposizione è proposta entro" in line and "DECORSO" not in line and "2024" not in line
    assert [x["how"] for x in log] == ["numero dell'archivio", "ambiguo: frase citata"]
    unknown = _term("x", "x", "art. 2051 c.c.", "1 anno", "x: un anno")
    assert render_computed.store_term_line(unknown, STORE) == ""                    # no store terms: the computed line
    analysis = {"fact_signals": {"terms": [ambiguous]}, "store_terms": STORE}
    termini = dict(render_computed.blocks(analysis))["Termini"]
    assert "«L'opposizione è proposta entro" in termini and "DECORSO" not in termini


def _post(rito, role):
    return {"rito": rito, "family": posture.family(rito), "role": role, "stage": "impugnazione",
            "entry": posture.entry_for(rito, role, "impugnazione"), "prior_decisions": [], "querela_stated": False}


def test_the_impugnazione_closed_list_renders_its_terms_from_the_store():
    penal = dict(render_computed.stage_blocks(_post("penale", "imputato"), STORE))["Termini"]
    assert ("- Impugnazione: quindici giorni, per i provvedimenti emessi in seguito a procedimento in camera di "
            "consiglio e nel caso previsto dall'articolo 544 comma 1 (art. 585, co. 1, lett. a), c.p.p.)") in penal
    assert "Decorrenza (art. 585, co. 2, lett. b), c.p.p.): «dalla lettura del provvedimento in udienza" in penal
    assert "Da verificare: il caso non dice la data della lettura" in penal
    civil = dict(render_computed.stage_blocks(_post("civile", "attore"), STORE))["Termini"]
    assert "Ricorso per cassazione: sessanta giorni (art. 325, co. 2, c.p.c.)" in civil
    assert ("Appello, ricorso per cassazione, revocazione: sei mesi dalla pubblicazione della sentenza (art. 327, co. 1, "
            "c.p.c.)") in civil
    assert civil.count("Decorrenza (art. 326, co. 1, c.p.c.): «I termini stabiliti nell'articolo 325") == 1     # v67
    unread = {k: v for k, v in STORE.items() if k != "326 c.p.c."}
    civil = dict(render_computed.stage_blocks(_post("civile", "attore"), unread))["Termini"]
    assert "Art. 326 c.p.c.: testo non letto, termini da verificare" in civil          # not in this store
    data = json.loads(Path(render_computed.STAGE_FILE).read_text(encoding="utf-8"))
    for family in ("penale", "civile"):
        block = next(b for b in data[family]["impugnazione"] if b["title"] == "Termini")
        assert block["store"] and not block.get("lines")                       # no second copy of the law
    assert render_computed.store_articles(_post("civile", "attore")) == ["325 c.p.c.", "326 c.p.c.", "327 c.p.c."]


def test_the_answer_reads_the_store_terms_it_shows_once():
    fetched = []

    def fetch(refs):
        fetched.append(list(refs))
        return [_row(f"{n} {c}") for n, c in refs if f"{n} {c}" in TEXTS]
    analysis = {"fact_signals": {"terms": [_term("x", "opposizione", "art. 641 c.p.c.", "40 giorni", "x")]}}
    got = flow.store_terms_for(_post("penale", "imputato"), analysis, [_row("327 c.p.c.")], fetch=fetch)
    assert fetched == [[("585", "c.p.p."), ("641", "c.p.c.")]] and sorted(got) == ["585 c.p.p.", "641 c.p.c."]
    assert [t["qty"] for t in pf.terms_for(got["641 c.p.c."], "opposizione")] == [40]     # v67: the act's own


def test_the_numbers_check_never_rewrites_an_ambiguous_term():
    store = flow.store_numbers([_row(*AMBIGUOUS)])
    assert store["999 c.p.c."]["ambiguous"] == ["giorni"]
    log = []
    draft = f"**1. {flow.ANSWER_TITLES[0]}**\n\nL'opposizione va proposta entro trenta giorni (art. 999 c.p.c.)."
    assert flow.store_numbers_out(draft, store, [], log) == draft and log[0]["store"] == "ambiguo"
