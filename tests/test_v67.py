"""v67 (store on the graph's texts): (1) 325 c.p.c. co. 1 read as the corpus writes it -- «E'» for «È», an
abbreviation's dot inside the act («all'art. 404») --, and --show --text prints each comma the store reads; (2) a
term's subject is the act that owns its number («decreto … da emettere entro trenta giorni»: emissione del decreto),
so 641 c.p.c. co. 1 is two clear terms, no ambiguity; (3) «1-bis.» starts a comma of its own (bis / ter / quater …),
so 585 c.p.p. lett. c)'s condition stops at its comma; (4) a sentence with two «decorr…» is one decorrenza (326
c.p.c.). The texts are brocardi.it's (tests/data/store_articles.json), not the graph's."""

import json
import os
import sys
import unicodedata
from pathlib import Path

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripts"))

from src.rag import procedural_facts as pf, render_computed, rule_quote  # noqa: E402

TEXTS = json.loads((HERE / "data" / "store_articles.json").read_text(encoding="utf-8"))["articles"]


def _text(label):
    return "\n".join(f"{k}. {t}" for k, t in TEXTS[label]["commi"])        # numbered, «1-bis.» too, as the graph has it


def _facts(label, text=None):
    number, code = label.split(" ", 1)
    return pf.from_row({"_article": (number, code), "s": {"plain_text": text if text is not None else _text(label)}})


def _terms(facts):
    return [(t["comma"], t.get("letter"), t["subject"], t["qty"], t["unit"]) for t in facts["termini"]]


# --- (1) 325 c.p.c. co. 1 as the corpus writes it ----------------------------------------------------------------------------

def test_325_cpc_co_1_reads_its_terms_in_the_corpus_spellings():
    co1, co2 = (t for _, t in TEXTS["325 c.p.c."]["commi"])
    expected = [("co. 1", None, "appello, revocazione, opposizione di terzo", 30, "giorni"),
                ("co. 1", None, "revocazione, opposizione di terzo", 30, "giorni"),
                ("co. 2", None, "ricorso per cassazione", 60, "giorni")]
    variants = [co1.replace("È anche", "E' anche"),                                   # the corpus' «E'» (art. 150 c.c.)
                co1.replace("È anche", "E’ anche").replace("comma, è di", "comma, e' di"),
                co1.replace("all'articolo 404", "all'art. 404"),                       # an abbreviation's dot
                unicodedata.normalize("NFD", co1)]                                     # a decomposed accent
    for co1_variant in variants:
        assert _terms(_facts("325 c.p.c.", f"1. {co1_variant}\n2. {co2}")) == expected, co1_variant


def test_show_text_prints_each_comma_the_store_reads():
    import fill_procedural_store as fps
    lines = fps.text_lines({"s": {"plain_text": _text("585 c.p.p.")}}, "art. 585 c.p.p.")
    assert lines[0] == "  art. 585 c.p.p. | testo: 6 commi (lines)"
    assert lines[2].startswith("  art. 585 c.p.p. | co. 1-bis | «I termini previsti dal comma 1 sono aumentati")
    source = (HERE.parent / "scripts" / "fill_procedural_store.py").read_text(encoding="utf-8")
    assert 'parser.add_argument("--text"' in source and "if args.show and args.text:" in source


# --- (2) the subject is the act that owns the number --------------------------------------------------------------------------

def test_641_cpc_co_1_is_two_clear_terms_each_its_own_act():
    facts = _facts("641 c.p.c.")
    decree, payment = facts["termini"][:2]
    assert (decree["subject"], decree["qty"], decree["since"]) == ("emissione del decreto", 30,
                                                                   "dal deposito del ricorso")
    assert (payment["subject"], payment["qty"]) == ("pagamento della somma, consegna della cosa, opposizione", 40)
    assert not any(t.get("ambiguous") for t in facts["termini"]) and "termine_ambiguo" not in facts
    lines, info = pf.term_lines(facts, "641 c.p.c.")
    assert lines[0] == ("Emissione del decreto: trenta giorni dal deposito del ricorso, se esistono le condizioni "
                        "previste nell'articolo 633 (art. 641, co. 1, c.p.c.)")
    assert not info["ambiguous"]


def test_the_owner_of_a_number_in_each_form():
    cases = [("Il ricorso deve essere depositato in cancelleria entro venti giorni dalla notificazione.",
              "deposito del ricorso"),                                                     # a passive
             ("L'opposizione è proposta entro quindici giorni.", "opposizione"),          # «proposta»: the act itself
             ("Il pubblico ministero, entro trenta giorni dalla ricezione della notizia, richiede l'archiviazione.",
              "archiviazione"),                                                            # set off: the act after it
             ("Si propone con atto di citazione da notificarsi nel termine perentorio di venti giorni.",
              "notificazione della citazione"),                                            # «da <verb>»
             ("Il giudice ordina alla parte di depositare la memoria entro dieci giorni.", "deposito della memoria"),
             ("Il giudice ordina di depositare nella cancelleria i documenti entro dieci giorni.", "deposito"),
             ("Il convenuto deve costituirsi entro venti giorni.", "costituzione")]                 # a modal
    for text, subject in cases:
        (term,) = pf.termini([(1, text)])
        assert term["subject"] == subject, text


def test_section_4_finds_the_act_by_its_first_word():
    store = {"641 c.p.c.": _facts("641 c.p.c.")}
    assert [t["qty"] for t in pf.terms_for(store["641 c.p.c."], "opposizione al decreto ingiuntivo")] == [40]
    term = {"id": "opposizione_di", "label": "opposizione al decreto ingiuntivo", "ref": "art. 641 c.p.c.",
            "length": "40 giorni", "text": "opposizione: termine di 40 giorni NON DECORSO", "status": "NON DECORSO"}
    assert render_computed.store_term_line(term, store, []) == ""       # the store's forty days: the computed line kept


# --- (3) a «1-bis.» comma is its own ------------------------------------------------------------------------------------------

def test_a_suffixed_comma_starts_its_own_comma_in_lines_and_flat_text():
    for text in (_text("585 c.p.p."), " ".join(_text("585 c.p.p.").split())):
        commi = rule_quote.commi_of(text, suffixed=True)
        assert [k for k, _ in commi] == [1, "1-bis", 2, 3, 4, 5]
        assert [k for k, _ in rule_quote.commi_of(text)] == [1, 2, 3, 4, 5]      # every other reader: as before
        facts = pf.extract("585", "c.p.p.", commi)
        assert facts["termini"][2]["when"] == "nel caso previsto dall'articolo 544 comma 3"
    later = ("1. Il primo comma, lungo almeno quaranta lettere come ogni altro.\n2. Il secondo comma, lungo almeno "
             "quaranta lettere.\n2-ter. Un comma aggiunto dopo, lungo almeno quaranta lettere.\n2-quater. Un altro comma "
             "aggiunto dopo, lungo abbastanza.\n3. Il terzo comma, lungo almeno quaranta lettere anche lui.")
    assert [k for k, _ in rule_quote.commi_of(later, suffixed=True)] == [1, 2, "2-ter", "2-quater", 3]


def test_a_list_items_condition_stops_at_its_sentence():
    commi = [(1, "Il termine per proporre impugnazione è: a) di quindici giorni, nel caso previsto dall'art. 544 "
                 "comma 1; b) di trenta giorni, nel caso previsto dall'articolo 544 comma 3. I termini sono aumentati.")]
    first, second = pf.termini(commi)
    assert first["when"] == "nel caso previsto dall'art. 544 comma 1"
    assert second["when"] == "nel caso previsto dall'articolo 544 comma 3"


# --- (4) one decorrenza per sentence ------------------------------------------------------------------------------------------

def test_326_cpc_co_1_decorrenza_once():
    facts = _facts("326 c.p.c.")
    assert [d["comma"] for d in facts["decorrenze"]] == ["co. 1"]
    assert facts["decorrenze"][0]["text"].startswith("I termini stabiliti nell'articolo 325 sono perentori e decorrono")
    assert len(pf.from_lines(facts, "326 c.p.c.")) == 1


def test_an_older_entry_is_read_again():
    assert pf.VERSION == 4 and 3 in pf.READ_VERSIONS
    row = {"_article": ("641", "c.p.c."), "_proc": {"version": 3, "termini": [{"subject": "opposizione", "qty": 30}]},
           "s": {"plain_text": _text("641 c.p.c.")}}
    assert pf.by_number([row], "c.p.c.")["641"]["termini"][0]["subject"] == "emissione del decreto"
