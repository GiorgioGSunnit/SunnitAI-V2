"""v64: the review of the v63 sim -- the general part (c.p. Libro I) never a grading candidate, in the norms section
only through the concepts table (a cenni subject, the kind of a charged article's element), the posture table's
sanction articles for an imputato, or the user's own naming; an «applicazione» paragraph touching neither the client nor
a fact of the case a citations row only; the procedural section never for an alternative the alternatives section
rejects; §1 and the conclusions never stating a requirement of the charge its record does not list (a check, no longer
an instruction); no querela term once a charge is stated; no paragraph opening on a fragment. Synthetic records; the
model and the graph are stubbed."""

import os
import sys
from pathlib import Path

import pytest

for key, value in (("NEO4J_URI", "bolt://x"), ("NEO4J_USER", "n"), ("NEO4J_PASSWORD", "x"), ("LLM_API_KEY", "k"),
                   ("EMBEDDING_PROVIDER", "openai"), ("EMBEDDING_API_KEY", "k")):
    os.environ.setdefault(key, value)

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from src.rag import answer_ledger, cenni, issue_records, posture, record_check, render_computed  # noqa: E402
from src.rag import interaction_copy as C  # noqa: E402
from src.rag import interactive_flow as flow  # noqa: E402

PROVATO, DA_PROVARE = flow.PROVATO, flow.DA_PROVARE
NORMS, LEAD, CLOSING = flow.ANSWER_TITLES[1], flow.ANSWER_TITLES[0], flow.ANSWER_TITLES[4]
E584 = ["con atti diretti a commettere uno dei delitti preveduti dagli articoli 581 e 582", "cagiona la morte di un uomo"]
RUBRICHE = {"40": "Rapporto di causalità", "41": "Concorso di cause", "42": "Responsabilità per dolo o per colpa",
            "43": "Elemento psicologico del reato", "62": "Circostanze attenuanti comuni",
            "62-bis": "Circostanze attenuanti generiche", "133": "Gravità del reato: valutazione agli effetti della pena",
            "70": "Circostanze oggettive e soggettive", "83": "Evento diverso da quello voluto dall'agente",
            "584": "Omicidio preterintenzionale", "61": "Circostanze aggravanti comuni"}


def _row(number, code="c.p.", text="…"):
    return {"_article": (number, code), "s": {"titolo_articolo": RUBRICHE.get(number, ""), "plain_text": text}}


def _threat():
    reqs = [{"kind": "requisito", "comma": 1, "text": t, "status": PROVATO} for t in E584]
    return {"rubrica": "Omicidio preterintenzionale", "requisites": reqs, "side": "threat", "charged": 1}


# --- 1. the general part: never graded, in only through its own roads -----------------------------------------------------

@pytest.mark.parametrize("ref, general", [(("40", "c.p."), True), (("62-bis", "c.p."), True), (("240", "c.p."), True),
                                          (("241", "c.p."), False), (("584", "c.p."), False), (("40", "c.p.p."), False),
                                          (("1173", "c.c."), False)])
def test_the_general_part_is_c_p_libro_i(ref, general):
    assert cenni.general_part(ref) is general


def test_a_general_part_article_is_never_a_grading_candidate_unless_the_user_names_it():
    rows = [_row("70"), _row("83"), _row("584"), _row("61")]
    kept, dropped = cenni.general_part_gate(rows, {("61", "c.p.")})
    assert [r["_article"][0] for r in kept] == ["584", "61"] and dropped == ["70 c.p.", "83 c.p."]
    assert flow.general_part_out([("584", "c.p."), ("70", "c.p."), ("61", "c.p.")], {("61", "c.p.")}) == \
        [("584", "c.p."), ("61", "c.p.")]


def test_a_charged_elements_kind_names_its_general_part_articles():
    assert cenni.element_refs(E584[1], ["c.p."]) == [("40", "c.p."), ("41", "c.p.")]
    assert cenni.element_refs(E584[0], ["c.p."]) == [("42", "c.p."), ("43", "c.p.")]
    assert cenni.element_refs("cagiona per colpa una lesione personale", ["c.p."]) == \
        [("40", "c.p."), ("41", "c.p."), ("42", "c.p."), ("43", "c.p.")]
    assert cenni.element_refs("una malattia nel corpo o nella mente", ["c.p."]) == []
    assert cenni.element_refs(E584[1], ["c.c."]) == []                              # c.p.'s concepts only in its cases


def test_the_roads_the_threats_elements_and_the_imputatos_sanctions(monkeypatch):
    fetched = []

    def fetch(refs):
        fetched.append(list(refs))
        return [_row(n, c) for n, c in refs]
    state = {"records": {"584 c.p.": _threat(), "582 c.p.": {"rubrica": "Lesione personale", "requisites": [],
                                                              "side": "weapon"}}}
    imputato = posture.of({}, {})
    imputato.update(role="imputato", entry=posture.entry_for("penale", "imputato", "primo_grado"))
    roads = flow.general_rows(state, imputato, {"nesso causale": [_row("40")]}, {"frame": {"areas": ["penale"]}},
                              [_row("40"), _row("584")], fetch=fetch)
    labels = {why: [flow._label(flow._key(r["_article"])) for r in rows] for why, rows in roads.items()}
    assert labels == {"cenni: nesso causale": ["40 c.p."],
                      "elemento dell'addebito, art. 584 c.p.": ["42 c.p.", "43 c.p.", "40 c.p.", "41 c.p."],
                      C.APPLICATION_VIA_SANCTION: ["62 c.p.", "62-bis c.p.", "133 c.p."]}
    assert fetched == [[("42", "c.p."), ("43", "c.p."), ("41", "c.p."), ("62", "c.p."), ("62-bis", "c.p."),
                        ("133", "c.p.")]]                                          # once, only what no row holds
    victim = {**imputato, "role": "parte_civile"}
    assert C.APPLICATION_VIA_SANCTION not in flow.general_rows({"records": {}}, victim, {}, {}, [], fetch=fetch)


def test_the_sanctions_are_the_imputatos_entries_only():
    for stage in ("indagini", "primo_grado", "impugnazione"):
        assert posture.entry_for("penale", "imputato", stage)["sanctions"] == ["62 c.p.", "62-bis c.p.", "133 c.p."]
    assert "sanctions" not in posture.entry_for("penale", "parte_civile", "impugnazione")


# --- 2. an «applicazione» paragraph touches the client or a fact, or it is a citations row --------------------------------

def _app(why="cenni: x"):
    return {"rubrica": "", "requisites": [], "applicazione": why}


def test_an_application_that_touches_nothing_is_a_citations_row_only():
    state = {"client": {"name": "Sempronio"}, "records": {
        "70 c.p.": {**_app(), "rubrica": RUBRICHE["70"]}, "62 c.p.": {**_app(), "rubrica": RUBRICHE["62"]},
        "41 c.p.": {**_app(), "rubrica": RUBRICHE["41"]}},
        "sentences": [{"id": "S1", "text": "Sempronio sferra uno schiaffo a Caio, che cade battendo il capo sul "
                                           "marciapiede.", "certainty": "accertato"}]}
    draft = (f"**2. {NORMS}**\n\n**Art. 70 c.p. — {RUBRICHE['70']}**\n\nLe circostanze oggettive e soggettive "
             f"riguardano la natura, la specie, i mezzi e ogni altra modalità dell'azione.\n\n"
             f"**Art. 62 c.p. — {RUBRICHE['62']}**\n\nA Sempronio spetta l'attenuante del soccorso prestato.\n\n"
             f"**Art. 41 c.p. — {RUBRICHE['41']}**\n\nLa caduta sul marciapiede dopo lo schiaffo non interrompe il "
             f"nesso.\n\n**5. {CLOSING}**\n\nX.")
    texts = {"70 c.p.": "Sono circostanze oggettive quelle che concernono la natura, la specie, i mezzi, l'oggetto, "
                        "il tempo, il luogo e ogni altra modalità dell'azione."}
    moved = {}
    text, rows = flow.ledger_out(draft, state, moved, texts)
    norms = dict(answer_ledger.sections_of(text))[NORMS]
    assert "Art. 70 c.p." not in norms and "**Art. 62 c.p." in norms and "**Art. 41 c.p." in norms
    heads = {r["article"]: r for r in rows if r["requisite"] is None}
    assert heads["70 c.p."]["status"] == issue_records.APPLICATION and not heads["70 c.p."]["in_body"]
    assert heads["70 c.p."]["reason"] == C.LINE_REASON_NO_FACT and moved["ungrounded_applications"] == ["70 c.p."]
    assert heads["62 c.p."]["in_body"] and heads["41 c.p."]["in_body"]


# --- 3. the alternatives' verdicts decide the procedural section -----------------------------------------------------------

ALTS = [{"name": "lesioni personali", "rubricated": "Lesione personale"},
        {"name": "lesioni colpose", "rubricated": "Lesioni personali colpose"},
        {"name": "omissione di soccorso", "rubricated": "Omissione di soccorso"}]
ROWS = [{"_article": ("582", "c.p."), "s": {"titolo_articolo": "Lesione personale"}},
        {"_article": ("590", "c.p."), "s": {"titolo_articolo": "Lesioni personali colpose"}},
        {"_article": ("593", "c.p."), "s": {"titolo_articolo": "Omissione di soccorso"}}]
WRITTEN = ("[1] SÌ: lo schiaffo era diretto a ledere. Altro.\n[2] NO: la condotta fu dolosa, non colposa.\n"
           "[3] no — Sempronio prestò soccorso.")


def test_each_alternative_carries_the_writers_verdict():
    state = {"rows": ROWS, "read": {"alternatives": ALTS}}
    items, orphan = flow.parse_alternatives(WRITTEN, posture.alternative_rows(state))
    assert [(i["verdict"], i["sentence"]) for i in items] == [
        (flow.FITS, "Lo schiaffo era diretto a ledere."), (flow.REJECTED, "La condotta fu dolosa, non colposa."),
        (flow.REJECTED, "Sempronio prestò soccorso.")] and not orphan
    text = flow.alternatives_text(WRITTEN, posture.alternative_rows(state))
    assert text.startswith("**Art. 582 c.p. — Lesione personale**\nLo schiaffo era diretto a ledere.")
    assert "SÌ" not in text and "NO:" not in text


def test_the_alternatives_are_written_once_before_the_procedure_and_a_rejected_one_has_no_line():
    state = {"rows": ROWS, "read": {"alternatives": ALTS},          # v65: a «SÌ» stands on a sound record
             "records": {"582 c.p.": {"rubrica": "Lesione personale", "requisites": [
                 {"kind": "requisito", "comma": 1, "text": "cagiona una lesione", "status": PROVATO}]}}}
    calls = []

    def write(jobs):
        calls.append(jobs)
        return [(t, WRITTEN) for t, *_ in jobs]
    done = flow.write_alternatives(state, [], "Il caso.", write)
    assert done["fits"] == ["582 c.p."] and len(calls) == 1
    records = {"584 c.p.": {**_threat()}, "582 c.p.": {"rubrica": "Lesione personale", "alternative": True},
               "590 c.p.": {"rubrica": "Lesioni personali colpose", "alternative": True},
               "593 c.p.": {"rubrica": "Omissione di soccorso", "alternative": True}}
    offences = [("584", "Omicidio preterintenzionale"), ("582", "Lesione personale"),
                ("590", "Lesioni personali colpose"), ("593", "Omissione di soccorso")]
    kept, log = posture.procedure_offences(records, offences, done["fits"])
    assert kept == offences[:2] and log["rejected"] == ["590 c.p.", "593 c.p."] and log["fits_by"] == "sezione"
    body = f"**1. {LEAD}**\n\nLa difesa.\n\n**2. {NORMS}**\n\nX."
    out = flow.requested_sections(state, body, [{"kind": "parere"}, {"kind": "alternative"}],
                                  lambda jobs: pytest.fail("the alternatives are written once"), {}, "Il caso.",
                                  alternatives_done=done)
    assert dict(answer_ledger.sections_of(out))[C.ALTERNATIVES_SECTION].strip() == done["text"]


# --- 4. §1 and the conclusions state no requirement of the charge its record lacks ------------------------------------------

def _check(sentence, title=LEAD):
    ctx = {"asserting": [LEAD, CLOSING], "threats": {"584 c.p.": {"rubrica": "Omicidio preterintenzionale",
                                                                   "elements": E584}},
           "allowed": {"584 c.p."}, "parties": ["Sempronio", "Caio"]}
    return record_check.check([(title, f"\n\n{sentence} La difesa lo contesta.\n\n")], ctx)


@pytest.mark.parametrize("sentence, removed", [
    ("L'art. 584 c.p. richiede il dolo specifico per la morte.", True),
    ("L'omicidio preterintenzionale presuppone la consapevolezza del rischio di morte.", True),
    ("Ai fini dell'art. 584 c.p. occorre che vi sia il dolo eventuale rispetto alla morte.", True),
    ("L'art. 584 c.p. richiede che la morte sia conseguenza di atti diretti a ledere.", False),
    ("Per l'art. 584 c.p. è necessario che Sempronio abbia cagionato la morte di Caio.", False),
    ("La difesa richiede l'assoluzione di Sempronio dall'art. 584 c.p.", False),
])
def test_a_requirement_of_the_charge_must_be_a_recorded_element(sentence, removed):
    sections, log = _check(sentence)
    assert (sentence not in sections[0][1]) is removed and "La difesa lo contesta." in sections[0][1]
    if removed:
        assert log["rules"] == {"requisito_non_registrato": 1}
        assert "elementi registrati: «con atti diretti a" in log["removed"][0]["reason"]


def test_the_requirement_check_reads_only_the_lead_and_the_conclusions():
    sentence = "L'art. 584 c.p. richiede il dolo specifico per la morte."
    assert sentence not in _check(sentence, CLOSING)[0][0][1]
    assert sentence in _check(sentence, flow.ANSWER_TITLES[2])[0][0][1]


# --- 5. no querela term once a charge is stated ------------------------------------------------------------------------------

@pytest.mark.parametrize("stage, charged, term", [("indagini", True, False), ("indagini", False, True),
                                                   ("pre_filing", False, True), ("primo_grado", False, False)])
def test_the_querela_term_goes_once_a_charge_is_stated(stage, charged, term):
    facts = {"582": {"querela": True, "ufficio": [], "competenza": "tribunale monocratico", "pena_max": 3}}
    text = render_computed.criminal_procedure({
        "store_offences_only": True, "offences": [("582", "Lesione personale")], "offence_facts": facts,
        "charge_stated": charged, "posture": {"stage": stage, "family": "penale", "role": "imputato"}})
    assert "prescrizione del reato (art. 157 c.p.)" in text
    assert ("il termine per proporla" in text) is term


# --- 6. no paragraph opening on a fragment -----------------------------------------------------------------------------------

def test_a_paragraph_opening_with_a_parenthesis_or_lowercase_is_dropped():
    draft = (f"**1. {LEAD}**\n\nLa linea difensiva.\n\n(lesione personale dolosa) o all'art. 590 c.p. in subordine.\n\n"
             f"in subordine la riqualificazione.\n\n«Citazione» iniziale.\n\n**2. {flow.ANSWER_TITLES[3]}**\n\n"
             f"(nota del codice)\n\n**3. {CLOSING}**\n\nEsito.")
    log = []
    out = flow.fragment_paragraphs(draft, log)
    lead = dict(answer_ledger.sections_of(out))[LEAD]
    assert "lesione personale dolosa" not in lead and "in subordine la" not in lead
    assert "La linea difensiva." in lead and "«Citazione» iniziale." in lead
    assert "(nota del codice)" in out                                   # the procedural section is the code's
    assert len(log) == 2 and log[0].startswith("(lesione personale dolosa)")
    assert flow.fragment_paragraphs(f"**1. {LEAD}**\n\nTutto bene.", []) == f"**1. {LEAD}**\n\nTutto bene."
