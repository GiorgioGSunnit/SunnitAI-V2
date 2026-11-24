"""v79.16.0 -- the synthetic PENAL fixture end to end: the client is the victim (persona offesa) of an online sale that
never arrived, and her one numbered question holds three «?» («Marta Serra può costituirsi parte civile? A quali
condizioni? Con quali limiti?»). Invented names and facts; statutory text of 640 c.p. and 74 c.p.p. Through the real
entry (tests/_fixture_79_15_0.py) with ANALYSIS_STEP, CASE_FRAME, APPLICABILITY_FIT, LAWYER_METHOD, FACT_RECORD and
QUESTION_RETRIEVAL on; every model faked at its source.

Expected: the question read as ONE item (qa_questions_detected 1, one answer, never «A quali condizioni?» answered on
its own); the record's provenance -- the client's own allegation «provato» (chi = the client), the counterparty's
«contestato (allegazione di Luca Prandi)»; the question's search brings 74 c.p.p. (the facts never imply it), kept by
the fit, offered first to the question's writer call, its verified affermazione the answer's law; §2 and §10 from the
record and the verified elements; the bridges; the context block in every writer call.
"""

import json
import os
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _fixture_79_15_0 as H  # noqa: E402
import test_v79_14_2_fixture as F  # noqa: E402

CASE_ID = "v79160_penal"
CLIENT = "Marta Serra"
FACT_PAID = "Marta Serra ha pagato a Luca Prandi 3.000 euro con un bonifico per un motorino annunciato online"
MARTA = "Secondo Marta Serra, Luca Prandi ha usato artifizi e raggiri e non ha mai consegnato il motorino"
PRANDI = "Luca Prandi sostiene di aver spedito il motorino con un corriere"
QUERELA = "Marta Serra ha sporto querela per truffa"
QUESTION = "Marta Serra può costituirsi parte civile? A quali condizioni? Con quali limiti?"
TEXT = (f"Assisto Marta Serra, persona offesa. {FACT_PAID}. {MARTA}. {PRANDI}. {QUERELA}. Il pubblico ministero "
        f"procede per truffa ai sensi dell'art. 640 c.p.\n1. {QUESTION}")
ART_74 = ("L'azione civile per le restituzioni e per il risarcimento del danno di cui all'articolo 185 del codice penale "
          "può essere esercitata nel processo penale dal soggetto al quale il reato ha recato danno ovvero dai suoi "
          "successori universali, nei confronti dell'imputato e del responsabile civile.")
ARTICLES = (("640", "c.p.", F.ART_640, "Truffa", "LEGAL_DOC::cp", "Codice Penale 2026"),)
QUESITO_ARTICLES = (("74", "c.p.p.", ART_74, "Legittimazione all'azione civile", "LEGAL_DOC::cpp",
                     "Codice di procedura penale 2026"),)
DOCUMENTS = ({"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"},
             {"id": "LEGAL_DOC::cpp", "name": "Codice di procedura penale 2026"})
TITLE, ANCHOR, RUBRICA = "Truffa nella vendita online", "640 c.p.", "Truffa."
ENV = {"FACT_RECORD": "true", "QUESTION_RETRIEVAL": "true"}
FRAME_REPLY = {
    "parti": [{"nome": "Marta Serra", "ruolo_nel_fatto": "acquirente", "lato": "cliente"},
              {"nome": "Luca Prandi", "ruolo_nel_fatto": "venditore", "lato": "controparte"}],
    "cliente": {"nome": "Marta Serra", "perche": "«Assisto Marta Serra, persona offesa»"},
    "ruolo": {"valore": "persona offesa", "perche": "«Assisto Marta Serra, persona offesa»"},
    "rito": {"valore": "penale", "perche": "«Il pubblico ministero procede per truffa»"},
    "stadio": {"valore": "indagini", "perche": "«ha sporto querela»"},
    "addebito_o_domanda": {"articolo": "640 c.p.", "descrizione": "truffa",
                           "frase": "«procede per truffa ai sensi dell'art. 640 c.p.»"},
    "istituto": None,
    "quesiti": [QUESTION]}

RECORD = [
    ("Assisto Marta Serra, persona offesa.", "istruzione", None, None),
    (FACT_PAID, "fatto", "accertato", None),
    (MARTA, "fatto", "allegato", "Marta Serra"),
    (PRANDI, "fatto", "allegato", "Luca Prandi"),
    (QUERELA, "fatto", "accertato", None),
    ("Il pubblico ministero procede per truffa ai sensi dell'art. 640 c.p.", "fatto", "accertato", None),
    (QUESTION, "richiesta", None, None),
]


def fact_record_reply(chunk):
    out = []
    for text, tipo, stato, chi in RECORD:
        if text in chunk:
            item = {"citazione": text, "tipo": tipo}
            if stato:
                item["stato"] = stato
            if chi:
                item["chi"] = chi
            out.append(item)
    return out


def quesito_search(query):
    return [("74", "c.p.p."), ("640", "c.p.")] if "parte civile" in query else [("640", "c.p.")]


def _sid(parsed_or_human, start):
    facts = parsed_or_human if isinstance(parsed_or_human, dict) else {
        i: t for i, t in re.findall(r"^(S\d+) \[[^\]]+\] (.*)$", parsed_or_human, re.MULTILINE)}
    return next(i for i, t in facts.items() if t.startswith(start))


def fit_reply(messages, kwargs):
    human = messages[-1].content
    if human.startswith("PRETESA") and "ARTICOLI APPLICABILI" in human:     # v79.17.1 (item 9): the pretesa's recall --
        return json.dumps({"governa_pretesa": ["A640cp"]})                    # the first pass tags no «governa» here
    paid = _sid(human, "Marta Serra ha pagato")
    items = {"640 c.p.": F._fit_item("640 c.p.", "si", f"Marta Serra ha pagato ({paid}): «procura a sé o ad altri un "
                                                     "ingiusto profitto con altrui danno».", cf=[paid]),
             "74 c.p.p.": F._fit_item("74 c.p.p.", "si", "«Può essere esercitata nel processo penale dal soggetto al "
                                                       "quale il reato ha recato danno».", ambito="rimedio", cf=[paid])}
    labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
    return json.dumps({"giudizi": [items[x] for x in labels if x in items]}, ensure_ascii=False)


def link(topic, reqs, sents):
    sid = {s["text"]: s["id"] for s in sents}
    marta = next(i for t, i in sid.items() if t.startswith("Secondo Marta Serra"))
    prandi = next(i for t, i in sid.items() if t.startswith("Luca Prandi sostiene"))
    out = {}
    for label, r in reqs:
        text = str(r.get("text") or "").lower()
        if label.startswith("640") and "artifizi" in text:
            out[r["id"]] = {"sostiene": [marta], "contraddice": [], "manca": ""}
        elif label.startswith("640") and "profitto" in text:
            out[r["id"]] = {"sostiene": [prandi], "contraddice": [], "manca": ""}
    return out


def _pick(elements, word):
    return next(e for e in elements if word in e.lower())


def analysis_reply(parsed, attempt):
    """One call: the line on 640's artifizi; «provato» on the client's own allegation (stands: chi = the client) and
    on the counterparty's (downgraded: contestato, allegazione di Luca Prandi); one verified statement on 640 c.p.,
    one on 74 c.p.p. (the question's article)."""
    facts, norms = parsed["facts"], parsed["norms"]
    el = norms["640 c.p."]["elements"]
    marta, prandi = _sid(facts, "Secondo Marta Serra"), _sid(facts, "Luca Prandi sostiene")
    artifizi, profitto = _pick(el, "artifizi"), _pick(el, "profitto")

    def ref(e):
        return H.eid(parsed, "640 c.p.", e)
    elementi = []
    for e in el:
        if e == artifizi:
            elementi.append({"articolo": "640 c.p.", "elemento": ref(e), "esito": "provato", "cita_fatti": [marta],
                             "parte_che_afferma": None})
        elif e == profitto:
            elementi.append({"articolo": "640 c.p.", "elemento": ref(e), "esito": "provato", "cita_fatti": [prandi],
                             "parte_che_afferma": None})
        else:
            elementi.append({"articolo": "640 c.p.", "elemento": ref(e), "esito": "da_provare", "cita_fatti": [],
                             "parte_che_afferma": None})
    statements = [{"testo": "Chiunque, con artifizi o raggiri, inducendo taluno in errore, procura a sé o ad altri un "
                            "ingiusto profitto con altrui danno, è punito.",
                   "cita_norma": [norms["640 c.p."]["periods"][0]]}]
    if "74 c.p.p." in norms:
        statements.append({"testo": "L'azione civile per le restituzioni e per il risarcimento del danno può essere "
                                    "esercitata nel processo penale dal soggetto al quale il reato ha recato danno.",
                           "cita_norma": [norms["74 c.p.p."]["periods"][0]]})
    return {"linea_principale": {"articolo": "640 c.p.", "elemento_in_gioco": ref(artifizi),
                                 "tesi": "Gli artifizi risultano dalla denuncia di Marta Serra.",
                                 "cita_fatti": [marta], "cita_norma": [norms["640 c.p."]["periods"][0]]},
            "linea_subordinata": None, "elementi": elementi, "affermazioni_di_diritto": statements}


WRITER = {"640 c.p.": "L'art. 640 c.p. punisce chi, con artifizi o raggiri, inducendo taluno in errore, procura a sé o "
                      "ad altri un ingiusto profitto con altrui danno.",
          "74 c.p.p.": "L'art. 74 c.p.p. consente a chi ha subito un danno dal reato di esercitare l'azione civile nel "
                       "processo penale."}
QA_74 = ("L'art. 74 c.p.p. consente alla persona danneggiata dal reato di esercitare l'azione civile nel processo "
         "penale contro l'imputato.")


def writer_line(label):
    return WRITER[label]


def writer_question(title):
    return QA_74


case = sys.modules[__name__]


def test_the_penal_fixture_end_to_end(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, case)
    state, answer, by = run["state"], run["answer"], run["by"]
    c = state["block_counters"]
    target = os.environ.get("FIXTURE_79_16_0_OUT")
    if target:
        H.write_report(target, "v79.16.0 synthetic PENAL fixture (tests/test_v79_16_0_fixture_penal.py)", run, "a")
    assert [x for x in run["failed"] if x.split(":")[0] in ("analysis_step", "applicability_fit", "legacy_plan",
                                                             "lawyer_method_compose", "lawyer_method_plan",
                                                             "verified_context")] == []
    # C1: the numbered item with three «?» is one question, answered once, in place
    assert state["intake"]["questions"] == [QUESTION] and c["qa_questions_detected"] == 1
    qa = answer.split("**1. ")[0]
    items = [x for x in qa.splitlines() if re.match(r"^\d\. «", x)]
    assert len(items) == 1 and items[0].startswith(f"1. «{QUESTION}» — "), items
    assert "«A quali condizioni?»" not in answer and "«Con quali limiti?»" not in answer
    # A1 / A2: the record's sentences; the client's allegation provato, the counterparty's contestato
    assert state["fact_record"]["status"] == "ok" and c.get("fact_record_fallback", 0) == 0
    sents = {s["text"]: s for s in state["sentences"] if s["source"] == "caso"}
    assert sents[MARTA]["said_by"] == "Marta Serra" and sents[PRANDI]["said_by"] == "Luca Prandi"
    assert QUESTION not in sents
    assert c["analysis_provenance_from_record"] == 2 and c["analysis_provato_downgraded"] == 1
    col = H.esito_column(answer)
    assert any("artifizi" in x and x.endswith("| provato") for x in col), col
    assert any("profitto" in x and "| contestato (allegazione di Luca Prandi)" in x for x in col), col
    # B: the question's search brings 74 c.p.p., kept by the fit, first in its writer call; its statement the answer's
    q_log = by["retrieval_quesiti"][0][0]
    assert q_log["queries"] == 1 and q_log["per_quesito"][0]["top"][0] == "74 c.p.p." and q_log["added"] == ["74 c.p.p."]
    assert c["retrieval_quesiti_added"] == ["74 c.p.p."] and c["fit_kept_from_quesiti"] == 1
    q_jobs = [j for j in run["jobs"] if "parte civile" in str(j[0])]
    assert q_jobs and q_jobs[0][2].startswith("MATERIALE VERIFICATO") and "art. 74 c.p.p." in q_jobs[0][2]   # v79.17.0
    assert "[A74cpp.1]" in q_jobs[0][2] and "Pretesa: accusa: «truffa» (art. 640 c.p.)" in q_jobs[0][2]
    assert QA_74.split(". ")[0] in items[0]          # v79.17.0 (item 4): the writer's sentence stays, no replacement
    assert c["qa_answer_insufficient"] == 0
    # A3: §2 and §10 from the record and the verified elements
    fatti = H.section(answer, "fatti")
    assert FACT_PAID in fatti.split("**Fatti ipotizzabili**")[0]
    assert f"- {PRANDI} (allegazione di Luca Prandi)" in fatti
    # v79.18.0 (item 13): the client's own stated fact is never an allegation -- certo
    assert f"- {MARTA}\n" in fatti.split("**Fatti ipotizzabili**")[0] and "(allegazione di Marta Serra)" not in fatti
    assert "Nessun elemento resta da provare" not in fatti and "contestato —" in fatti
    assert "Da verificare se «" in H.section(answer, "verificare")
    # D2: the bridges; the step's log carries the F-ids
    assert H.section(answer, "questione").startswith("La questione è se l'accusa «truffa» sia fondata sull'art. 640 c.p., posto che")   # v79.17.0 (item 5)
    assert H.section(answer, "orientamenti").startswith("Pronunce sulle norme della linea: art. 640 c.p.")
    assert H.section(answer, "soluzioni").startswith("Le linee seguono gli esiti del punto 4.")
    assert by["analysis_step"][0][0]["facts_cited"]["640 c.p."]
    assert isinstance(c["sections_dedup_dropped"], int)
