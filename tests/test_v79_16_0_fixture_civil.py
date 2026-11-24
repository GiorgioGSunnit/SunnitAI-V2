"""v79.16.0 -- the synthetic CIVIL fixture end to end: a claimant with three numbered quesiti, one of which (the
prescription term) needs an article the facts never imply (2946 c.c.). Invented names and facts; statutory text of
1385, 1453, 2043 and 2946 c.c. Through the real entry (tests/_fixture_79_15_0.py) with ANALYSIS_STEP, CASE_FRAME,
APPLICABILITY_FIT, LAWYER_METHOD and, this version's, FACT_RECORD and QUESTION_RETRIEVAL on; every model faked at its
source (the fact record's reply, the quesiti's search and the rows they fetch included), every other call refused.

Expected (the brief's A-D): the fact record built «ok» and the numbered sentences its facts (each carrying its F-id);
the analysis step's provenance from the record (the counterparty's allegation «contestato (allegazione di Ettore
Bassi)», the client's payment «provato»), the F-ids in the step's log line; §2 from the record (certi / ipotizzabili
by the record's stato, «Fatti mancanti» the open elements -- never «Nessun elemento resta da provare»); the three
questions searched (retrieval_quesiti: 2946 c.c. brought in by the third alone, kept by the fit, offered to its
writer call first); the Q&A one answer per numbered item in the user's order, the third's reviewed answer emptied,
re-asked once and still empty -> the insufficient line (qa_answer_reasked, qa_answer_insufficient); every writer call
with the verified context block; §3, §5, §6, §9 opened by the code's bridges; no sentence printed twice
(sections_dedup_dropped); §10 the open elements' questions, no prose from §4.
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
import test_v79_15_0_fixture_civil as CIVIL  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402

CASE_ID = "v79160_civil"
CLIENT = "Anna Corsi"
BASSI = CIVIL.BASSI
Q1 = "Anna Corsi può esigere il doppio della caparra?"
Q2 = "Può chiedere anche il risarcimento del danno?"
Q3 = "Entro quale termine si prescrive l'azione di Anna Corsi?"
FACT_SIGNED = ("Anna Corsi ha firmato con Ettore Bassi un contratto preliminare di compravendita di un appartamento")
FACT_PAID = "Anna Corsi ha versato a Ettore Bassi 15.000 euro a titolo di caparra confirmatoria"
ASK = "Anna Corsi vuole recedere dal contratto ed esigere il doppio della caparra ai sensi dell'art. 1385 c.c."
TEXT = (f"Assisto Anna Corsi. {FACT_SIGNED}. {FACT_PAID}. {BASSI}. {ASK}\n"
        f"Si chiede:\n1. {Q1}\n2. {Q2}\n3. {Q3}")
ART_2946 = "Salvi i casi in cui la legge dispone diversamente, i diritti si estinguono per prescrizione con il decorso di dieci anni."
ARTICLES = CIVIL.ARTICLES
QUESITO_ARTICLES = (("2946", "c.c.", ART_2946, "Prescrizione ordinaria", "LEGAL_DOC::cc", "Codice Civile 2026"),)
DOCUMENTS = CIVIL.DOCUMENTS
TITLE, ANCHOR, RUBRICA = CIVIL.TITLE, CIVIL.ANCHOR, CIVIL.RUBRICA
ENV = {"FACT_RECORD": "true", "QUESTION_RETRIEVAL": "true"}
FRAME_REPLY = {**CIVIL.FRAME_REPLY, "quesiti": [Q1, Q2, Q3]}

# the fact record's reply: the text's elements, the exact words, their kind, a fact's status and speaker
RECORD = [
    ("Assisto Anna Corsi.", "istruzione", None, None),
    (FACT_SIGNED, "fatto", "accertato", None),
    (FACT_PAID, "fatto", "accertato", None),
    (BASSI, "fatto", "allegato", "Ettore Bassi"),
    (ASK, "richiesta", None, None),
    ("Si chiede:", "estraneo", None, None),
    (Q1, "richiesta", None, None), (Q2, "richiesta", None, None), (Q3, "richiesta", None, None),
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
    """The quesiti's search faked: the prescription question brings 2946 c.c., the others the claim article."""
    if "prescrive" in query:
        return [("2946", "c.c."), ("1385", "c.c.")]
    return [("1385", "c.c.")]


_sid = CIVIL._sid


def fit_reply(messages, kwargs):
    human = messages[-1].content
    if human.startswith("PRETESA") and "ARTICOLI APPLICABILI" in human:     # v79.17.1 (item 9): the pretesa's recall --
        return json.dumps({"governa_pretesa": ["A1385cc"]})                    # the first pass tags no «governa» here
    paid = _sid(human, "Anna Corsi ha versato")
    signed = _sid(human, "Anna Corsi ha firmato")
    items = {"1385 c.c.": F._fit_item("1385 c.c.", "si", f"Anna Corsi ha versato la somma ({paid}): «a titolo di "
                                                       "caparra, una somma di danaro».", cf=[paid]),
             "1453 c.c.": F._fit_item("1453 c.c.", "si", f"Il contratto ({signed}) è un contratto «con prestazioni "
                                                       "corrispettive».", ambito="rimedio", cf=[signed]),
             "2043 c.c.": F._fit_item("2043 c.c.", "si", "«Qualunque fatto doloso o colposo» obbliga «a risarcire il "
                                                       "danno».", ambito="rimedio", cf=[signed]),
             "2946 c.c.": F._fit_item("2946 c.c.", "si", "«I diritti si estinguono per prescrizione con il decorso di "
                                                       "dieci anni».", ambito="rimedio", cf=[signed])}
    labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
    return json.dumps({"giudizi": [items[x] for x in labels if x in items]}, ensure_ascii=False)


link = CIVIL.link
analysis_reply = CIVIL.analysis_reply
WRITER = {**CIVIL.WRITER,
          "2946 c.c.": "L'art. 2946 c.c. prevede che i diritti si estinguono per prescrizione con il decorso di dieci anni."}
QA_KEPT, QA_DROPPED = CIVIL.QA_KEPT, CIVIL.QA_DROPPED
QA_PRESCRIZIONE = ("L'art. 2946 c.c. prevede che i diritti si estinguono per prescrizione con il decorso di dieci "
                   "anni dal giorno in cui possono essere fatti valere.")


def writer_line(label):
    return WRITER[label]


def writer_question(title):
    if "prescrive" in title:        # a rule on 2946 c.c. with no verified counterpart: dropped, re-asked, still empty
        return QA_PRESCRIZIONE
    return f"{QA_KEPT} {QA_DROPPED}"


case = sys.modules[__name__]


def test_the_civil_fixture_end_to_end(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, case)
    state, answer, by = run["state"], run["answer"], run["by"]
    c = state["block_counters"]
    target = os.environ.get("FIXTURE_79_16_0_OUT")
    if target:
        H.write_report(target, "v79.16.0 synthetic CIVIL fixture (tests/test_v79_16_0_fixture_civil.py)", run, "w")
    assert [x for x in run["failed"] if x.split(":")[0] in ("analysis_step", "applicability_fit", "legacy_plan",
                                                             "lawyer_method_compose", "lawyer_method_plan",
                                                             "verified_context")] == []
    # A1: the record built «ok», the numbered sentences its facts, no fallback
    record = state["fact_record"]
    assert record["status"] == "ok" and c.get("fact_record_fallback", 0) == 0
    sents = [s for s in state["sentences"] if s["source"] == "caso"]
    assert sents and all(s.get("fact") for s in sents), sents
    assert [s["text"] for s in sents] == [FACT_SIGNED, FACT_PAID, BASSI]       # the requests and the ask never facts
    assert next(s for s in sents if s["text"] == BASSI)["said_by"] == "Ettore Bassi"
    assert by["fact_record"][0][0]["status"] == "ok"
    # A2: the provenance from the record -- the client's payment provato, the counterparty's allegation contestato
    step = by["analysis_step"][0][0]
    assert step["status"] == "ok" and c["analysis_provenance_from_record"] == 2 and c["analysis_provenance_fallback"] == 0
    assert c["analysis_provato_downgraded"] == 1
    col = H.esito_column(answer)
    assert any("| contestato (allegazione di Ettore Bassi)" in x for x in col), col
    assert any("a titolo di caparra" in x and x.endswith("| provato") for x in col), col
    # A4: the F-ids in the log line, the facts' text on the page
    cited = step["facts_cited"]["1385 c.c."]
    assert cited and all(re.fullmatch(r"F\d+", f) for ids in cited.values() for f in ids), cited
    assert not re.search(r"\bF\d+\b", H.section(answer, "norme"))
    # A3: §2 from the record
    fatti = H.section(answer, "fatti")
    certi = fatti.split("**Fatti ipotizzabili**")[0]
    assert FACT_PAID in certi and FACT_SIGNED in certi and BASSI not in certi
    assert f"- {BASSI} (allegazione di Ettore Bassi)" in fatti
    mancanti = fatti.split("**Fatti mancanti**")[1]
    assert "Nessun elemento resta da provare" not in mancanti
    assert mancanti.count("da provare —") == 3 and "contestato —" in mancanti, mancanti
    assert "apre" not in mancanti and "— se «" in mancanti
    # B: the three questions searched; 2946 c.c. brought by the third only, kept by the fit, in its writer call first
    q_log = by["retrieval_quesiti"][0][0]
    assert q_log["queries"] == 3 and [e["n"] for e in q_log["per_quesito"]] == [1, 2, 3]
    assert q_log["per_quesito"][2]["top"][0] == "2946 c.c." and q_log["added"] == ["2946 c.c."]
    assert c["retrieval_quesiti_queries"] == 3 and c["retrieval_quesiti_added"] == ["2946 c.c."]
    assert c["fit_kept_from_quesiti"] == 1
    q3_jobs = [j for j in run["jobs"] if "prescrive" in str(j[0])]
    # v79.17.0 (item 4): the question's call receives the verified material alone -- its own article (2946 c.c.) with
    # its periods by id, the pretesa, the lines, the facts by id; no CASO block, no SCHEDE
    assert q3_jobs and q3_jobs[0][2].startswith("MATERIALE VERIFICATO") and "art. 2946 c.c." in q3_jobs[0][2]
    assert "[A2946cc.1]" in q3_jobs[0][2] and "CASO:" not in q3_jobs[0][2] and "Quesito Q3" in q3_jobs[0][2]
    assert "Pretesa: domanda: «esigere il doppio della caparra" in q3_jobs[0][2]
    # D1 -> v79.17.0 (item 4): every §4 and quesito writer call carries the one verified material block
    assert all("MATERIALE VERIFICATO" in j[2] for j in run["jobs"] if j[0].startswith("Norme") or "?" in j[0]), \
        [j[0] for j in run["jobs"]]
    # C1: one answer per numbered item, in the user's order
    assert c["qa_questions_detected"] == 3
    qa = H.section(answer, "risposte") if H.section(answer, "risposte") else answer.split("**1. ")[0]
    items = [x for x in qa.splitlines() if re.match(r"^\d\. «", x)]
    assert [x[:2] for x in items] == ["1.", "2.", "3."] and Q1 in items[0] and Q2 in items[1] and Q3 in items[2]
    # C2 -> v79.17.0 (item 4): the third answer's 2946 c.c. sentence cites a fitted article -- kept as written (no
    # lemma matching), nothing insufficient; the first answer's sentences likewise
    assert QA_PRESCRIZIONE in items[2] and QA_KEPT in items[0]
    assert c["qa_answer_insufficient"] == 0 and c["qa_answer_fragment"] == 0
    assert "Il corpus non contiene informazioni sufficienti per rispondere a questo quesito." not in answer
    assert A.LAW_STATEMENTS_TITLE not in answer
    # D2: the bridges
    assert H.section(answer, "questione").startswith("La questione è se la domanda «esigere il doppio della caparra ai "
                                                     "sensi dell'art. 1385 c.c.» sia fondata sull'art. 1385 c.c., posto che")
    assert H.section(answer, "orientamenti").startswith("Pronunce sulle norme della linea: art. 1385 c.c.")
    assert H.section(answer, "soluzioni").startswith("Le linee seguono gli esiti del punto 4.")
    assert H.section(answer, "conclusioni").startswith("**Esiti dell'art. 1385 c.c.**")
    # A3 / D: §10 the open elements' questions, code-written; D3: the dedup counter set
    verificare = H.section(answer, "verificare")
    assert "Da verificare se «" in verificare and "Nessun punto" not in verificare
    assert isinstance(c["sections_dedup_dropped"], int) and c["rulings_off_line_dropped"] == 0
    assert c["line_table_weaker"] == 0
    # the 79.15 invariants hold
    assert "**Rimedi e parte generale collegati alla linea**: art. 1453 c.c." in H.section(answer, "norme")
    assert by["legacy_plan"][0][0]["rendered"] is False
