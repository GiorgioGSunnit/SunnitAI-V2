"""v79.15.0 build B -- the synthetic CIVIL fixture end to end (invented names and facts; statutory text of 1385, 1453
and 2043 c.c.) through the real entry, intake.request_intake -> interactive_flow.single_shot -> write_answer ->
_write_answer, with ANALYSIS_STEP, CASE_FRAME, APPLICABILITY_FIT and LAWYER_METHOD on; every model faked at its source,
every other call and every outbound socket refused (tests/_fixture_79_15_0.py).

The client, Anna Corsi, gave Ettore Bassi a caparra confirmatoria on a preliminary contract; he sold the flat to a
third party and blames her («Ettore Bassi sostiene che …»); she wants to withdraw and claim double the caparra under
1385 c.c. (the frame's verified domanda). 1453 and 2043 c.c. are fitted remedies (2043 c.c. because the verified frame
is contractual). The faked analysis first offers 1453 c.c. as the line -- rejected (a remedy; the frame's 1385 c.c.
missing) and re-asked once with both reasons; the second reply puts the line on 1385 c.c., one element «provato» on
the counterparty's assertion only (the code downgrades it), one law statement on 1385 c.c. citing its period.

Expected: two analysis calls; §1 «Linea principale: art. 1385 c.c. — …»; no line on 1453 or 2043 c.c. anywhere; §4's
table only for 1385 c.c., the counterparty's assertion «contestato (allegazione di Ettore Bassi)», 1453 and 2043 c.c.
named among the remedies with no table and no count; every «x su y» on 1385 c.c.; the Q&A prose stating what 1453 and
2043 c.c. provide with no verified counterpart removed (qa_sentences_dropped), the verified 1385 c.c. statement kept;
the old engine logged «legacy_plan» and printed nowhere. With FIXTURE_79_15_0_OUT set, the report is written there.
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

CASE_ID = "v79150_civil"
CLIENT = "Anna Corsi"
BASSI = ("Ettore Bassi sostiene di aver venduto l'appartamento a un terzo perché Anna Corsi non si è presentata dal "
         "notaio")
TEXT = ("Assisto Anna Corsi. Anna Corsi ha firmato con Ettore Bassi un contratto preliminare di compravendita di un "
        "appartamento. Anna Corsi ha versato a Ettore Bassi 15.000 euro a titolo di caparra confirmatoria. "
        f"{BASSI}. Anna Corsi vuole recedere dal contratto ed esigere il doppio della caparra ai sensi dell'art. 1385 "
        "c.c. Anna Corsi può esigere il doppio della caparra? Può chiedere anche il risarcimento del danno?")
ART_1385 = ("1. Se al momento della conclusione del contratto una parte dà all'altra, a titolo di caparra, una somma di "
            "danaro o una quantità di altre cose fungibili, la caparra, in caso di adempimento, deve essere restituita "
            "o imputata alla prestazione dovuta.\n"
            "2. Se la parte che ha dato la caparra è inadempiente, l'altra può recedere dal contratto, ritenendo la "
            "caparra; se inadempiente è invece la parte che l'ha ricevuta, l'altra può recedere dal contratto ed "
            "esigere il doppio della caparra.\n"
            "3. Se però la parte che non è inadempiente preferisce domandare l'esecuzione o la risoluzione del "
            "contratto, il risarcimento del danno è regolato dalle norme generali.")
ART_1453 = ("1. Nei contratti con prestazioni corrispettive, quando uno dei contraenti non adempie le sue obbligazioni, "
            "l'altro può a sua scelta chiedere l'adempimento o la risoluzione del contratto, salvo, in ogni caso, il "
            "risarcimento del danno.\n"
            "2. La risoluzione può essere domandata anche quando il giudizio è stato promosso per ottenere "
            "l'adempimento; ma non può più chiedersi l'adempimento quando è stata domandata la risoluzione.")
ARTICLES = (("1385", "c.c.", ART_1385, "Caparra confirmatoria", "LEGAL_DOC::cc", "Codice Civile 2026"),
            ("1453", "c.c.", ART_1453, "Risolubilità del contratto per inadempimento", "LEGAL_DOC::cc",
             "Codice Civile 2026"),
            ("2043", "c.c.", F.ART_2043, "Risarcimento per fatto illecito", "LEGAL_DOC::cc", "Codice Civile 2026"))
DOCUMENTS = ({"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"},)
TITLE, ANCHOR, RUBRICA = "Caparra confirmatoria e recesso", "1385 c.c.", "Caparra confirmatoria."
FRAME_REPLY = {
    "parti": [{"nome": "Anna Corsi", "ruolo_nel_fatto": "promissaria acquirente", "lato": "cliente"},
              {"nome": "Ettore Bassi", "ruolo_nel_fatto": "promittente venditore", "lato": "controparte"}],
    "cliente": {"nome": "Anna Corsi", "perche": "«Assisto Anna Corsi»"},
    "ruolo": {"valore": "attore", "perche": "«Anna Corsi vuole recedere dal contratto ed esigere il doppio della "
                                            "caparra»"},
    "rito": {"valore": "civile", "perche": "«un contratto preliminare di compravendita di un appartamento»"},
    "stadio": None,
    "addebito_o_domanda": {"articolo": "1385 c.c.", "descrizione": "recesso e doppio della caparra",
                           "frase": "«esigere il doppio della caparra ai sensi dell'art. 1385 c.c.»"},
    "istituto": {"testo": "caparra confirmatoria", "frase": "«a titolo di caparra confirmatoria»"},
    "quesiti": ["Anna Corsi può esigere il doppio della caparra?", "Può chiedere anche il risarcimento del danno?"]}


def _sid(facts, start):
    if not isinstance(facts, dict):
        facts = {i: t for i, t in re.findall(r"^(S\d+) \[[^\]]+\] (.*)$", facts, re.MULTILINE)}
    return next(i for i, t in facts.items() if t.startswith(start))


def fit_reply(messages, kwargs):
    """The fit faked: the three norms «si», each citing its first period."""
    human = messages[-1].content
    paid = _sid(human, "Anna Corsi ha versato")
    signed = _sid(human, "Anna Corsi ha firmato")
    items = {"1385 c.c.": F._fit_item("1385 c.c.", "si", f"Anna Corsi ha versato la somma ({paid}): «a titolo di "
                                                       "caparra, una somma di danaro»."),
             "1453 c.c.": F._fit_item("1453 c.c.", "si", f"Il contratto ({signed}) è un contratto «con prestazioni "
                                                       "corrispettive».", ambito="rimedio"),
             "2043 c.c.": F._fit_item("2043 c.c.", "si", "«Qualunque fatto doloso o colposo» obbliga «a risarcire il "
                                                       "danno».", ambito="rimedio")}
    labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
    return json.dumps({"giudizi": [items[x] for x in labels if x in items]}, ensure_ascii=False)


def link(topic, reqs, sents):
    """The element link faked: the caparra on the payment, the receiver's breach on the counterparty's own account."""
    sid = {s["text"]: s["id"] for s in sents}
    paid = next(i for t, i in sid.items() if t.startswith("Anna Corsi ha versato"))
    bassi = next(i for t, i in sid.items() if t.startswith("Ettore Bassi sostiene"))
    out = {}
    for label, r in reqs:
        text = str(r.get("text") or "").lower()
        if label.startswith("1385") and "a titolo di caparra" in text:
            out[r["id"]] = {"sostiene": [paid], "contraddice": [], "manca": ""}
        elif label.startswith("1385") and "ricevuta" in text:
            out[r["id"]] = {"sostiene": [bassi], "contraddice": [], "manca": ""}
    return out


def _line(parsed, article, word, tesi, facts):
    norm = parsed["norms"][article]
    element = next((e for e in norm["elements"] if word in e.lower()), norm["elements"][0])
    return {"articolo": article, "elemento_in_gioco": H.eid(parsed, article, element), "tesi": tesi,
            "cita_fatti": facts,
            "cita_norma": [norm["periods"][0]]}


def analysis_reply(parsed, attempt):
    """The analysis faked: first the line on the remedy 1453 c.c. (rejected, the frame's 1385 c.c. missing); on the
    re-ask the line on 1385 c.c. -- the payment «provato» on its fact, the receiver's breach «provato» on the
    counterparty's own assertion only (downgraded by the code), the rest open; one law statement on 1385 c.c."""
    facts = parsed["facts"]
    paid, bassi = _sid(facts, "Anna Corsi ha versato"), _sid(facts, "Ettore Bassi sostiene")
    if attempt == 1:
        return {"linea_principale": _line(parsed, "1453 c.c.", "adempie", "Anna Corsi può chiedere la risoluzione "
                                                                           "del preliminare.", [bassi]),
                "linea_subordinata": None, "elementi": [], "affermazioni_di_diritto": []}
    norm = parsed["norms"]["1385 c.c."]

    def ref(e):                 # v79.15.1: the element cited by its id
        return H.eid(parsed, "1385 c.c.", e)
    elementi = []
    for e in norm["elements"]:
        low = e.lower()
        if "a titolo di caparra" in low:
            elementi.append({"articolo": "1385 c.c.", "elemento": ref(e), "esito": "provato",
                             "cita_fatti": [paid],
                             "parte_che_afferma": None})
        elif "ricevuta" in low:
            elementi.append({"articolo": "1385 c.c.", "elemento": ref(e), "esito": "provato",
                             "cita_fatti": [bassi],
                             "parte_che_afferma": None})
        else:
            elementi.append({"articolo": "1385 c.c.", "elemento": ref(e), "esito": "da_provare",
                             "cita_fatti": [],
                             "parte_che_afferma": None})
    period2 = norm["periods"][1] if len(norm["periods"]) > 1 else norm["periods"][0]
    return {"linea_principale": _line(parsed, "1385 c.c.", "ricevuta", "Anna Corsi può recedere ed esigere il doppio "
                                      "della caparra se prova l'inadempimento di Ettore Bassi.", [paid, bassi]),
            "linea_subordinata": None, "elementi": elementi,
            "affermazioni_di_diritto": [
                {"testo": "Se inadempiente è la parte che ha ricevuto la caparra, l'altra può recedere dal contratto "
                          "ed esigere il doppio della caparra.", "cita_norma": [period2]}]}


WRITER = {
    "1385 c.c.": "L'art. 1385 c.c. prevede che, se inadempiente è la parte che ha ricevuto la caparra, l'altra può recedere "
                 "dal contratto ed esigere il doppio della caparra.",
    # grounded in the article's words (Block B keeps them) but with no verified affermazione_di_diritto: removed
    "1453 c.c.": "L'art. 1453 c.c. prevede che, quando uno dei contraenti non adempie le sue obbligazioni, l'altro può a "
                 "sua scelta chiedere l'adempimento o la risoluzione del contratto.",
    "2043 c.c.": "L'art. 2043 c.c. prevede che qualunque fatto doloso o colposo, che cagiona ad altri un danno ingiusto, "
                 "obbliga colui che ha commesso il fatto a risarcire il danno."}


# the writer's answer to a question: one statement of 1385 c.c. the analysis verified (kept), one grounded in the
# article's third comma but with no verified counterpart (removed: qa_sentences_dropped)
QA_KEPT = ("L'art. 1385 c.c. consente alla parte non inadempiente di recedere dal contratto ed esigere il doppio della "
           "caparra ricevuta dall'altra parte.")
QA_DROPPED = ("L'art. 1385 c.c. prevede che, se la parte non inadempiente preferisce domandare l'esecuzione o la "
              "risoluzione del contratto, il risarcimento del danno è regolato dalle norme generali.")


def writer_line(label):
    return WRITER[label]


def writer_question(title):
    return f"{QA_KEPT} {QA_DROPPED}"


case = sys.modules[__name__]


def test_the_civil_fixture_end_to_end(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, case)
    state, answer, by = run["state"], run["answer"], run["by"]
    c = state["block_counters"]
    target = os.environ.get("FIXTURE_79_15_0_OUT")
    if target:
        H.write_report(target, "v79.15.0 synthetic CIVIL fixture (tests/test_v79_15_0_fixture_civil.py)", run, "w")
    assert [x for x in run["failed"] if x.split(":")[0] in ("analysis_step", "applicability_fit", "legacy_plan",
                                                             "lawyer_method_compose", "lawyer_method_plan")] == []
    # the remedy line rejected, re-asked once with both reasons, then the line on the claim article
    step = by["analysis_step"][0][0]
    assert step["status"] == "ok" and c["analysis_calls"] == 2 == len(run["model"].calls) and c["analysis_reasked"] == 1
    assert c["analysis_line_rejected"] == 1 and c["analysis_frame_charge_missing"] == 0
    assert c["analysis_line_undetermined"] == 0
    reask = run["model"].calls[1]["last"]
    assert "rimedio o parte generale" in reask and "art. 1385 c.c." in reask
    assert [x["articolo"] for x in step["lines"]] == ["1385 c.c."]
    # the counterparty's own assertion never «provato»
    assert c["analysis_provato_downgraded"] == 1
    sintesi = H.section(answer, "sintesi")
    # v79.15.3 (item 1): the tesi is the code's -- a claimant (ruolo attore), the element in play contested
    assert sintesi.startswith("Linea principale: art. 1385 c.c. — «inadempiente è invece la parte che l'ha ricevuta» "
                              "è da provare.")
    assert "Linea subordinata" not in sintesi
    for remedy in ("1453", "2043"):
        assert not any(remedy in x for x in answer.splitlines() if "Linea principale" in x or "Linea subordinata" in x)
    # §4: one table, 1385 c.c.; the remedies named, no table, no count
    col = H.esito_column(answer)
    assert col and all(x.startswith("1385 c.c.") for x in col), col
    assert any("| contestato (allegazione di Ettore Bassi)" in x for x in col), col
    assert not any(x.endswith("| provato") and "ricevuta" in x for x in col), col
    norme = H.section(answer, "norme")
    assert "**Rimedi e parte generale collegati alla linea**: art. 1453 c.c." in norme and "art. 2043 c.c." in norme
    assert "**Art. 1453 c.c.**" not in norme and "**Art. 2043 c.c.**" not in norme
    counts = [x for x in answer.splitlines() if re.search(r"\b\d+ su \d+\b", x)]
    assert counts and all("1385 c.c." in x and "1453" not in x and "2043" not in x for x in counts), counts
    # §6 / §9 the verified line only
    sol = H.section(answer, "soluzioni")
    assert "art. 1385 c.c." in sol and "1453" not in sol and "2043" not in sol
    assert H.section(answer, "conclusioni").startswith("**Esiti dell'art. 1385 c.c.**")      # v79.16.0 (D2): the counts open §9
    assert H.section(answer, "conclusioni").splitlines()[1].startswith("**Linea principale**: art. 1385 c.c.")
    # the Q&A: the unverified statements of 1453 and 2043 c.c. removed, the verified 1385 c.c. one printed
    # v79.15.1 (item 4): the unverified 1385 c.c. statement is replaced by its verified one, no longer dropped
    assert c["qa_sentences_dropped"] + c["qa_sentences_replaced"] >= 1
    assert WRITER["1453 c.c."] not in answer and WRITER["2043 c.c."] not in answer and QA_DROPPED not in answer
    assert QA_KEPT in H.section(answer, "risposte") or QA_KEPT in answer.split("**1. ")[0]
    assert "**Affermazioni di diritto verificate**" in answer
    assert "l'altra può recedere dal contratto ed esigere il doppio della caparra (art. 1385 c.c.)." in answer
    # the old engine: logged, compared, printed nowhere
    legacy = by["legacy_plan"][0][0]
    assert legacy["rendered"] is False
    if legacy["linea_principale"] and legacy["linea_principale"]["tesi"]:
        assert legacy["linea_principale"]["tesi"] not in answer
