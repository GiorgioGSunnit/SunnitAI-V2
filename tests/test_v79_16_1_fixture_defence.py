"""v79.16.1 -- the synthetic CIVIL DEFENCE fixture end to end: opposizione a decreto ingiuntivo. The client, Alfa S.r.l.,
was ordered to pay Beta S.p.A. the price of goods it says were never delivered; Beta had sent a diffida ad adempiere.
1460 c.c. (eccezione d'inadempimento) and 1454 c.c. (diffida ad adempiere) are both fitted «si», verified by their ids.
Invented names and facts; statutory text of 1460 and 1454 c.c. Through the real entry (tests/_fixture_79_15_0.py) with
ANALYSIS_STEP, CASE_FRAME, APPLICABILITY_FIT, LAWYER_METHOD, FACT_RECORD and QUESTION_RETRIEVAL on.

Expected: the fit's two «si» verified by ids (fit_si_verified 2), each shown in the step's prompt as «fit: si (verificato)
— …»; the model's first pick -- 1454's procedural clause «Decorso il termine … risoluto di diritto» as the element in
play -- refused (analysis_line_procedural_rejected 1) and re-asked once with the allowed E-ids; the second reply's line
on 1460 c.c. leads (line_from_fit_si 1); its element «provato» on a pertinent fact; the gate line «retrieval_quesiti»
with source frame; §8 with no penal block (sections_rito_blocks_dropped 0 here: none was rendered).
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

CASE_ID = "v79161_defence"
CLIENT = "Alfa S.r.l."
FACT_ORDER = "Alfa S.r.l. ha ordinato a Beta S.p.A. una partita di 200 lavatrici al prezzo di 60.000 euro"
FACT_NO_DELIVERY = "Beta S.p.A. non ha adempiuto: non ha consegnato le lavatrici alla data pattuita"
FACT_REFUSAL = "Alfa S.r.l. rifiuta di adempiere al pagamento finché Beta S.p.A. non consegna la merce"
BETA = "Beta S.p.A. sostiene di aver spedito la merce e ha inviato ad Alfa S.r.l. una diffida ad adempiere entro quindici giorni"
FACT_DECREE = "Beta S.p.A. ha ottenuto un decreto ingiuntivo per 60.000 euro e Alfa S.r.l. intende proporre opposizione"
Q1 = "Alfa S.r.l. può rifiutare il pagamento finché la merce non viene consegnata?"
TEXT = (f"Assisto Alfa S.r.l., opponente. {FACT_ORDER}. {FACT_NO_DELIVERY}. {FACT_REFUSAL}. {BETA}. {FACT_DECREE}.\n"
        f"1. {Q1}")
ART_1460 = ("Nei contratti con prestazioni corrispettive, ciascuno dei contraenti può rifiutarsi di adempiere la sua "
            "obbligazione, se l'altro non adempie o non offre di adempiere contemporaneamente la propria, salvo che termini "
            "diversi per l'adempimento siano stati stabiliti dalle parti o risultino dalla natura del contratto.\n"
            "Tuttavia non può rifiutarsi l'esecuzione se, avuto riguardo alle circostanze, il rifiuto è contrario alla buona "
            "fede.")
ART_1454 = ("Alla parte inadempiente l'altra può intimare per iscritto di adempiere in un congruo termine, con dichiarazione "
            "che, decorso inutilmente detto termine, il contratto s'intenderà senz'altro risoluto.\n"
            "Il termine non può essere inferiore a quindici giorni, salvo diversa pattuizione delle parti o salvo che, per la "
            "natura del contratto o secondo gli usi, risulti congruo un termine minore.\n"
            "Decorso il termine senza che il contratto sia stato adempiuto, questo è risoluto di diritto.")
ARTICLES = (("1460", "c.c.", ART_1460, "Eccezione d'inadempimento", "LEGAL_DOC::cc", "Codice Civile 2026"),
            ("1454", "c.c.", ART_1454, "Diffida ad adempiere", "LEGAL_DOC::cc", "Codice Civile 2026"))
QUESITO_ARTICLES = ()
DOCUMENTS = ({"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"},)
TITLE, ANCHOR, RUBRICA = "Eccezione d'inadempimento e diffida", "1460 c.c.", "Eccezione d'inadempimento."
ENV = {"FACT_RECORD": "true", "QUESTION_RETRIEVAL": "true"}
FRAME_REPLY = {
    "parti": [{"nome": "Alfa S.r.l.", "ruolo_nel_fatto": "acquirente", "lato": "cliente"},
              {"nome": "Beta S.p.A.", "ruolo_nel_fatto": "venditrice", "lato": "controparte"}],
    "cliente": {"nome": "Alfa S.r.l.", "perche": "«Assisto Alfa S.r.l., opponente»"},
    "ruolo": {"valore": "convenuto", "perche": "«Alfa S.r.l. intende proporre opposizione»"},
    "rito": {"valore": "civile", "perche": "«decreto ingiuntivo»"},
    "stadio": None,
    "addebito_o_domanda": {"articolo": None, "descrizione": "pagamento del prezzo",
                           "frase": "«un decreto ingiuntivo per 60.000 euro»"},
    "istituto": {"testo": "eccezione d'inadempimento", "frase": "«rifiuta di adempiere al pagamento finché»"},
    "quesiti": [Q1]}
RECORD = [
    ("Assisto Alfa S.r.l., opponente.", "istruzione", None, None),
    (FACT_ORDER, "fatto", "accertato", None), (FACT_NO_DELIVERY, "fatto", "accertato", None),
    (FACT_REFUSAL, "fatto", "accertato", None), (BETA, "fatto", "allegato", "Beta S.p.A."),
    (FACT_DECREE, "fatto", "accertato", None), (Q1, "richiesta", None, None),
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
    return [("1460", "c.c.")]


def _sid(parsed_or_human, start):
    facts = parsed_or_human if isinstance(parsed_or_human, dict) else {
        i: t for i, t in re.findall(r"^(S\d+) \[[^\]]+\] (.*)$", parsed_or_human, re.MULTILINE)}
    return next(i for i, t in facts.items() if t.startswith(start))


def fit_reply(messages, kwargs):
    """Both norms «si», each citing its period and its fact -- verified by ids."""
    human = messages[-1].content
    refusal, diffida = _sid(human, "Alfa S.r.l. rifiuta"), _sid(human, "Beta S.p.A. sostiene")
    items = {"1460 c.c.": dict(F._fit_item("1460 c.c.", "si", f"Alfa rifiuta di adempiere in quanto Beta non ha adempiuto "
                                                            f"({refusal}).", cf=[refusal]), governa=["pretesa", "Q1"]),
             "1454 c.c.": dict(F._fit_item("1454 c.c.", "si", f"Beta ha intimato per iscritto di adempiere ({diffida}).",
                                           cf=[diffida]), governa=["pretesa"])}      # v79.17.0 (item 3): what each governs
    labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
    return json.dumps({"giudizi": [items[x] for x in labels if x in items]}, ensure_ascii=False)


def link(topic, reqs, sents):
    sid = {s["text"]: s["id"] for s in sents}
    no_delivery = next(i for t, i in sid.items() if t.startswith("Beta S.p.A. non ha adempiuto"))
    out = {}
    for label, r in reqs:
        if label.startswith("1460") and "non adempie" in str(r.get("text") or ""):
            out[r["id"]] = {"sostiene": [no_delivery], "contraddice": [], "manca": ""}
    return out


def _pick(elements, word):
    return next(e for e in elements if word in e.lower())


def analysis_reply(parsed, attempt):
    """First the line on 1454's procedural clause («Decorso il termine … risoluto di diritto»): refused; on the re-ask
    the line on 1460's «l'altro non adempie …», provato on the non-delivery (a pertinent, certain fact)."""
    facts, norms = parsed["facts"], parsed["norms"]
    no_delivery, refusal = _sid(facts, "Beta S.p.A. non ha adempiuto"), _sid(facts, "Alfa S.r.l. rifiuta")
    if attempt == 1:
        clause = _pick(norms["1454 c.c."]["elements"], "risoluto di diritto")
        return {"linea_principale": {"articolo": "1454 c.c.", "elemento_in_gioco": H.eid(parsed, "1454 c.c.", clause),
                                     "tesi": "Il contratto è risoluto di diritto.", "cita_fatti": [refusal],
                                     "cita_norma": [norms["1454 c.c."]["periods"][-1]]},
                "linea_subordinata": None, "elementi": [], "affermazioni_di_diritto": []}
    el = norms["1460 c.c."]["elements"]
    main = _pick(el, "non adempie")
    elementi = [{"articolo": "1460 c.c.", "elemento": H.eid(parsed, "1460 c.c.", main), "esito": "provato",
                 "cita_fatti": [no_delivery], "parte_che_afferma": None}]
    elementi += [{"articolo": "1460 c.c.", "elemento": H.eid(parsed, "1460 c.c.", e), "esito": "da_provare",
                  "cita_fatti": [], "parte_che_afferma": None} for e in el if e != main]
    return {"linea_principale": {"articolo": "1460 c.c.", "elemento_in_gioco": H.eid(parsed, "1460 c.c.", main),
                                 "tesi": "Alfa può rifiutare il pagamento finché Beta non consegna.",
                                 "cita_fatti": [no_delivery, refusal], "cita_norma": [norms["1460 c.c."]["periods"][0]]},
            "linea_subordinata": None, "elementi": elementi,
            "affermazioni_di_diritto": [
                {"testo": "Nei contratti con prestazioni corrispettive, ciascuno dei contraenti può rifiutarsi di adempiere "
                          "la sua obbligazione, se l'altro non adempie.", "cita_norma": [norms["1460 c.c."]["periods"][0]]}]}


WRITER = {"1460 c.c.": "L'art. 1460 c.c. consente a ciascun contraente di rifiutarsi di adempiere se l'altro non adempie.",
          "1454 c.c.": "L'art. 1454 c.c. consente alla parte non inadempiente di intimare per iscritto l'adempimento."}
QA = ("L'art. 1460 c.c. consente a ciascun contraente di rifiutarsi di adempiere la propria obbligazione se l'altro non "
      "adempie la sua. Beta S.p.A. non ha consegnato le lavatrici alla data pattuita.")


def writer_line(label):
    return WRITER[label]


def writer_question(title):
    return QA


case = sys.modules[__name__]


def test_the_civil_defence_fixture_end_to_end(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, case)
    state, answer, by = run["state"], run["answer"], run["by"]
    c = state["block_counters"]
    target = os.environ.get("FIXTURE_79_16_1_OUT")
    if target:
        H.write_report(target, "v79.16.1 synthetic CIVIL DEFENCE fixture (tests/test_v79_16_1_fixture_defence.py)", run, "w")
    assert [x for x in run["failed"] if x.split(":")[0] in ("analysis_step", "applicability_fit", "legacy_plan",
                                                             "lawyer_method_compose", "lawyer_method_plan",
                                                             "verified_context")] == []
    # item 2: both «si» verified by ids
    fit = by["applicability_fit"][0][0]["counters"]
    assert fit["fit_si"] == 2 and fit["fit_si_verified"] == 2 and fit["fit_unverified"] == 0 and fit["fit_added_content"] == 0
    assert all(j["verified"] and j["cita_fatti"] for j in state["applicability"]["judgments"])
    # item 3: the fit in the prompt, the procedural pick refused and re-asked, 1460 leads from a verified si
    first = run["model"].calls[0]["last"]
    assert "ARTICOLO: 1460 c.c.\nRubrica: Eccezione d'inadempimento\nfit: si (verificato) — Alfa rifiuta di adempiere" in first
    assert "ARTICOLO: 1454 c.c.\nRubrica: Diffida ad adempiere\nfit: si (verificato) — Beta ha intimato" in first
    step = by["analysis_step"][0][0]
    assert step["status"] == "ok" and c["analysis_calls"] == 2 and c["analysis_reasked"] == 1
    assert c["analysis_line_procedural_rejected"] == 1 and c["analysis_line_rejected"] == 1
    assert "clausola processuale, temporale o rimediale" in step["rejected"][0]["why"]
    reask = run["model"].calls[1]["last"]
    allowed = reask.split("Articoli ammessi")[1].split("Correggi")[0]
    eids_1454 = run["model"].calls[0]["parsed"]["norms"]["1454 c.c."]["eids"]
    elements_1454 = run["model"].calls[0]["parsed"]["norms"]["1454 c.c."]["elements"]
    procedural = [e for e, t in zip(eids_1454, elements_1454) if "risoluto di diritto" in t or "intimare" in t]
    assert procedural and not any(e in allowed for e in procedural) and "A1460cc:E1" in allowed
    assert [x["articolo"] for x in step["lines"]] == ["1460 c.c."] and step["line_tier"] == "fit_si"
    assert c["line_from_fit_si"] == 1 and c["line_from_unverified"] == 0 and c["line_from_fit_dipende"] == 0
    # v79.17.0 (items 2, 3): the fit was given the pretesa and Q1; both verdicts govern the pretesa, 1460 Q1 too; the
    # binding applied, no context-only norm; the material block tags them; the prompt's article blocks say «governa»
    assert c["fit_governs_pretesa"] == 2 and c["fit_governs_quesiti"] == 1 and c["fit_context_only"] == 0
    assert c["binding_pretesa_applied"] == 1 and c["binding_pretesa_excluded"] == []
    assert state["binding_pretesa"]["principale"] == ["1460 c.c.", "1454 c.c."] and state["binding_pretesa"]["quesiti"] == {"Q1": ["1460 c.c."]}
    human = run["model"].calls[0]["last"]
    assert "governa: pretesa, Q1" in human and "governa: pretesa" in human
    assert "[governa: pretesa, Q1]" in q_jobs_block(run) and "Pretesa: opposizione: «pagamento del prezzo»" in q_jobs_block(run)
    assert c["frame_pretesa_verified"] == 1 and c["frame_asked_side"] == 0
    assert H.section(answer, "sintesi").startswith("Pretesa: opposizione — «pagamento del prezzo».\n\nLinea principale: "
                                                   "art. 1460 c.c.")       # v79.17.0 (item 5): the pretesa first
    # item 4: the provato stands on a pertinent fact («non ha adempiuto» ~ «non adempie»)
    col = H.esito_column(answer)
    assert any("non adempie" in x and x.endswith("| provato") for x in col), col
    assert c["analysis_provato_no_pertinence"] == 0 and c["analysis_provenance_from_record"] == 1
    # item 1: the gate line, under its plain name, with its source
    gate = by["retrieval_quesiti"][0][0]
    assert gate["queries"] == 1 and gate["per_quesito"][0]["top"] == ["1460 c.c."]
    assert c["qa_questions_detected"] == 1
    # item 5: a civil opinion holds no penal block
    procedure = H.section(answer, "processuali")
    assert "Procedibilità e competenza" not in procedure and "Profilo penale" not in procedure
    assert c["sections_rito_blocks_dropped"] == 0
    assert by["legacy_plan"][0][0]["rendered"] is False


def q_jobs_block(run):
    """The one quesito's writer material (v79.17.0: verified_material.block)."""
    return next(j[2] for j in run["jobs"] if "?" in str(j[0]))
