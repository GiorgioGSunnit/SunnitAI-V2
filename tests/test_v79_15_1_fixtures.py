"""v79.15.1 build A -- the synthetic fixtures end to end through the real entry (intake.request_intake ->
interactive_flow.single_shot -> write_answer -> _write_answer) with ANALYSIS_STEP, CASE_FRAME, APPLICABILITY_FIT and
LAWYER_METHOD on; every model faked at its source, every other call and every outbound socket refused
(tests/_fixture_79_15_0.py). Invented names and facts; statutory article texts.

1. the 79.15.0 CIVIL fixture (a claimant on 1385 c.c.; the first reply's line on the remedy 1453 c.c.): the elements
   cited by id; the re-ask lists the allowed articles with their E-ids and the claimant line; §7 prints the
   counterparty's allegation as an objection, never «Il materiale non indica obiezioni»
2. the 79.15.0 PENAL fixture (an indagato; the Procura's and the company's assertions): one objection per allegation
3. a NEW PENAL fixture whose charge has no article number in the text («indagato per appropriazione indebita»): the
   frame verifies the descrizione, no article; the model's line fails twice (an element written as text, then the
   general-part 62-bis c.p.) -> the rubrica fallback: 646 c.p. «Appropriazione indebita» leads, every element «da
   provare», a §4 table, a §6 row -- never «da determinare»
Build B (item 4 -- the prose replaced, not deleted -- and the fixtures):
4. the CIVIL fixture with prose to review (CIVIL_QA: 1439 c.c. among the candidates, removed by the fit's verified
   «no»):
   in the Q&A a 1385 c.c. rule with no verified twin replaced by the verified 1385 statement, a 1453 c.c. rule with none
   dropped with its «Inoltre, …», the removed 1439 c.c. sentence dropped with its «Pertanto, …», a 1218 c.c. rule (no
   fitted norm, never named) dropped; §4's 1453 / 2043 c.c. rules dropped; §7's written objection stating an
   unverified 1385 c.c. rule dropped, never replaced by a statement of law (review), so the objection leaves §7
5. (a) the 79.15.0-style PENAL case whose two replies are both rejected (TRUFFA: the truffa text of
   test_v79_14_2_fixture): the 79.14 frame carries no article and no descrizione, the older frame's
   central_issue_candidates «Truffa» -> 640 c.p. by rubrica; the Q&A's 1439 c.c. sentence (the evidence's Q3) and its
   follower dropped (and «l'art. 1439» with no code, review), a 641 c.p. rule (no fitted norm) dropped, the 640 c.p.
   rule replaced
6. a CLAIMANT civil case (APPALTO, 1667 c.c., the client the committente; the lawyer's reviewed schema gives the
   elements' kinds): the first line on the acceptance («negazione») refused, re-asked with the constitutive E-id; the
   line on the garanzia, contested by the company's allegation -> §7's objection; the exceptions raised -> objections
v79.15.2 (hotfix): every faked analysis reply gives «articolo» in the id form the prompt sends (H.id_form: a line's
«1385cc» / «646cp» / «62biscp», an element verdict's «A1385cc») -- the server's model's form, which 79.15.1 rejected
(«articolo non tra le norme applicabili: 640cp»); the logged articolo is the display label; every run records
analysis_fallback_skipped_reason.
With FIXTURE_79_15_1_OUT set, the reports are written there.
"""

import json
import os
import re
import sys
import types
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import _fixture_79_15_0 as H  # noqa: E402
from src.rag import analysis_step as A  # noqa: E402
import test_v79_14_2_fixture as F  # noqa: E402
import test_v79_15_0_fixture_civil as CIVIL  # noqa: E402
import test_v79_15_0_fixture_penal as PENAL  # noqa: E402

NO_OBJECTIONS = "Il materiale non indica obiezioni della controparte sui fatti narrati."


def _report(run, title, mode):
    target = os.environ.get("FIXTURE_79_15_1_OUT")
    if target:
        H.write_report(target, title, run, mode)


# --- 3. the new penal case: the charge named, no article number ------------------------------------------------------

NA_CLIENT = "Luca Fontana"
NA_PROCURA = ("Secondo la Procura, Luca Fontana ha trasferito 6.000 euro dal conto dell'associazione al proprio conto "
              "personale")
NA_TEXT = ("Assisto Luca Fontana, indagato per appropriazione indebita. Luca Fontana era il tesoriere "
           "dell'associazione sportiva Aurora e gestiva il suo conto corrente. "
           f"{NA_PROCURA}. Luca Fontana nega di aver trasferito il denaro per sé. Il pubblico ministero procede per "
           "appropriazione indebita e le indagini preliminari sono in corso. Quali rischi corre Luca Fontana?")
NA_FRAME = {
    "parti": [{"nome": "Luca Fontana", "ruolo_nel_fatto": "tesoriere", "lato": "cliente"},
              {"nome": "associazione sportiva Aurora", "ruolo_nel_fatto": "associazione", "lato": "controparte"}],
    "cliente": {"nome": "Luca Fontana", "perche": "«Assisto Luca Fontana, indagato per appropriazione indebita»"},
    "ruolo": {"valore": "indagato", "perche": "«indagato per appropriazione indebita»"},
    "rito": {"valore": "penale", "perche": "«Il pubblico ministero procede per appropriazione indebita»"},
    "stadio": {"valore": "indagini", "perche": "«le indagini preliminari sono in corso»"},
    "addebito_o_domanda": {"articolo": None, "descrizione": "appropriazione indebita",
                           "frase": "«procede per appropriazione indebita»"},
    "istituto": None,
    "quesiti": ["Quali rischi corre Luca Fontana?"]}


def _na_fit(messages, kwargs):
    human = messages[-1].content
    sid = PENAL._sid(human, "Luca Fontana era il tesoriere")
    items = {"646 c.p.": F._fit_item("646 c.p.", "si", f"Luca Fontana gestiva il conto ({sid}): «di cui abbia, a "
                                                     "qualsiasi titolo, il possesso»."),
             "62-bis c.p.": F._fit_item("62-bis c.p.", "si", "Il giudice «può prendere in considerazione altre "
                                                           "circostanze diverse».", ambito="circostanza")}
    labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
    return json.dumps({"giudizi": [items[x] for x in labels if x in items]}, ensure_ascii=False)


def _na_link(topic, reqs, sents):
    sid = {s["text"]: s["id"] for s in sents}
    procura = next(i for t, i in sid.items() if t.startswith("Secondo la Procura"))
    out = {}
    for label, r in reqs:
        if label.startswith("646") and "appropria" in str(r.get("text") or "").lower():
            out[r["id"]] = {"sostiene": [procura], "contraddice": [], "manca": ""}
    return out


def _na_analysis(parsed, attempt):
    """The analysis faked: first the line on 646 c.p. with the element written as text (no id: rejected), then on the
    general-part 62-bis c.p. (rejected) -- both lines fail, the re-ask spent."""
    facts = parsed["facts"]
    procura = PENAL._sid(facts, "Secondo la Procura")
    if attempt == 1:
        el = parsed["norms"]["646 c.p."]["elements"][0]
        return {"linea_principale": {"articolo": "646 c.p.", "elemento_in_gioco": el, "tesi": "Il trasferimento non "
                                     "è provato.", "cita_fatti": [procura],
                                     "cita_norma": [parsed["norms"]["646 c.p."]["periods"][0]]},
                "linea_subordinata": None, "elementi": [], "affermazioni_di_diritto": []}
    return {"linea_principale": {"articolo": "62-bis c.p.", "elemento_in_gioco": "A62biscp:E1",
                                 "tesi": "Le attenuanti generiche.", "cita_fatti": [procura], "cita_norma": []},
            "linea_subordinata": None, "elementi": [], "affermazioni_di_diritto": []}


NOARTICLE = types.SimpleNamespace(
    CASE_ID="v79151_penal_noarticle", CLIENT=NA_CLIENT, TEXT=NA_TEXT, FRAME_REPLY=NA_FRAME,
    ARTICLES=(("646", "c.p.", F.ART_646, "Appropriazione indebita", "LEGAL_DOC::cp", "Codice Penale 2026"),
              ("62-bis", "c.p.", PENAL.ART_62BIS, "Circostanze attenuanti generiche", "LEGAL_DOC::cp",
               "Codice Penale 2026")),
    DOCUMENTS=({"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"},),
    TITLE="Ammanchi dal conto dell'associazione", ANCHOR="646 c.p.", RUBRICA="Appropriazione indebita.",
    fit_reply=_na_fit, link=_na_link, analysis_reply=H.id_form(_na_analysis), writer_line=PENAL.writer_line)


# --- v79.15.1 build B (item 4): the CIVIL fixture with the Q&A, §4 and §7 prose to review -----------------------------
# The 79.15.0 caparra case, plus 1439 c.c. (Dolo) among the candidates -- the fit's verified «no» removes it -- and the
# writer's prose: in the Q&A a 1385 c.c. rule with no verified counterpart (replaced by the verified 1385 statement), a
# 1453 c.c. rule with none for its article (dropped) and its anaphoric follower («Inoltre, …»: dropped with it), a
# sentence on the removed 1439 c.c. and its follower («Pertanto, …»), a rule of 1218 c.c. -- no fitted norm, never
# named by the user; in §7 the thesis call's objection stating a 1385 c.c. rule the analysis did not verify.

QA_REPLACED = CIVIL.QA_DROPPED          # 1385 c.c., no verified twin: replaced by the verified 1385 statement
QA_FOLLOW = "Ai sensi del primo comma, la caparra versata resta imputata alla prestazione dovuta."
QA_1218 = ("L'art. 1218 c.c. obbliga il debitore che non esegue esattamente la prestazione dovuta al risarcimento del "
           "danno.")
QA_1453 = ("L'art. 1453 c.c. prevede che il contraente non inadempiente possa chiedere la risoluzione del contratto, "
           "salvo il risarcimento del danno.")
QA_1453_FOLLOW = "Inoltre, il risarcimento spetta in ogni caso alla parte non inadempiente."
QA_1439 = "Il preliminare sarebbe anche annullabile per dolo di Ettore Bassi ai sensi dell'art. 1439 c.c."
QA_1439_FOLLOW = "Pertanto, Anna Corsi potrebbe chiedere l'annullamento del preliminare."
OBJ_RULE = ("La controparte sosterrà che l'art. 1385 c.c. prevede la perdita della caparra per chi non si presenta "
            "dal notaio.")


def _civil_qa_fit(messages, kwargs):
    got = json.loads(CIVIL.fit_reply(messages, kwargs))
    human = messages[-1].content
    if "ARTICOLO: 1439 c.c." in human:
        signed = CIVIL._sid(human, "Anna Corsi ha firmato")
        got["giudizi"].append(F._fit_item("1439 c.c.", "no", f"«Anna Corsi ha firmato con Ettore Bassi un contratto "
                                                            f"preliminare di compravendita di un appartamento» "
                                                            f"({signed}).", ambito="rimedio", cf=[signed]))
    return json.dumps(got, ensure_ascii=False)


def _civil_qa_question(title):
    if "risarcimento" in str(title).lower():
        return f"{QA_1453} {QA_1453_FOLLOW} {QA_1439} {QA_1439_FOLLOW} {CIVIL.QA_KEPT}"
    return f"{QA_REPLACED} {QA_FOLLOW} {QA_1218} {CIVIL.QA_KEPT}"


def _civil_qa_writer(label):
    if label.startswith("1439"):
        return "L'art. 1439 c.c. prevede che il dolo è causa di annullamento del contratto."
    return CIVIL.writer_line(label)


CIVIL_QA = types.SimpleNamespace(**{k: getattr(CIVIL, k) for k in dir(CIVIL) if not k.startswith("__")})
CIVIL_QA.CASE_ID = "v79151_civil_qa"
CIVIL_QA.ARTICLES = CIVIL.ARTICLES + (("1439", "c.c.", F.ART_1439, "Dolo", "LEGAL_DOC::cc", "Codice Civile 2026"),)
CIVIL_QA.fit_reply = _civil_qa_fit
CIVIL_QA.analysis_reply = H.id_form(CIVIL.analysis_reply)         # v79.15.2: «1453cc», «1385cc», «A1385cc»
CIVIL_QA.writer_question = _civil_qa_question
CIVIL_QA.writer_line = _civil_qa_writer
CIVIL_QA.objections = [{"obiezione": OBJ_RULE, "risposta": "Va provato che Ettore Bassi abbia ricevuto la caparra."}]


# --- v79.15.1 build B (a): the 79.15.0-style PENAL case whose two replies are both rejected -------------------------
# The truffa case of the evidence (test_v79_14_2_fixture's synthetic text: 640 c.p. named, 1439 c.c. removed by the
# fit's verified «no», 2043 c.c. excluded by the binding): the 79.14 frame verifies the charge's sentence but carries no
# article and no descrizione (the frame model left them out), so no frame article can lead; the older frame
# (CASE_FRAME_ENABLED's) names «Truffa» among its central_issue_candidates -- every word in the case, so it counts. The
# model puts the line on 640 c.p. twice: the element written as text, then an E-id 640 c.p. does not have -- both
# rejected -> the rubrica fallback: «Truffa» is 640 c.p.'s rubrica. The Q&A (the evidence's Q3) states 1439 c.c.

TR_QUESTION = "Il contratto di vendita dell'orologio è annullabile?"
TR_TEXT = f"{F.TEXT} {TR_QUESTION}"
TR_FRAME = json.loads(json.dumps(F.FRAME_REPLY))
TR_FRAME["quesiti"] = TR_FRAME["quesiti"] + [TR_QUESTION]
TR_FRAME["addebito_o_domanda"] = {"articolo": None, "descrizione": None,
                                  "frase": "«procede per truffa ai sensi dell'art. 640 c.p.»"}
TR_QA_640 = "L'art. 640 c.p. prevede la reclusione per chi vende un bene dichiarandone falsamente le qualità."
TR_QA_1439 = "Il contratto di vendita dell'orologio è annullabile per dolo ai sensi dell'art. 1439 c.c."
TR_QA_1439_FOLLOW = "Pertanto, Carla Tasso può chiedere l'annullamento della vendita."
TR_QA_641 = "L'art. 641 c.p. punisce anche chi contrae un'obbligazione dissimulando il proprio stato d'insolvenza."
TR_QA_1439_BARE = "Secondo l'art. 1439 il contratto concluso con raggiri è annullabile."   # (review) no code: 1439 c.c.


def _tr_link(topic, reqs, sents):
    return {}


def _tr_analysis(parsed, attempt):
    facts = parsed["facts"]
    sale = PENAL._sid(facts, "Bruno Gelso ha venduto")
    norm = parsed["norms"]["640 c.p."]
    if attempt == 1:
        return {"linea_principale": {"articolo": "640 c.p.", "elemento_in_gioco": norm["elements"][0],
                                     "tesi": "Gli artifizi non sono provati.", "cita_fatti": [sale],
                                     "cita_norma": [norm["periods"][0]]},
                "linea_subordinata": None, "elementi": [], "affermazioni_di_diritto": []}
    return {"linea_principale": {"articolo": "640 c.p.", "elemento_in_gioco": "A640cp:E9",
                                 "tesi": "Gli artifizi non sono provati.", "cita_fatti": [sale],
                                 "cita_norma": [norm["periods"][0]]},
            "linea_subordinata": None, "elementi": [],
            "affermazioni_di_diritto": [
                {"testo": "L'art. 640 c.p. punisce chi, con artifizi o raggiri, inducendo taluno in errore, procura a "
                          "sé o ad altri un ingiusto profitto con altrui danno.", "cita_norma": [norm["periods"][0]]}]}


def _tr_writer(label):
    return f"L'art. {label} riguarda i fatti narrati."


def _tr_question(title):
    return f"{TR_QA_640} {TR_QA_1439} {TR_QA_1439_FOLLOW} {TR_QA_641} {TR_QA_1439_BARE}"


TRUFFA = types.SimpleNamespace(
    CASE_ID="v79151_penal_truffa", CLIENT="Bruno Gelso", TEXT=TR_TEXT, FRAME_REPLY=TR_FRAME,
    OLDER_FRAME={"central_issue_candidates": [{"nome": "Truffa"}]},
    ARTICLES=F.ARTICLES, DOCUMENTS=({"id": "LEGAL_DOC::cp", "name": "Codice Penale 2026"},
                                    {"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"}),
    TITLE="Truffa nella vendita dell'orologio", ANCHOR="640 c.p.", RUBRICA="Truffa.",
    fit_reply=F._fit_reply, link=_tr_link, analysis_reply=H.id_form(_tr_analysis), writer_line=_tr_writer,
    writer_question=_tr_question)


# --- v79.15.1 build B: a CLAIMANT civil case (appalto: the client the committente) -----------------------------------
# Marta Galli, committente, wants the garanzia of 1667 c.c.; the company alleges the work was done properly and that she
# accepted it. The frame verifies no procedural role -- her ruolo_nel_fatto «committente» makes her the claimant. The
# model first puts the line on 1667's exception clause («il committente ha accettato l'opera»): refused, re-asked with
# the constitutive E-ids; then on the garanzia element, contested by the company's allegation -> the allegation is
# §7's objection, the exception raised in the first reply another.

AP_CLIENT = "Marta Galli"
AP_EDIL = ("La Edilnova S.r.l. sostiene che i lavori sono stati eseguiti a regola d'arte e che Marta Galli ha "
           "accettato "
           "l'opera senza riserve")
AP_TEXT = ("Assisto Marta Galli. Marta Galli, committente, ha affidato alla Edilnova S.r.l., appaltatrice, il "
           "rifacimento del bagno con un contratto di appalto. Dopo la consegna Marta Galli ha scoperto infiltrazioni "
           f"d'acqua sotto il pavimento. {AP_EDIL}. Marta Galli ha denunziato i vizi alla Edilnova S.r.l. con una "
           "raccomandata dieci giorni dopo la "
           "scoperta. Marta Galli vuole far valere la garanzia per i vizi ai sensi dell'art. 1667 c.c. Marta Galli può "
           "ottenere la garanzia per i vizi?")
ART_1667 = ("1. L'appaltatore è tenuto alla garanzia per le difformità e i vizi dell'opera. La garanzia non è dovuta "
            "se "
            "il committente ha accettato l'opera e le difformità o i vizi erano da lui conosciuti o erano "
            "riconoscibili, purché, in questo caso, non siano stati in mala fede taciuti dall'appaltatore.\n"
            "2. Il committente deve, a pena di decadenza, denunziare all'appaltatore le difformità o i vizi entro "
            "sessanta giorni dalla scoperta. La denunzia non è necessaria se l'appaltatore ha riconosciuto le "
            "difformità o i vizi o se li ha occultati.\n"
            "3. L'azione contro l'appaltatore si prescrive in due anni dal giorno della consegna dell'opera. Il "
            "committente convenuto per il pagamento può sempre far valere la garanzia, purché le difformità o i vizi "
            "siano stati denunziati entro sessanta giorni dalla scoperta e prima che siano decorsi i due anni dalla "
            "consegna.")
AP_FRAME = {
    "parti": [{"nome": "Marta Galli", "ruolo_nel_fatto": "committente", "lato": "cliente"},
              {"nome": "Edilnova S.r.l.", "ruolo_nel_fatto": "appaltatrice", "lato": "controparte"}],
    "cliente": {"nome": "Marta Galli", "perche": "«Assisto Marta Galli»"},
    "ruolo": None,
    "rito": {"valore": "civile", "perche": "«con un contratto di appalto»"},
    "stadio": None,
    "addebito_o_domanda": {"articolo": "1667 c.c.", "descrizione": "garanzia per i vizi",
                           "frase": "«far valere la garanzia per i vizi ai sensi dell'art. 1667 c.c.»"},
    "istituto": {"testo": "contratto di appalto", "frase": "«con un contratto di appalto»"},
    "quesiti": ["Marta Galli può ottenere la garanzia per i vizi?"]}
def _schema_el(eid, testo, kind):
    return {"id": eid, "testo": testo, "kind": kind, "kind_note": None, "comma": "1", "branches": [],
            "connective": None, "parti": [], "grounding": "explicit", "fonte": {"comma": "1", "testo": testo},
            "spiegazione": None}


# the lawyer's reviewed schema of 1667 c.c. (element_schemas' overlay): the garanzia a «condizione» (constitutive), the
# acceptance a «negazione» and the denunzia a «termine» -- the schema's own kinds decide which is an exception
AP_SCHEMA = {"tipo": ["claim-conferring"], "destinatario": [{"ruolo": "altro", "condizione": None}],
             "legittimato": [{"ruolo": "creditore", "condizione": None}], "soggetto_passivo": [],
             "rapporto": "garanzia per i vizi dell'opera appaltata",
             "elementi": [_schema_el("E1", "L'appaltatore è tenuto alla garanzia per le difformità e i vizi dell'opera",
                                     "condizione"),
                          _schema_el("E2", "il committente ha accettato l'opera e le difformità o i vizi erano da lui "
                                           "conosciuti o erano riconoscibili", "negazione"),
                          dict(_schema_el("E3", "denunziare all'appaltatore le difformità o i vizi entro sessanta "
                                                "giorni dalla scoperta", "termine"),
                               comma="2", fonte={"comma": "2", "testo": "denunziare all'appaltatore le difformità o i "
                                                                       "vizi entro sessanta giorni dalla scoperta"})],
             "logic": {"op": "all_of", "args": [{"ref": "E1"}, {"ref": "E2"}, {"ref": "E3"}]},
             "eccezioni": [], "conseguenza": None, "commi_speciali": [], "collegati": []}
def _ap_fit(messages, kwargs):
    human = messages[-1].content
    found = CIVIL._sid(human, "Dopo la consegna")
    items = {"1667 c.c.": F._fit_item("1667 c.c.", "si", f"Marta Galli ha scoperto infiltrazioni ({found}): «la "
                                                       "garanzia per le difformità e i vizi dell'opera»."),
             "1453 c.c.": F._fit_item("1453 c.c.", "si", "Il contratto è «con prestazioni corrispettive».",
                                      ambito="rimedio")}
    labels = [line.split(": ", 1)[1] for line in human.splitlines() if line.startswith("ARTICOLO: ")]
    return json.dumps({"giudizi": [items[x] for x in labels if x in items]}, ensure_ascii=False)


def _ap_link(topic, reqs, sents):
    return {}


def _ap_pick(norm, words):
    return next(i for i, e in zip(norm["eids"], norm["elements"]) if all(w in e.lower() for w in words))


def _ap_analysis(parsed, attempt):
    facts, norm = parsed["facts"], parsed["norms"]["1667 c.c."]
    found, edil = CIVIL._sid(facts, "Dopo la consegna"), CIVIL._sid(facts, "La Edilnova S.r.l. sostiene")
    told = CIVIL._sid(facts, "Marta Galli ha denunziato")
    accepted = _ap_pick(norm, ("accettato",))
    if attempt == 1:
        return {"linea_principale": {"articolo": "1667 c.c.", "elemento_in_gioco": accepted,
                                     "tesi": "Marta Galli non ha accettato l'opera.", "cita_fatti": [edil],
                                     "cita_norma": [norm["periods"][0]]},
                "linea_subordinata": None, "elementi": [], "affermazioni_di_diritto": []}
    vizi = _ap_pick(norm, ("tenuto", "garanzia"))
    elementi = []
    for i in norm["eids"]:
        if i == vizi:
            elementi.append({"articolo": "1667 c.c.", "elemento": i, "esito": "contestato",
                             "cita_fatti": [found, edil], "parte_che_afferma": "Edilnova S.r.l."})
        elif "denunziare" in norm["elements"][norm["eids"].index(i)].lower():
            elementi.append({"articolo": "1667 c.c.", "elemento": i, "esito": "provato", "cita_fatti": [told],
                             "parte_che_afferma": None})
        else:
            elementi.append({"articolo": "1667 c.c.", "elemento": i, "esito": "da_provare", "cita_fatti": [],
                             "parte_che_afferma": None})
    return {"linea_principale": {"articolo": "1667 c.c.", "elemento_in_gioco": vizi,
                                 "tesi": "La Edilnova S.r.l. deve la garanzia per le infiltrazioni scoperte dopo la "
                                         "consegna.", "cita_fatti": [found],
                                 "cita_norma": [norm["periods"][0]]},
            "linea_subordinata": None, "elementi": elementi,
            "affermazioni_di_diritto": [
                {"testo": "L'appaltatore è tenuto alla garanzia per le difformità e i vizi dell'opera.",
                 "cita_norma": [norm["periods"][0]]}]}


def _ap_writer(label):
    if label.startswith("1667"):
        return ("L'art. 1667 c.c. prevede che l'appaltatore è tenuto alla garanzia per le difformità e i vizi "
                "dell'opera.")
    return "L'art. 1453 c.c. riguarda i fatti narrati."


APPALTO = types.SimpleNamespace(
    CASE_ID="v79151_civil_appalto", CLIENT=AP_CLIENT, TEXT=AP_TEXT, FRAME_REPLY=AP_FRAME,
    ARTICLES=(("1667", "c.c.", ART_1667, "Difformità e vizi dell'opera", "LEGAL_DOC::cc", "Codice Civile 2026"),
              ("1453", "c.c.", CIVIL.ART_1453, "Risolubilità del contratto per inadempimento", "LEGAL_DOC::cc",
               "Codice Civile 2026")),
    DOCUMENTS=({"id": "LEGAL_DOC::cc", "name": "Codice Civile 2026"},),
    TITLE="Vizi del bagno rifatto", ANCHOR="1667 c.c.", RUBRICA="Difformità e vizi dell'opera.",
    fit_reply=_ap_fit, link=_ap_link, analysis_reply=H.id_form(_ap_analysis), writer_line=_ap_writer,
    ENV={"ELEMENT_SCHEMAS": "true"})
APPALTO.REVIEWED_SCHEMAS = [{"label": "1667 c.c.", "reviewed": True, "reviewer": "test", "schema_version": 1,
                             "source_hash": __import__("src.rag.element_schemas", fromlist=["x"]).text_hash(ART_1667),
                             "schema": AP_SCHEMA}]


# v79.15.2: the 79.15.0 penal fixture with its reply in the id form («646cp», «A646cp»)
PENAL_ID = types.SimpleNamespace(**{k: getattr(PENAL, k) for k in dir(PENAL) if not k.startswith("__")})
PENAL_ID.analysis_reply = H.id_form(PENAL.analysis_reply)


# --- the tests --------------------------------------------------------------------------------------------------------

def _no_failed_steps(run):
    assert [x for x in run["failed"] if x.split(":")[0] in ("analysis_step", "applicability_fit", "legacy_plan",
                                                             "lawyer_method_compose", "lawyer_method_plan")] == []


def test_the_civil_fixture_by_ids_with_the_allegation_as_its_objection(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, CIVIL_QA)
    _report(run, "v79.15.1 synthetic CIVIL fixture (the 79.15.0 caparra case + 1439 c.c. removed by the fit + prose to "
                 "review: tests/test_v79_15_1_fixtures.py CIVIL_QA)", "w")
    _no_failed_steps(run)
    c, answer = run["state"]["block_counters"], run["answer"]
    first = run["model"].calls[0]["parsed"]["norms"]["1385 c.c."]
    assert first["eids"] == [f"A1385cc:E{i}" for i in range(1, len(first["elements"]) + 1)]
    reask = run["model"].calls[1]["last"]
    assert "Articoli ammessi come linea (A<articolo>) con gli id dei loro elementi:" in reask
    assert "- A1385cc (art. 1385 c.c.): A1385cc:E1" in reask and "A1453cc (art." not in reask
    assert "A2043cc (art." not in reask and "Il cliente agisce come attore" in reask
    step = run["by"]["analysis_step"][0][0]
    assert step["lines"][0]["eid"].startswith("A1385cc:E") and step["claimant"] is True
    assert c["analysis_bad_eid"] == 0 and c["analysis_line_exception"] == 0 and c["analysis_line_fallback_rubrica"] == 0
    # v79.15.2: the id form read -- «1453cc» rejected as a remedy (logged «1453 c.c.»), «1385cc» verified
    assert [r["articolo"] for r in step["rejected"] if r.get("kind") != "record"] == ["1453 c.c."]   # v79.19.4: records aside
    assert step["rejected"][0]["why"].startswith("art. 1453 c.c.: rimedio o parte generale")
    assert step["lines"][0]["articolo"] == "1385 c.c." and c["analysis_line_rejected"] == 1
    assert c["analysis_fallback_skipped_reason"] == "line_verified" == step["fallback_skipped_reason"]
    obiezioni = H.section(answer, "obiezioni")
    # v79.15.3 (item 3): the allegation's sentence as written; the answer the status in words, the client's sentences
    assert obiezioni.startswith(f"**Obiezione 1.** {CIVIL.BASSI}.\n**Risposta.** L'elemento «inadempiente è invece la "
                                "parte che l'ha ricevuta» (art. 1385 c.c.) è da provare: «Anna Corsi ha versato a "
                                "Ettore Bassi 15.000 euro a titolo di caparra confirmatoria».")
    assert "Esito verificato" not in obiezioni and "Fatti del cliente" not in obiezioni
    assert NO_OBJECTIONS not in answer
    assert c["objections_from_allegations"] == 1 and c["objections_from_exceptions"] == 0


def test_the_penal_fixture_one_objection_per_allegation(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, PENAL_ID)
    _report(run, "v79.15.1 synthetic PENAL fixture (the 79.15.0 case: tests/test_v79_15_0_fixture_penal.py; v79.15.2: "
                 "the reply in the id form)", "a")
    _no_failed_steps(run)
    c, answer = run["state"]["block_counters"], run["answer"]
    obiezioni = H.section(answer, "obiezioni")
    # v79.15.3 (item 3): one plain form for both -- the sentence as written, the status in words (the defence: «non è
    # provato»), the client's own sentence
    assert f"**Obiezione 1.** {PENAL.PROCURA}.\n**Risposta.** L'elemento «per procurare a sé o ad altri un ingiusto " \
           "profitto» (art. 646 c.p.) non è provato: «Dario Melis nega di aver trattenuto il denaro»." in obiezioni
    assert f"**Obiezione 2.** {PENAL.SOCIETA}.\n**Risposta.** L'elemento «si appropria il denaro o la cosa mobile " \
           "altrui di cui abbia» (art. 646 c.p.) non è provato." in obiezioni
    assert "Esito verificato" not in obiezioni and "Fatti del cliente" not in obiezioni
    assert NO_OBJECTIONS not in answer
    assert c["objections_from_allegations"] == 2 and c["objections_from_exceptions"] == 0
    assert c["analysis_bad_eid"] == 0 and run["by"]["analysis_step"][0][0]["claimant"] is False
    step = run["by"]["analysis_step"][0][0]         # v79.15.2: «646cp» verified on the first reply, logged «646 c.p.»
    assert c["analysis_calls"] == 1 and [r for r in step["rejected"] if r.get("kind") != "record"] == [] \
        and step["lines"][0]["articolo"] == "646 c.p."                                      # v79.19.4: records aside
    assert c["analysis_fallback_skipped_reason"] == "line_verified"


def test_a_penal_charge_with_no_article_number_finds_its_line_by_rubrica(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, NOARTICLE)
    _report(run, "v79.15.1 synthetic PENAL fixture, the charge with no article number "
                 "(tests/test_v79_15_1_fixtures.py)", "a")
    _no_failed_steps(run)
    state, answer = run["state"], run["answer"]
    c = state["block_counters"]
    frame = state["case_frame"]["frame"]
    assert frame["addebito_o_domanda"]["articolo"] is None
    assert frame["addebito_o_domanda"]["descrizione"] == "appropriazione indebita"
    assert not re.search(r"\b646\b", NA_TEXT)
    assert c["analysis_calls"] == 2 and c["analysis_line_rejected"] == 2 and c["analysis_bad_eid"] == 1
    assert c["analysis_frame_charge_missing"] == 0
    assert c["analysis_line_fallback_rubrica"] == 1 and c["analysis_line_undetermined"] == 0
    step = run["by"]["analysis_step"][0][0]         # v79.15.2: «646cp», then «62biscp» -- the remedy by its id form
    assert [r["articolo"] for r in step["rejected"]] == ["646 c.p.", "62-bis c.p."]
    assert "rimedio o parte generale" in step["rejected"][1]["why"]
    assert c["analysis_fallback_skipped_reason"] == "" == step["fallback_skipped_reason"]
    sintesi = H.section(answer, "sintesi")
    assert sintesi == "Linea principale: art. 646 c.p. — Appropriazione indebita: elementi da provare dall'accusa."
    assert "da determinare" not in answer
    col = H.esito_column(answer)
    assert col and all(x.startswith("646 c.p.") and x.endswith("| da provare") for x in col), col
    sol = H.section(answer, "soluzioni")
    assert "| Linea principale | art. 646 c.p. | — | Appropriazione indebita: elementi da provare dall'accusa | — |" \
        in sol
    assert H.section(answer, "conclusioni").startswith("**Linea principale**: art. 646 c.p. — Appropriazione indebita")
    assert "62-bis" not in sintesi and "62-bis" not in sol
    legacy = run["by"]["legacy_plan"][0][0]
    assert legacy["rendered"] is False



def _qa(answer):
    return answer.split("**1. ")[0]


def _events(run):
    return [e for d, _ in run["by"].get("lawyer_method_prose_review") or [] for e in d["events"]]


def test_the_civil_prose_is_replaced_and_never_left_in_fragments(monkeypatch, tmp_path):
    """Item 4 on the civil fixture: replaced, not deleted; a dropped sentence takes its anaphoric follower; an excluded
    and an unfitted article's sentence dropped; §4 and §7 likewise."""
    run = H.run(monkeypatch, tmp_path, CIVIL_QA)
    _no_failed_steps(run)
    c, answer = run["state"]["block_counters"], run["answer"]
    qa = _qa(answer)
    assert [x.get("article") for x in run["state"].get("fit_removed") or []] == ["1439 c.c."]
    statement = ("Se inadempiente è la parte che ha ricevuto la caparra, l'altra può recedere dal contratto ed esigere "
                 "il doppio della caparra (art. 1385 c.c.).")
    # the 1385 c.c. rule replaced by the verified statement, its «Ai sensi del primo comma» follower kept
    second = next(x for x in qa.splitlines() if x.startswith("2. "))
    assert QA_REPLACED not in answer and f"— {statement} {QA_FOLLOW}" in second
    # the 1453 c.c. rule dropped with «Inoltre», the 1439 c.c. one with «Pertanto»; 1218 c.c. dropped; never a fragment
    for gone in (QA_1453, QA_1453_FOLLOW, QA_1439, QA_1439_FOLLOW, QA_1218):
        assert gone not in answer, gone
    first = next(x for x in qa.splitlines() if x.startswith("1. "))
    assert first.endswith(f"— {CIVIL.QA_KEPT}"), first
    assert "1439" not in answer and "1218" not in answer
    for line in qa.splitlines():
        assert "— Inoltre" not in line and "— Pertanto" not in line
    # §4: the remedies' rules with no verified statement dropped; §7: the written objection's rule dropped
    assert CIVIL.WRITER["1453 c.c."] not in answer and CIVIL.WRITER["2043 c.c."] not in answer
    obiezioni = H.section(answer, "obiezioni")
    # (review) the written objection's unverified rule is dropped, never replaced by a statement of law: the objection,
    # emptied, leaves §7 (79.14.1's rule); only the verified objection from the allegation stays
    assert OBJ_RULE not in obiezioni and "**Obiezione 2.**" not in obiezioni and statement not in obiezioni
    assert f"**Obiezione 1.** {CIVIL.BASSI}." in obiezioni         # v79.15.3 (item 3): the sentence as written
    # the counters: each sentence once
    assert c["qa_sentences_replaced"] == 1 and c["qa_sentences_dropped"] == 1     # (review: §7's objection no longer
    assert c["qa_sentences_dropped_dependent"] == 2 and c["norms_sentences_dropped"] == 2      # replaced -- dropped)
    assert c["prose_excluded_article_dropped"] == 1 and c["prose_unfitted_article_dropped"] == 1
    assert c["objections_sentences_dropped"] == 1
    kinds = sorted((e["sede"], e["esito"]) for e in _events(run))
    assert kinds == sorted([("norme", A.DROPPED), ("norme", A.DROPPED), ("obiezioni", A.DROPPED),
                            ("risposte", A.DROPPED), ("risposte", A.DEPENDENT), ("risposte", A.ARTICLE_EXCLUDED),
                            ("risposte", A.DEPENDENT), ("risposte", A.REPLACED), ("risposte", A.ARTICLE_UNFITTED)])


def test_the_truffa_case_both_replies_rejected_finds_its_line_by_the_older_frames_name(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, TRUFFA)
    _report(run, "v79.15.1 synthetic PENAL fixture (a): the 79.15.0-style truffa case, both replies rejected "
                 "(tests/test_v79_15_1_fixtures.py TRUFFA)", "a")
    _no_failed_steps(run)
    state, answer = run["state"], run["answer"]
    c = state["block_counters"]
    frame = state["case_frame"]["frame"]
    assert not frame["addebito_o_domanda"]["articolo"] and not frame["addebito_o_domanda"]["descrizione"]
    assert c["analysis_calls"] == 2 and c["analysis_line_rejected"] == 2 and c["analysis_bad_eid"] == 2
    assert c["analysis_frame_charge_missing"] == 0 and c["analysis_line_undetermined"] == 0
    assert c["analysis_line_fallback_rubrica"] == 1 and c["analysis_fallback_skipped_reason"] == ""
    line = state["analysis_step"]["analysis"]["linea_principale"]
    assert line["stato"] == A.FALLBACK_LINE and line["nome"] == "Truffa" and line["articolo"] == "640 c.p."
    assert H.section(answer, "sintesi") == "Linea principale: art. 640 c.p. — Truffa: elementi da provare dall'accusa."
    col = H.esito_column(answer)
    assert col and all(x.startswith("640 c.p.") and x.endswith("| da provare") for x in col), col
    assert "| Linea principale | art. 640 c.p. | — | Truffa: elementi da provare dall'accusa | — |" in \
        H.section(answer, "soluzioni")
    assert "da determinare" not in answer
    # the Q&A: the evidence's Q3 -- the removed 1439 c.c. and its follower gone; 641 c.p. gone; 640 c.p. replaced
    assert "1439" not in answer and TR_QA_1439_FOLLOW not in answer and "641" not in answer
    assert TR_QA_640 not in answer
    assert "L'art. 640 c.p. punisce chi, con artifizi o raggiri, inducendo taluno in errore" in _qa(answer)
    # (review) the 1439 c.c. sentence by number alone («l'art. 1439») dropped as excluded too: two per answer
    assert TR_QA_1439_BARE not in answer
    assert c["prose_excluded_article_dropped"] == 4 and c["qa_sentences_dropped_dependent"] == 2   # two answers
    assert c["prose_unfitted_article_dropped"] == 2 and c["qa_sentences_replaced"] == 2
    assert run["by"]["legacy_plan"][0][0]["rendered"] is False


def test_a_claimants_line_is_constitutive_and_the_allegation_is_the_objection(monkeypatch, tmp_path):
    run = H.run(monkeypatch, tmp_path, APPALTO)
    _report(run, "v79.15.1 synthetic CLAIMANT civil fixture: appalto, the client the committente "
                 "(tests/test_v79_15_1_fixtures.py APPALTO)", "a")
    _no_failed_steps(run)
    state, answer = run["state"], run["answer"]
    c = state["block_counters"]
    step = run["by"]["analysis_step"][0][0]
    assert step["claimant"] is True
    assert c["analysis_calls"] == 2 and c["analysis_line_exception"] == 1 and c["analysis_line_rejected"] == 1
    assert "(schema)" in step["rejected"][0]["why"]
    reask = run["model"].calls[1]["last"]
    assert "- A1667cc (art. 1667 c.c.): A1667cc:E1\n" in reask and "Il cliente agisce come attore" in reask
    assert {(x["eid"], x["fonte"]) for x in step["exceptions"]} == {("A1667cc:E2", "schema"), ("A1667cc:E3", "schema")}
    assert step["lines"][0]["eid"] == "A1667cc:E1" and step["lines"][0]["articolo"] == "1667 c.c."
    assert [r["articolo"] for r in step["rejected"] if r.get("kind") != "record"] == ["1667 c.c."]   # v79.15.2; .19.4
    assert c["analysis_fallback_skipped_reason"] == "line_verified"
    sintesi = H.section(answer, "sintesi")
    # v79.15.3 (item 1): the code's tesi -- a claimant (the committente), the element in play contested
    assert sintesi == ("Linea principale: art. 1667 c.c. — «L'appaltatore è tenuto alla garanzia per le difformità e i "
                       "vizi dell'opera» è da provare.")
    assert "La Edilnova S.r.l. deve la garanzia" not in answer                  # the model's tesi: the log only
    assert "accettato l'opera" not in sintesi
    obiezioni = H.section(answer, "obiezioni")
    assert obiezioni.startswith(f"**Obiezione 1.** {AP_EDIL}.\n**Risposta.** L'elemento «L'appaltatore è tenuto alla "
                                "garanzia per le difformità e i vizi dell'opera» (art. 1667 c.c.) è da provare: «Dopo "
                                "la consegna Marta Galli ha scoperto infiltrazioni d'acqua sotto il pavimento».")
    assert "**Obiezione 2.** Eccezione dell'art. 1667 c.c.: «il committente ha accettato l'opera" in obiezioni
    assert NO_OBJECTIONS not in answer
    assert c["objections_from_allegations"] == 1 and c["objections_from_exceptions"] == 2
