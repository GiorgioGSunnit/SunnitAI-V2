"""v61: intake -- what the user wants, read once per request before any route runs (src/chatbot/api.py, ahead of the
route decision: it chooses among the routes that were there, which stay as they were).

detect(message, upload, history) -> one dict, logged as [VERBOSE:intake] and attached to the request state:

  deliverable       parere | atto | cenni | quesito | revisione_documento | calcolo | ricerca_giurisprudenza | altro
  requested_blocks  the explicit asks inside the message, in the message's order: [{kind: cenni | parere | alternative
                    | atto | quesito, subject}]; empty for a plain narrative (one ask at most, the parere's own).
                    Passed to the plan, not consumed yet (v61).
  rito              civile | lavoro | famiglia | esecuzione | penale | penale_gdp | amministrativo | tributario |
                    fallimentare | sconosciuto. Tributario and fallimentare are out of the corpus: a parere on them
                    gets the copy that says so (route out_of_scope, interaction_copy.OUT_OF_CORPUS_MATTER).
  role              the client's position: attore | convenuto | imputato | persona_offesa | parte_civile | terzo |
                    sconosciuto
  stage             pre_filing | indagini | primo_grado | impugnazione | esecuzione | sconosciuto
  prior_decisions   [{court, outcome, date}] the decisions the text states
  client            {name, side, parties}: the client the text names (interactive_flow.stated, fact_signals.client_from,
                    the exam's «assunte le vesti del legale di X»), its side from the role (accused | victim | terzo |
                    ""), the parties the text names (query_router.party_kinds, the concrete ones)
  facts             the text narrates a case (events, and two parties or the user)
  confidence        {deliverable, rito, role, stage: rule | model | default}
  why               {field: the rule that decided it}
  route             what the API runs: case_first | generation | document | rag | out_of_scope (route_of)

Deterministic rules first, one model call (ask_model) only for the fields the rules leave empty -- for a case (parere,
atto) or an unknown deliverable; a question's role and stage are not asked. What is still empty takes its default:
parere for a text that narrates a case (altro without one, revisione_documento for a message about its upload),
pre_filing for a case, and role and rito "sconosciuto": the existing client logic and area frame decide them
downstream as before (resolve() fills them from the analysis, confidence default).

Routing (route_of): parere -> the case-first single shot; atto -> document generation (the case-first first read as
context when the text narrates facts), an upload to fill -> the document route; cenni, quesito, calcolo, altro -> the
plain RAG answer; revisione_documento -> the document route (the plain answer without an upload); ricerca_giurisprudenza
and a parere on an out-of-corpus matter -> the out-of-scope copy.
"""

import json
import logging
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .verbose_logger import vlog

logger = logging.getLogger(__name__)

DELIVERABLES = ("parere", "atto", "cenni", "quesito", "revisione_documento", "calcolo", "ricerca_giurisprudenza",
                "altro")
RITI = ("civile", "lavoro", "famiglia", "esecuzione", "penale", "penale_gdp", "amministrativo", "tributario",
        "fallimentare", "sconosciuto")
ROLES = ("attore", "convenuto", "imputato", "persona_offesa", "parte_civile", "terzo", "sconosciuto")
STAGES = ("pre_filing", "indagini", "primo_grado", "impugnazione", "esecuzione", "sconosciuto")
BLOCK_KINDS = ("cenni", "parere", "alternative", "atto", "quesito")
UNKNOWN = "sconosciuto"
OUT_OF_CORPUS = ("tributario", "fallimentare")
CASE_DELIVERABLES = ("parere", "atto")
RULE, MODEL, DEFAULT = "rule", "model", "default"
FIELDS = ("deliverable", "rito", "role", "stage")

CASE_FIRST, GENERATION, DOCUMENT, RAG, OUT_OF_SCOPE = "case_first", "generation", "document", "rag", "out_of_scope"
ROUTES = {"parere": CASE_FIRST, "atto": GENERATION, "cenni": RAG, "quesito": RAG, "revisione_documento": DOCUMENT,
          "calcolo": RAG, "ricerca_giurisprudenza": OUT_OF_SCOPE, "altro": RAG}
# the document route's own intents (api._chat_impl's document branch), by deliverable
DOC_INTENTS = {"revisione_documento": "analyse", "parere": "defensive_draft"}

MODEL_TOKENS = 160
MODEL_VIEW_HEAD, MODEL_VIEW_TAIL = 500, 900     # a long message: its start and its end, where the ask sits
UPLOAD_VIEW, HISTORY_TURNS, HISTORY_CHARS = 700, 4, 300
QUESTION_MAX_WORDS = 35                         # one short question, no facts: a quesito
ARTICLE_ONLY_MAX_WORDS = 15                     # «art. 1418 c.c. cause di nullità del contratto»: an article to explain


def _flat(text: Any) -> str:
    return " ".join(str(text or "").split())


def _low(text: Any) -> str:
    return _flat(text).lower()


# --- the text's sentences and its ask -------------------------------------------------------------------------------

_SPLIT_RE = re.compile(r"(?<=[.?!])\s+(?=[«\"“(]?[A-ZÀ-Ü])|\n+")


def sentences(text: str) -> List[str]:
    """The text's sentences: a stop followed by a capital opens the next («art. 584 cp. Il candidato»); v75: never
    after an abbreviation («il Sig. Caio»: sentence_split)."""
    from .sentence_split import rejoin
    out: List[str] = []
    for line in re.split(r"\n+", text or ""):
        out += [s for s in (_flat(x) for x in rejoin(_SPLIT_RE.split(line))) if s]
    return out


# What makes a sentence the request: an imperative or the exam's subjunctive, a wish (the client's goal: «vuole
# separarsi»), a question.
_ASK_MARK_RE = re.compile(
    r"\?|\bil candidat[oa]\b|\b(?:redig|predispon|prepar[ai]|scriv[ia]|stend[ia]|formul[ai]|elabor[ai]|fammi|dammi|"
    r"genera|crea(?:mi)?\b|spiega|illustr|indic[ah]|espon[gi]|individu|qualific|prospett|delinea|valut[ai]\b|"
    r"analizz|rived|cerca|trova|elenca|calcol|dimmi|descriv|esamin|tratt[ai]\b|discuta|definisca|suggeri|renda\b|"
    r"vorrei|voglio|mi serve|ho bisogno|puoi|potresti|chiedo|chiede (?:di|un|una)|premess[io]|ipotizzando|"
    r"si rivolge|si reca|vuol\w*|vogliono|intend\w*|desider\w*)", re.IGNORECASE)


def ask_part(text: str) -> str:
    """The text's request: the sentences that ask (a question, an imperative, the exam's «il candidato …») and its last
    sentence; the whole of a short text (two sentences or fewer)."""
    found = sentences(text)
    if len(found) <= 2:
        return _flat(text)
    asked = [s for s in found[:-1] if _ASK_MARK_RE.search(s)]
    return " ".join(asked + [found[-1]])


# --- the deliverable --------------------------------------------------------------------------------------------------

# case law asked for: a search verb before a ruling's noun, the Court's orientation, «giurisprudenza su …»
_CASELAW_NOUN = r"(?:sentenz\w*|pronunc\w*|massim[ae]\b|precedenti giurisprudenzial\w*|giurisprudenz\w*|decisioni)"
_CASELAW_RE = re.compile(
    r"\b(?:cerca(?:mi)?|trova(?:mi)?|elenca(?:mi)?|indica(?:mi)?|segnala(?:mi)?|dammi|mostrami|riporta(?:mi)?|"
    r"cita(?:mi)?|quali sono|ci sono|esistono|c['’]è|conosci)\b[^.?!]{0,60}?\b" + _CASELAW_NOUN +
    r"|\borientament\w*\s+(?:della\s+(?:suprema\s+)?(?:corte\s+(?:di\s+)?)?cassazione|giurisprudenzial\w*|"
    r"di legittimità|delle sezioni unite|della giurisprudenza|dei giudici)"
    r"|\bcome (?:si (?:è|sono) (?:pronunciat|espress)\w*|ha(?:nno)? (?:deciso|statuito)) (?:la|le) (?:suprema )?"
    r"(?:corte (?:di )?)?(?:cassazione|sezioni unite)"
    r"|\b(?:cosa|che cosa) (?:dice|ha detto|ha deciso|sostiene) la (?:suprema )?(?:corte (?:di )?)?cassazione"
    r"|\bgiurisprudenza (?:recente |più recente |di legittimità |di merito )?(?:su|sul|sulla|sullo|sugli|sulle|sui|"
    r"in materia|in tema)\b"
    r"|\bsentenze (?:recenti )?(?:della cassazione |di cassazione )?(?:su|sul|sulla|in materia|in tema)\b",
    re.IGNORECASE)
# a parere or a strategy asked for
_PARERE_ASK_RE = re.compile(
    r"\bparere\b|\bstrategi\w*|\blinea difensiva\b|\bdifend\w*|\bdifes[ao]\b|\bargomentazioni difensive\b|"
    r"\bcome (?:posso|può|possiamo|devo|deve|dobbiamo|potrebbe) (?:agire|tutelar\w*|difender\w*|procedere|"
    r"comportar\w*|muover\w*)\b|\bcosa (?:posso|può|possiamo|devo|deve|dovrei|dovrebbe|può fare) fare\b|"
    r"\bche (?:cosa )?(?:posso|può|possiamo) fare\b|\bcome (?:agire|tutelarsi|tutelarl\w*|difendersi|difenderl\w*|"
    r"procedere|muoversi|comportarsi|rispondere|impostare la difesa)\b|\bcome (?:mi|ci|lo|la|li) (?:difendo|tutelo|"
    r"difendiamo|tuteliamo)\b|\bquali (?:azioni|tutele|rimedi|strumenti|iniziative|difese|"
    r"eccezioni|possibilità)\b|\b(?:azioni|iniziative) (?:da (?:intraprendere|assumere|proporre)|proponibili|"
    r"esperibili|possibili)\b|\bha ragione\b|\b(?:può|possono|posso|possiamo|potrebbe) agire\b|\bcontro chi\b|"
    r"\bnei confronti di chi\b|\ba quale titolo\b", re.IGNORECASE)
# an act to draft: a drafting verb before an act's name, the act's name opening the request, «mi serve un ricorso»
_ACT_NOUN = (r"(?:atto|atti|ricors\w*|citazion\w*|querel\w*|denunci\w*|diffid\w*|comparsa|memori[ae]|appello|precetto|"
             r"istanz[ae]|opposizion\w*|reclamo|esposto|decreto ingiuntivo|messa in mora|lettera|contratto|"
             r"scrittura privata|procura|testamento|transazione|nomina|intimazione|sfratto|bozza|modulo)")
_ACT_VERB = (r"(?:redig(?:i|imi|a|ere|ermi|ete)|predispon(?:i|imi|ga|ete|ere|ermi)|prepar(?:a|ami|i|are|armi|ate)|"
             r"scriv(?:i|imi|a|ere|ermi|ete)|stend(?:i|imi|a|ere)|formul(?:a|ami|i|are)|elabor(?:a|ami|i|are)|fammi|"
             r"genera(?:mi)?|crea(?:mi)?|impost(?:a|ami|i))")
_ACT_REQUEST_RE = re.compile(rf"\b{_ACT_VERB}\b((?:\s+[^\s.?!;]+){{0,4}}?)\s+(?:l['’]\s*|il\s+|la\s+|lo\s+|un\s+|"
                             rf"una\s+|un['’]\s*|gli\s+|le\s+|i\s+)?{_ACT_NOUN}\b", re.IGNORECASE)
_ACT_OPENING_RE = re.compile(
    r"^\W*(?:(?:un|una|l['’]|il|la)\s*)?(?:atto (?:di|d['’])\s*(?:citazione|precetto|appello|opposizione|querela|"
    r"costituzione|intervento)|ricorso\b|citazione\b|querela\b|denuncia(?:-querela)?\b|diffida\b|comparsa\b|"
    r"memoria (?:difensiva|di costituzione|ex)|atto\b|istanza\b|reclamo\b|esposto\b)"
    r"(?!\s+(?:presentat|sport|notificat|depositat|propost|ricevut|inviat)\w*)", re.IGNORECASE)
_ACT_WISH_RE = re.compile(rf"\b(?:mi serve|mi servirebbe|ho bisogno di|vorrei|voglio|mi occorre|serve)\s+"
                          rf"(?:un|una|un['’]|l['’]|il|la)\s*{_ACT_NOUN}\b", re.IGNORECASE)
# an upload to fill or duplicate (the document branch's own signals, api._GENERATE_SIGNALS)
_TEMPLATE_FILL_RE = re.compile(
    r"\b(?:compila\w*|riempi\w*|usa(?:lo|la)? come (?:template|modello)|usa quest[oa]|duplica\w*|copia(?:mi)?|"
    r"fanne una copia|fa(?:i|mmi) una copia|stess[oa] (?:modulo|documento|richiesta)|uguale (?:ma )?con|"
    r"identico ma con|(?:modifica|cambia|aggiorna|sostituisci) i dati|con i (?:miei )?dati|rigenera\w*|rifa(?:i|mmi))\b",
    re.IGNORECASE)
# the upload reviewed: «analizza», «rivedi», «cosa ne pensi», a summary
_REVIEW_RE = re.compile(
    r"\b(?:analizz\w*|rived\w*|revision\w*|(?:cosa|che) ne pensi|(?:cosa|che) pensi|controll\w*|verific\w*|"
    r"esamin\w*|comment\w*|riassum\w*|riassunto|sintetizz\w*|sintesi|leggi|confront\w*|paragon\w*|compar[ai]\w*|"
    r"cosa (?:dice|prevede|contiene|stabilisce)|ci sono (?:errori|problemi|criticità)|criticit\w*|clausol\w*)",
    re.IGNORECASE)
# a question about the upload: the document branch's analyse signals (api._ANALYSE_SIGNALS, Italian)
_DOC_QUESTION_RE = re.compile(
    r"\b(?:riassum\w*|analizz\w*|spiegami|interpret\w*|estrai|dimmi|descrivimi|cosa (?:dice|prevede|contiene|"
    r"stabilisce)|chi sono le parti|qual(?: è|['’]è)|quali sono|come funziona|riassunto|analisi|sintesi|estratto|"
    r"riepilogo|fammi un|dammi un|vorrei un|vorrei sapere)\b", re.IGNORECASE)
# an institution explained, no facts
_CENNI_RE = re.compile(
    r"\bcenn[io]\b|\bspiega(?:mi|ci|re)?\b|\billustra(?:mi|re|re)?\b|\bcos['’]\s*è\b|\bche cos['’]\s*è\b|"
    r"\bcosa (?:è|e'|sono|si intende|significa)\b|\bin cosa consiste\b|\bparlami\b|\bdescriv(?:i|imi)\b|"
    r"\bdefinizione\b|\bnozione\b|\bcome funziona\b|\bdifferenz\w* (?:tra|fra)\b|\bcome si distingu\w*\b|"
    r"\bdisciplina (?:del|della|dello|dell['’]|dei|degli|delle)\b|\bquali sono (?:i|gli|le) (?:requisiti|presupposti|"
    r"elementi|caratteri|effetti)\b|\bprincipi\w* (?:in tema|in materia)\b|\bistitut[oi] (?:del|della|dell['’])\b",
    re.IGNORECASE)
# one point of law with a one-line answer: a term, a competence, a pena, the procedibilità
_QUESITO_RE = re.compile(
    r"\bentro (?:quanto|quale|che|quando)\b|\bqual(?:e|['’]è| è)? (?:è )?il termine\b|\btermine (?:per|di|entro)\b|"
    r"\bquanto tempo\b|\bin quanto tempo\b|\bsi prescriv\w*\b|\bprescrizion\w*\b|\bdecadenz\w*\b|"
    r"\bchi è competente\b|\b(?:giudice|tribunale|foro) competente\b|\b(?:qual|quale) (?:è )?il (?:giudice|tribunale|"
    r"foro)\b|\bcompeten(?:te|za)\b|\b(?:che|quale|qual è la) pena\b|\bquanto si rischia\b|\bpena (?:prevista|"
    r"massima|minima|edittale)\b|\bprocedibil\w*\b|\bperseguibil\w*\b|\ba querela\b|\bd['’]ufficio\b|"
    r"\bcontributo unificato\b|\bquanto costa\b", re.IGNORECASE)


_ARTICLE_NAMED_RE = re.compile(r"\bart(?:icol[oi])?\.?\s*\d+", re.IGNORECASE)


def caselaw_asked(text: str) -> bool:
    return bool(_CASELAW_RE.search(text or ""))


def act_requested(text: str) -> str:
    """The act the text asks to draft («redigi l'atto di citazione», «Querela contro …», «mi serve un ricorso»), or "".
    A drafting verb whose object is a parere is not one («rediga motivato parere sull'atto di citazione»)."""
    text = text or ""
    for m in _ACT_REQUEST_RE.finditer(text):
        if not re.search(r"\bparer\w*", m.group(1) or "", re.IGNORECASE):
            return _flat(m.group(0))
    first = sentences(text)[:1]
    m = _ACT_OPENING_RE.search(first[0]) if first else None
    if m:
        return _flat(m.group(0)).strip(" .,;:")
    m = _ACT_WISH_RE.search(text)
    return _flat(m.group(0)) if m else ""


# v61: today's short-topic rule, moved here from the API (api._is_topic_question calls it): a short message naming a
# legal topic, with no facts of the user's own and no request for a strategy, is a question whatever the classifier says
TOPIC_MAX_WORDS = 8
_OWN_FACTS_RE = re.compile(r"\b(io|ho|abbiamo|siamo|mi|me|mio|mia|miei|mie|nostro|nostra|nostri|nostre)\b",
                           re.IGNORECASE)
_STRATEGY_ASK_RE = re.compile(
    r"strategi|difend|difes[ao]\b|parere|cosa (?:posso|possiamo|devo|dobbiamo|dovrei|dovremmo) fare", re.IGNORECASE)


def topic_question(message: str) -> bool:
    message = message or ""
    return (len(message.split()) <= TOPIC_MAX_WORDS and not _OWN_FACTS_RE.search(message)
            and not _STRATEGY_ASK_RE.search(message))


def _narrative(text: str) -> bool:
    """Today's narrative check (interactive_flow.routes_here: the router, no tie-break)."""
    from . import interactive_flow
    return bool(text and interactive_flow.routes_here(text))


def _defence_request(text: str) -> bool:
    """The exam's and the defence's phrasing (calculation._DEFENCE_REQUEST_RE: «il candidato … rediga», «assunte le
    vesti», «linea difensiva»), never a short topic question -- today's shortcut to a defence."""
    from .calculation import _DEFENCE_REQUEST_RE
    return bool(_DEFENCE_REQUEST_RE.search(text or "")) and not topic_question(text)


def _generation_request(text: str) -> bool:
    """Today's generation fast path (document_generation.is_generation_request)."""
    from .document_generation import is_generation_request
    try:
        return bool(is_generation_request(text or ""))
    except Exception:
        return False


def _signals(text: str) -> Dict[str, Any]:
    from . import query_router
    try:
        return query_router.signals(text or "")
    except Exception as exc:
        logger.warning("intake: the router's signals failed: %s", exc)
        return {"parties": [], "concrete_parties": [], "events": 0}


_DETERMINED_RE = re.compile(r"\b(?:il|lo|la|l['’]|un|una|uno|un['’]|i|gli|le|al|alla|allo|dal|dalla|del|della|dello|"
                            r"dell['’]|sul|sulla|nel|nella)\s*(?:(?:propri|su|mi|nostr|lor)[oaie]\s+|loro\s+)?"
                            r"([a-zà-ù]+)", re.IGNORECASE)


def _role_nouns(text: str) -> set:
    """The role stems the text names with a determiner (party_names.ROLE_STEMS: «un paziente», «il venditore»)."""
    from .party_names import ROLE_STEMS
    found = set()
    for m in _DETERMINED_RE.finditer(text or ""):
        word = m.group(1).lower()
        stem = next((s for s in ROLE_STEMS if len(s) >= 4 and word.startswith(s)), "")
        if stem:
            found.add(stem)
    return found


def _names(text: str) -> set:
    """The capitalised names the text writes, a sentence's opening one included («Leonardo è proprietario …»: the
    router's names need a lower-case word before them), never a usual opener or an institution."""
    from . import query_router
    return {m.group(1).lower() for m in query_router._CAPITALISED_RE.finditer(text or "")
            if m.group(1) not in query_router._SENTENCE_OPENERS and m.group(1) not in query_router._NOT_NAMES
            and m.group(1).lower() not in _NOT_PARTY_WORDS}


_NOT_PARTY_WORDS = {"il", "la", "lo", "le", "gli", "un", "una", "come", "quale", "quali", "qual", "cosa", "che", "chi",
                    "sei", "sono", "assisto", "difendo", "redigi", "predisponi", "calcola", "spiegami", "trovami",
                    "cenni", "art", "contratto", "tribunale", "corte", "comune", "agenzia", "entrate", "grazie"}


# the user's own events, which the router does not count («mi hanno licenziato», «ho ricevuto un atto»)
_OWN_EVENT_RE = re.compile(r"\b(?:mi|ci) hanno\s+\w+[aiu]t[oaie]\b|\b(?:ho|abbiamo)\s+(?:appena\s+|già\s+)?\w+[aiu]t[oaie]\b|"
                           r"\bsono stat[oaie]\s+\w+[aiu]t[oaie]\b", re.IGNORECASE)


def _story(text: str) -> str:
    """The text without the sentences that only say who the lawyer assists («Sei l'avvocato di Leonardo.»): the facts
    are the rest."""
    return " ".join(x for x in sentences(text) if not (_LAWYER_RE.search(x) and len(x.split()) <= 8))


def narrates_facts(text: str, sig: Optional[Dict[str, Any]] = None, parties_min: int = 2) -> bool:
    """The text narrates a case: an event, and two parties (parties_min) -- the router's, the names it writes, or the
    roles it names («un paziente … in un ospedale») -- or the user's own facts. The lawyer's own line is not a fact
    (_story)."""
    story = _story(text)
    sig = sig if sig is not None and story == _flat(text) else _signals(story)
    parties = set(sig.get("parties") or []) | _names(story)
    if not sig.get("events") and not ("io" in parties and _OWN_EVENT_RE.search(story)):
        return False
    return "io" in parties or len(parties | _role_nouns(story)) >= parties_min


def deliverable_rule(message: str, upload: str = "", refers_to_upload: bool = False,
                     sig: Optional[Dict[str, Any]] = None) -> Tuple[str, str]:
    """(deliverable, the rule) from the rules alone; ("", "") when none decides. In order: case law asked for (never
    beside a parere or the exam's ask); with an upload, a fill, a defence, a review; an act to draft; a narrative
    (today's router), the exam's or the defence's phrasing, a parere asked on one's own facts; today's generation fast
    path; a calculation; an institution explained; one point of law."""
    message = message or ""
    ask = ask_part(message)
    sig = sig if sig is not None else _signals(message)
    if caselaw_asked(ask) and not _PARERE_ASK_RE.search(ask) and not _defence_request(message):
        return "ricerca_giurisprudenza", "chiede giurisprudenza"
    if upload:
        if _TEMPLATE_FILL_RE.search(message) and not _PARERE_ASK_RE.search(ask):
            return "atto", "il documento caricato da compilare"
        if _PARERE_ASK_RE.search(ask) or _defence_request(message):
            return "parere", "una strategia sul documento caricato"
        if _REVIEW_RE.search(ask):
            return "revisione_documento", "il documento caricato da esaminare"
    act = act_requested(message)
    if act:
        return "atto", f"l'atto chiesto: «{act[:60]}»"
    single = point_of_law_question(message, sig)
    if single:              # v65: a narrative closed by one question of law; v68: an article named or not
        return "quesito", f"una sola domanda di diritto: «{single[:60]}»"
    if _narrative(message):
        return "parere", "narrativa (router)"
    if _defence_request(message):
        return "parere", "la richiesta di una difesa (il candidato, le vesti, la linea difensiva)"
    if _PARERE_ASK_RE.search(ask) and narrates_facts(message, sig, parties_min=1) and not topic_question(message):
        return "parere", "un parere chiesto sui fatti di una parte"
    if upload and _narrative(f"{upload}\n\n{message}"):
        return "parere", "il caso è nel documento caricato"
    if upload and (refers_to_upload or _DOC_QUESTION_RE.search(message)):
        return "revisione_documento", "una domanda sul documento caricato"
    if _LAWYER_RE.search(message) and _ASK_MARK_RE.search(ask) and not caselaw_asked(ask):
        return "parere", "l'avvocato chiede per il proprio cliente"
    if _generation_request(message):
        return "atto", "la richiesta di un documento (generazione)"
    from .query_router import asks_calculation
    if asks_calculation(message) and not narrates_facts(message, sig):
        if not decorrenza_date(message):        # v69: no date to count from -- the term and its article, no number
            return "quesito", "un calcolo chiesto senza una data di decorrenza: il termine e il suo articolo"
        return "calcolo", "un importo da calcolare"
    if _QUESITO_RE.search(ask) and not _PARERE_ASK_RE.search(ask):
        return "quesito", "un punto di diritto (termine, competenza, pena)"
    if narrates_facts(message, sig) and not topic_question(message):
        return "", ""
    if _CENNI_RE.search(ask):
        return "cenni", "un istituto da spiegare, senza fatti"
    if _ARTICLE_NAMED_RE.search(message) and len(message.split()) <= ARTICLE_ONLY_MAX_WORDS and "?" not in message:
        return "cenni", "un articolo nominato, senza fatti"
    if "?" in message and len(sentences(message)) <= 2 and len(message.split()) <= QUESTION_MAX_WORDS \
            and not _OWN_FACTS_RE.search(message):
        return "quesito", "una domanda breve, senza fatti"
    return "", ""


# --- the requested blocks -----------------------------------------------------------------------------------------------

_SUBJECT_END = (r"(?=,?\s+(?:il candidat|delin|redig|illustr|indich|espong|individu|prospett|qualific|definisc|"
                r"esamin|valut|si soffermi|ipotizz|e (?:delin|redig|illustr|indich|espong|individu|prospett|definisc))|"
                r"[.;?!]|$)")
_BLOCK_RES = (
    ("cenni", re.compile(r"\b(?:premess[io]|svolt[io]|fatt[io]|con)?\s*(?:(?:brevi|sintetici|alcuni)\s+)?cenni\s+"
                         r"(?:sull['’]|sulla|sullo|sulle|sugli|sui|sul|su|in tema di|in materia di|a)\s*(.+?)"
                         + _SUBJECT_END, re.IGNORECASE)),
    ("cenni", re.compile(r"\bpremess[io]\s+(?:brevemente\s+)?(?:i\s+principi|le\s+nozioni|l['’]inquadramento|"
                         r"un\s+inquadramento)\s+(?:in tema di|in materia di|sull['’]|sulla|sul|su|della|del|di)\s*"
                         r"(.+?)" + _SUBJECT_END, re.IGNORECASE)),
    ("parere", re.compile(r"\b(?:delinea|delinei|redig\w*|illustr\w*|indich\w*|espong\w*|individu\w*|prospett\w*|"
                          r"definisc\w*|renda|fornisca|qualifichi|qualifica)\s+(?:[^\s,.;?!]+\s+){0,5}?"
                          r"((?:strategi\w*|linea difensiva|linee difensive|parere(?: motivato)?|argomentazioni "
                          r"difensive|iniziative|azioni|questioni (?:giuridiche|di diritto)|giuridicamente (?:il|i) "
                          r"fatt[oi])[^,.;?!]{0,80})", re.IGNORECASE)),
    ("alternative", re.compile(r"\b(?:ipotizzando|valutando|indicando|individuando|prospettando)\s+(?:anche\s+)?"
                               r"((?:(?:le|i|gli)\s+)?(?:possibili|eventuali)?\s*(?:reati|qualificazioni|"
                               r"fattispecie|ipotesi|soluzioni)?\s*alternativ\w*[^,.;?!]{0,80})", re.IGNORECASE)),
    ("atto", _ACT_REQUEST_RE),
)
_ARTICLE_RE = re.compile(r"^(?:il|lo|la|l['’]|i|gli|le)\s*", re.IGNORECASE)


def _subjects(text: str) -> List[str]:
    """«sull'omicidio preterintenzionale, il nesso causale e l'elemento soggettivo» -> the three subjects."""
    parts = re.split(r",\s*|\s+e\s+(?=(?:il|lo|la|l['’]|i|gli|le|sul|sulla|sull['’])\b)", text or "")
    out = []
    for p in parts:
        p = _ARTICLE_RE.sub("", _flat(p)).strip(" .,;:")
        p = re.sub(r"^(?:sul|sulla|sullo|sull['’]|sui|sugli|sulle)\s*", "", p, flags=re.IGNORECASE)
        if p:
            out.append(p)
    return out


# v65: the questions the message lists after its lead-in («si chiede di valutare: 1. … 2. …», «in particolare: a) …»)
_QUESITI_LEAD_RE = re.compile(
    r"\b(?:si\s+chiede(?:\s+(?:al\s+candidat[oa]\s+)?(?:di\s+)?(?:valutare|stabilire|chiarire|indicare|precisare|"
    r"verificare|esaminare|rispondere(?:\s+ai\s+seguenti\s+quesiti)?|sapere|se))?|in\s+particolare|i\s+(?:seguenti\s+)?"
    r"quesiti(?:\s+sono)?|le\s+(?:seguenti\s+)?domande(?:\s+sono)?|(?:vorrei|vorremmo|vogliamo)\s+sapere)\s*:",
    re.IGNORECASE)
_ITEM_MARK_RE = re.compile(r"(?:(?<=^)|(?<=[\n:;.?!]))\s*(?:\d{1,2}[.)]|[a-hA-H][.)]|[-•–*])\s+")
QUESITO_CHARS = 300


def quesiti(message: str) -> List[Tuple[int, str]]:
    """v65: [(position, question)] the message lists after a lead-in, in order -- numbered («1.», «1)»), lettered
    («a)»), dashed or bulleted items; without markers, the lead-in's «;»-separated clauses (two at least), else its
    one clause up to the sentence's end. v69: with no lead-in, two or more question sentences -- each one, in order
    (question_sentences)."""
    if not _QUESITI_LEAD_RE.search(message or ""):
        return question_sentences(message)
    out: List[Tuple[int, str]] = []
    for lead in _QUESITI_LEAD_RE.finditer(message or ""):
        start = lead.end()
        tail = message[start:]
        nxt = _QUESITI_LEAD_RE.search(tail)
        tail = tail[:nxt.start()] if nxt else tail
        marks = list(_ITEM_MARK_RE.finditer(tail))
        items: List[Tuple[int, str]] = []
        if marks:
            for j, m in enumerate(marks):
                end = marks[j + 1].start() if j + 1 < len(marks) else len(tail)
                items.append((start + m.end(), tail[m.end():end]))
        else:
            first = re.split(r"(?<=[.?!])\s+(?=[A-ZÀ-Ü])|\n\s*\n", tail.strip(), maxsplit=1)[0]
            parts = [x for x in re.split(r"\s*;\s*", first) if x.strip()]
            at = start + len(tail) - len(tail.lstrip())
            for part in (parts if len(parts) >= 2 else [first]):
                items.append((at + max(first.find(part), 0), part))
        for pos, text in items:
            text = _flat(text).strip(" ;,").rstrip(".").strip()
            if len(text.split()) >= 2:
                out.append((pos, text[:QUESITO_CHARS]))
    return out


_QUESTION_SPLIT_RE = re.compile(r"\?")
QUESTIONS_MIN = 2


def question_sentences(message: str) -> List[Tuple[int, str]]:
    """v69: [(position, question)] -- the message's question sentences (ending in «?»), in order, when there are two or
    more of two words or more; [] otherwise (one question is the single quesito's, point_of_law_question)."""
    out: List[Tuple[int, str]] = []
    at = 0
    for sentence in sentences(message or ""):
        pos = (message or "").find(sentence, at)
        at = pos + len(sentence) if pos >= 0 else at
        text = _flat(sentence).strip()
        if text.endswith("?") and len(text.split()) >= 2:
            out.append((max(pos, 0), text[:QUESITO_CHARS]))
    return out if len(out) >= QUESTIONS_MIN else []
# v68: a single question that still asks for the case's analysis -- a course of action («come impugnare il
# licenziamento?», «come ottengo l'affidamento?», «come contestarlo?», «cosa può eccepire?») or the qualification of the
# facts («quali reati sono configurabili?») -- is the parere's, never a quesito
_CASE_ASK_RE = re.compile(r"\bcome\s+(?:[a-zà-ü]+(?:are|ere|ire|(?:ar|er|ir)(?:lo|la|li|le|ne|si|mi|ti|ci|vi))|posso|"
                          r"possiamo|devo|dobbiamo|ottengo|otteniamo|faccio|facciamo|procedo|procediamo|agisco|agiamo)\b|"
                          r"\b(?:che\s+)?cosa\s+(?:posso|può|possiamo|possono|devo|deve|dobbiamo|dovrei|dovrebbe)\b|"
                          r"\bquali?\s+(?:reati|reato|illeciti|illecito|responsabilit\w*|qualificazion\w*|fattispecie)\b|"
                          r"\bconfigurabil\w*|\bqualificazione giuridica\b", re.IGNORECASE)


def point_of_law_question(message: str, sig: Optional[Dict[str, Any]] = None) -> str:
    """v65: the one question a narrative ends on when it is a point of law on a named article («… È applicabile l'art.
    2051 c.c.?»): its last sentence asks, names an article, and is the message's only question, with no list of
    questions nor an exam's or a parere's ask; "" otherwise. v68: whether or not it names an article -- a narrative with
    exactly one question and no list is a quesito («… La causa spetta al giudice di pace o al tribunale?»); a strategy
    asked (_PARERE_ASK_RE: «quali azioni», «cosa può fare», «parere» …), a course of action or the facts' qualification
    (_CASE_ASK_RE: «come impugnare …?», «quali reati sono configurabili?») is still a parere."""
    found = sentences(message)
    if len(found) < 2 or (message or "").count("?") != 1 or quesiti(message):
        return ""
    asked = next((x for x in found if "?" in x), "")
    if not asked:
        return ""
    if _PARERE_ASK_RE.search(asked) or _CASE_ASK_RE.search(asked) or _PARERE_ASK_RE.search(ask_part(message)) \
            or _defence_request(message) or not narrates_facts(message, sig, parties_min=1):
        return ""
    return asked


def requested_blocks(message: str) -> List[Dict[str, str]]:
    """The explicit asks inside the message, in its order -- «premessi brevi cenni su A, B e C, … delinei una
    strategia difensiva, ipotizzando anche reati alternativi» -> cenni A, cenni B, cenni C, parere, alternative. Empty
    for a plain narrative: no ask, or the one parere ask alone. v65: each listed question a «quesito» block with its
    text (quesiti), in its place."""
    found: List[Tuple[int, str, str]] = [(pos, "quesito", text) for pos, text in quesiti(message)]
    for kind, pattern in _BLOCK_RES:
        for m in pattern.finditer(message or ""):
            if kind == "atto":
                if re.search(r"\bparer\w*", m.group(1) or "", re.IGNORECASE):
                    continue
                subject = _flat(m.group(0))
                found.append((m.start(), kind, subject))
            elif kind == "cenni":
                for i, subject in enumerate(_subjects(m.group(1))):
                    found.append((m.start(1) + i, kind, subject))
            else:
                found.append((m.start(), kind, _flat(m.group(1)).strip(" .,;:")))
    found.sort(key=lambda x: x[0])
    blocks, seen = [], set()
    for _, kind, subject in found:
        key = (kind, subject.lower())
        if subject and key not in seen:
            seen.add(key)
            blocks.append({"kind": kind, "subject": subject})
    if len(blocks) == 1 and blocks[0]["kind"] == "parere":
        return []
    return blocks


# --- v65: the articles the user names, with the verb around them -------------------------------------------------------

# the client wants to use it (invocare, esperire, eccepire, far valere …); used against the client (a condemnation, a
# payment or a charge asked under it)
_USE_VERB_RE = re.compile(r"\b(?:invoc\w*|esperi\w*|eccepi\w*|eccepisc\w*|far(?:e|ne)?\s+valere|fa(?:re)?\s+valere|"
                          r"avvaler\w*|avvalers\w*|richiamar\w*|azionar\w*|propor\w*\s+(?:l['’]\s*)?azion\w*|"
                          r"agire\s+(?:ex|ai sensi)|"
                          # v68: «se sussistano i presupposti per recedere / proporre … ai sensi di»
                          r"reced\w*|propor(?:re|rsi|r[eò]|ranno|ne)\b|esercit\w*\s+(?:il|la|l['’])\s*\w+)", re.IGNORECASE)
# v68: a request under the article (a condemnation, a payment, the resolution …, or suing under it): used against the
# client only when the counterparty is the verb's subject; a charge (contestare, imputare, addebitare; «è stato citato /
# convenuto in giudizio»): against the client unless the client is its subject
_REQUEST_VERB_RE = re.compile(r"\b(?:chied\w*|domand\w*|pretend\w*)\s+(?:la\s+condanna|il\s+risarcimento|il\s+pagamento|"
                              r"la\s+risoluzione|la\s+restituzione|l['’]\s*adempimento|la\s+riduzione)[^.;]{0,120}?"
                              r"\b(?:ai sensi|ex|a norma|in base)\b|"
                              r"\b(?:citare|convenire|cita|citano|conviene|convengono)\s+in giudizio[^.;]{0,120}?"
                              r"\b(?:ai sensi|ex|a norma)\b", re.IGNORECASE)
_CHARGE_VERB_RE = re.compile(r"\b(?:contest\w*|imput\w*|addebit\w*)\b", re.IGNORECASE)
# v68: «Tizio è stato convenuto in giudizio ai sensi …»: the one sued is its subject -- a threat unless the counterparty
_SUED_RE = re.compile(r"\b(?:citat[oaie]|convenut[oaie])\s+in giudizio[^.;]{0,120}?\b(?:ai sensi|ex|a norma)\b",
                      re.IGNORECASE)
# v68: the verb's own clause -- the subject of «chiedere» in «Tizio chiede se sussistano i presupposti per chiedere …» is
# no one named: the words after the question's «se» / «che» / «per»
_CLAUSE_OPEN_RE = re.compile(r"\b(?:se|che|per|di)\s+", re.IGNORECASE)
_SELF_SUBJECT_RE = re.compile(r"\b(?:posso|possiamo|intendo|intendiamo|vorrei|vorremmo|voglio|vogliamo|chiedo|"
                              r"chiediamo|il\s+(?:mio\s+)?cliente|la\s+(?:mia\s+)?cliente|l['’]\s*assistit[oa]|"
                              r"(?:mio|mia|nostro|nostra)\s+(?:cliente|assistit[oa]))\b", re.IGNORECASE)
_OTHER_SUBJECT_RE = re.compile(r"\b(?:controparte|p\.?\s?m\.?|pubblico ministero|procura|attor[ei]|attrice|convenut\w*|"
                               r"ricorrent\w*|resistent\w*|creditor\w*|debitor\w*)\b", re.IGNORECASE)
WEAPON, THREAT = "weapon", "threat"


def named_articles(message: str, client: str = "") -> List[Dict[str, str]]:
    """v65: the articles the message names, each once, with the verb around it and its use -- weapon when the client
    wants to use it (invocare / esperire / eccepire … with the client, the user or no one as the subject), threat when
    it is used against the client (a condemnation or a payment asked under it, a charge; or a use verb with another
    party as the subject); "" with no verb. [{"article", "verb", "use"}]. v68: the article the client's question is about
    («se sussistano i presupposti per chiedere / recedere / invocare / proporre … ai sensi di») is a weapon -- a request
    is a threat only when the counterparty is the subject of its verb (in the verb's own clause); a charge as before."""
    from . import defensive_generation as dg
    from .interactive_flow import _key, _label
    out: Dict[str, Dict[str, str]] = {}
    names = [re.escape(x) for x in (client or "").split() if len(x) > 2]
    for sentence in sentences(message or ""):
        from . import citations
        refs = [r for r in citations.article_refs(sentence) if len(r) == 2]      # v72: the shared tolerant parser
        if not refs:
            continue
        use, verb = "", ""
        charge, sued = _CHARGE_VERB_RE.search(sentence), _SUED_RE.search(sentence)
        m = charge or sued or _REQUEST_VERB_RE.search(sentence) or _USE_VERB_RE.search(sentence)
        if m:
            verb = _flat(m.group(0)).split()[0].lower()
            subject = sentence[:m.start()]
            mine = bool(_SELF_SUBJECT_RE.search(subject) or (names and re.search(rf"\b(?:{'|'.join(names)})\b",
                                                                                     subject)))
            if m is charge:
                use = WEAPON if mine else THREAT
            elif m is sued:
                use = WEAPON if not mine and (_OTHER_SUBJECT_RE.search(subject) or _names(subject) - {"io"}) else THREAT
            else:                           # v68: the verb's clause decides -- the counterparty its subject, or no one
                opens = list(_CLAUSE_OPEN_RE.finditer(subject))
                own = subject[opens[-1].end():] if opens else subject
                mine = mine and not opens or bool(_SELF_SUBJECT_RE.search(own) or (names and re.search(
                    rf"\b(?:{'|'.join(names)})\b", own)))
                other = not mine and bool(_OTHER_SUBJECT_RE.search(own) or _names(own) - {"io"})
                use = THREAT if other else WEAPON
        for ref in refs:
            label = _label(_key(ref))
            if label not in out or (use and not out[label]["use"]):
                out[label] = {"article": label, "verb": verb, "use": use}
    return list(out.values())


# --- the rito -------------------------------------------------------------------------------------------------------------

_RITO_MARKERS = (
    ("penale", re.compile(
        r"\breat\w*|\bimputat\w*|\bindagat\w*|\bpubblico ministero\b|\bil p\.?\s?m\.?\b|\bprocura della repubblica\b|"
        r"\bquerel\w*|\bdelitt\w*|\bcontravvenzion\w*|"
        r"\brapin\w*|\bfurt\w*|\btruff\w*|\bomicidi\w*|\blesioni personali\b|\bestorsion\w*|\busur\w*|\bpeculato\b|"
        r"\bcorruzion\w*|\bconcussion\w*|\bappropriazione indebita\b|\bricettazion\w*|\briciclaggio\b|\bbancarotta\b|"
        r"\bdiffamazion\w*|\batti persecutori\b|\bmaltrattament\w*|\bviolenza sessuale\b|\bspaccio\b|"
        r"\bstupefacent\w*|\barrestat\w*|\bfermat[oa] (?:poco dopo|dalla polizia|dai carabinieri|da una pattuglia)|"
        r"\bcustodia cautelare\b|\bmisur\w* cautelar\w* personal\w*|\bgiudizio immediato\b|\brito abbreviato\b|"
        r"\bpatteggiament\w*|\budienza preliminare\b|\bg\.?i\.?p\.?\b|\bg\.?u\.?p\.?\b|\bparte civile\b|"
        r"\bpersona offesa\b|\bdenuncia(?:-querela)?\b(?!\s+(?:dei|del|della|delle|di nuova|di danno)\b)|\bdolo\b|\bconcorso (?:nel|di persone nel) reato\b|"
        r"\bpena\b(?! convenzionale| di (?:decadenza|nullità|inammissibilità|improcedibilità))|\buccid\w*|\bucciso\b|\bladr[oi]\b|\bpistola\b|\barma\b|\blegittima difesa\b|"
        r"\beccesso colposo\b|\bstato di necessità\b|\bsottratt\w*|\bavviso di garanzia\b|\bschiaff\w*|\bpercoss\w*",
        re.IGNORECASE)),
    ("amministrativo", re.compile(
        r"\bt\.?a\.?r\.?\b|\bconsiglio di stato\b|\bprovvediment\w* (?:amministrativ\w*|di revoca|di esclusione|di "
        r"diniego|impugnat\w*)|\bpubblica amministrazione\b|\binteresse legittimo\b|\bautotutela\b|\b21[- ]?(?:nonies|"
        r"quinquies|octies)\b|\bsilenzio[- ](?:inadempimento|rifiuto|assenso)\b|"
        r"\bbando\b|\bgar[ae]\b(?! di)|\bappalt\w* pubblic\w*|\bstazione appaltante\b|\baggiudicazion\w*|"
        r"\bconcorso pubblico\b|\bconcorso (?:per|a) (?:titoli|esami)|\bgraduatori\w*|\bcommissione (?:di concorso|"
        r"giudicatrice|esaminatrice)|\bordinanza sindacale\b|\bpermesso di costruire\b|\bcodice dei contratti "
        r"pubblici\b|\bricorso straordinario\b|\billecito professionale\b|\bostensione\b|\baccesso agli atti\b|"
        r"\bvizi\w* (?:inficianti|del provvedimento|di legittimità)\b|\beccesso di potere\b|\bdifetto di "
        r"motivazione\b",
        re.IGNORECASE)),
    ("tributario", re.compile(
        r"\bagenzia delle entrate\b|\bcartell\w* (?:esattorial\w*|di pagamento)|\bavvis\w* di accertamento\b|"
        r"\baccertamento (?:fiscale|tributario|induttivo|sintetico)\b|\bcommissione tributaria\b|\bcorte di giustizia "
        r"tributaria\b|\bcontenzioso tributario\b|\bimpost[ae] (?:sui redditi|di registro|ipotecari\w*)|\birpef\b|\bires\b|"
        r"\birap\b|\biva (?:non versata|evasa|dovuta|a credito)|\bevasione fiscale\b|\briscossione coattiva\b|"
        r"\bagenzia (?:delle )?entrate[- ]riscossione\b|\bistanza di (?:autotutela|rimborso) (?:fiscale|tributari\w*)",
        re.IGNORECASE)),
    ("fallimentare", re.compile(
        r"\bdichiarazione di fallimento\b|\bdichiarat\w* fallit\w*|\bsentenza (?:dichiarativa )?di fallimento\b|"
        r"\bliquidazione giudiziale\b|\bcurator\w* (?:fallimentar\w*|della liquidazione)|\bcuratela\b|"
        r"\bconcordato (?:preventivo|fallimentare|minore)\b|\bprocedur\w* concorsual\w*|\bcodice della crisi\b|"
        r"\bsovraindebitamento\b|\brevocatoria fallimentare\b|\bstato passivo\b|\binsinuazion\w* al passivo\b|"
        r"\bammissione al passivo\b|\bcomposizione negoziata\b", re.IGNORECASE)),
    ("esecuzione", re.compile(
        r"\bpignorament\w*|\bpignorat\w*|\batto di precetto\b|\bprecetto\b|\besecuzione forzata\b|\bespropriazione "
        r"(?:forzata|immobiliare|mobiliare|presso terzi)\b|\bopposizione (?:all['’]esecuzione|agli atti esecutivi|"
        r"di terzo all['’]esecuzione|al precetto)\b|\btitolo esecutivo\b|\bufficiale giudiziario\b|\bvendita "
        r"(?:all['’]asta|forzata)\b|\bgiudice dell['’]esecuzione\b|\bterzo pignorato\b|\bdichiarazione del terzo\b",
        re.IGNORECASE)),
    ("lavoro", re.compile(
        r"\blicenziament\w*|\blicenziat\w*|\bdemansionament\w*|\bmansioni\b|\bdifferenze retributive\b|"
        r"\bretribuzion\w*|\btrattamento di fine rapporto\b|\btfr\b|\bgiudice del lavoro\b|\brito del lavoro\b|"
        r"\bmobbing\b|\bstraordinari\b|\bferie non godute\b|\bcontratt\w* (?:di lavoro|a termine|a tempo "
        r"determinato)\b|\brapporto di lavoro\b|\bc\.?c\.?n\.?l\.?\b|\bdatore di lavoro\b|\bdimission\w*|"
        r"\b(?:infortunio|malattia) (?:sul lavoro|professionale)\b|\binail\b|\bsubordinazion\w*", re.IGNORECASE)),
    ("famiglia", re.compile(
        r"\bseparazion\w*|\bseparar\w*|\bdivorzi\w*|\bscioglimento del matrimonio\b|\bcessazione degli effetti civili\b|"
        r"\baffidament\w* (?:condiviso|esclusivo|super-?esclusivo|dei figli|del figlio|della figlia|dei minori)\b|"
        r"\baffido\b|\bassegno (?:di mantenimento|divorzile|perequativo|per i figli)\b|\bmantenimento\b|"
        r"\bcollocamento (?:dei|del|della) (?:figli|figlio|figlia|minor\w*)\b|\bresponsabilità genitoriale\b|"
        r"\bunione civile\b|\bconvivenza di fatto\b|\badozion\w*|\bdisconoscimento\b|\bcasa (?:coniugale|familiare)\b|"
        r"\bcontributo (?:economico|al mantenimento)\b|\bfigli\b[^.]{0,40}\b(?:restino|vivano|stiano) con\b",
        re.IGNORECASE)),
    ("civile", re.compile(
        r"\bcontratt\w*|\brisarciment\w*|\bdann[oi] (?:subit|patit|material|moral|biologic|patrimonial|non "
        r"patrimonial|emergent)\w*|\blucro cessante\b|\bpropriet\w*|\bservit\w*|\blocazion\w*|\bconduttor\w*|"
        r"\blocator\w*|\bcondomin\w*|\bereditari\w*|\beredit\w*|\bsuccession\w*|\bcompravendit\w*|\bvendita\b|"
        r"\bvizi (?:della cosa|occulti|redibitori|dell['’]opera|dell['’]immobile|costruttivi)\b|\binadempiment\w*|\bobbligazion\w*|\busucapion\w*|\bpossess\w*|\bmutuo\b|\bfideiussion\w*|"
        r"\bresponsabilità (?:civile|extracontrattuale|contrattuale)\b|"
        r"\btribunale civile\b|\bdecreto ingiuntivo\b|\bcreditor\w*|\bdebitor\w*|\bpenale (?:richiesta|contrattuale|"
        r"convenzionale)\b|\bclausola penale\b|\bpreliminare\b|\bdivisione\b|\bcomunione\b|\bconsumator\w*|"
        r"\bgaranzia\b|\barricchimento\b|\bassicurazion\w*|\bsinistro\b|\bimmobil\w*|\bfondo\b|\bterren[oi]\b|"
        r"\bcredit[oi]\b|\bdebit[oi]\b|\bfattur\w*|\bfornitur\w*|\bnon (?:paga|pagava|ha pagato|ha mai pagato)\b|"
        r"\bsollecit\w*|\bmessa in mora\b|\bdiffid\w*|\bvicin[oi]\b|\bimmission\w*|\butili\b|\bsoci[oe]\b|"
        r"\bcitat\w* in giudizio\b|\bconvenut\w*|\beccezion\w*",
        re.IGNORECASE)),
)
_GDP_RE = re.compile(r"\bgiudice di pace\b", re.IGNORECASE)
# v69: the labour rito only with an employment relationship in the text -- «lavori» (an appalto's), «impresa edile»,
# «lavori straordinari» (a condominium's) never; the rule's markers and the model's answer alike
EMPLOYMENT_RE = re.compile(
    r"\blavorat(?:ore|rice|ori|rici)\b|\bdatore di lavoro\b|\bdatori di lavoro\b|\blicenziament\w*|\blicenziat\w*|"
    r"\bretribuzion\w*|\bstipendi\w*|\brapporto di lavoro\b|\bcontratt\w* di lavoro\b|\blavoro subordinato\b|"
    r"\bsubordinazion\w*|\btrattamento di fine rapporto\b|\btfr\b|\bc\.?c\.?n\.?l\.?\b|\bgiudice del lavoro\b|"
    r"\brito del lavoro\b|\bdemansionament\w*|\bmobbing\b|\binail\b|\binfortunio sul lavoro\b|\bdimission\w*|"
    r"\bassunt[oaie]\b|\bassunzione\b", re.IGNORECASE)


def employment(text: str) -> bool:
    """v69: the text tells an employment relationship (EMPLOYMENT_RE)."""
    return bool(EMPLOYMENT_RE.search(text or ""))
# a specialised matter's marker weighs twice a civil one: «fornitura» told in a fraud's story is not the matter
_RITO_WEIGHT = {"civile": 1}
# an article cited by its code says the matter most (n6: «furto» told, «art. 1227 c.c.» the matter)
_CODE_CITES = (
    ("civile", re.compile(r"\bart(?:icol[oi]|\.)?\s*\d+[\w-]*\s+c\.\s?c\.(?!\w)|\bcodice civile\b|\bc\.\s?p\.\s?c\.(?!\w)",
                          re.IGNORECASE)),
    ("penale", re.compile(r"\bart(?:icol[oi]|\.)?\s*\d+[\w-]*\s+(?:c\.\s?p\.(?!\s?[cp]\.)|cp\b)|\bc\.\s?p\.\s?p\.(?!\w)|"
                          r"\bcodice (?:di procedura )?penale\b", re.IGNORECASE)),
    ("amministrativo", re.compile(r"\bc\.\s?p\.\s?a\.(?!\w)|\b241/1990\b|\bcodice del processo amministrativo\b",
                                  re.IGNORECASE)),
)
CODE_CITE_WEIGHT = 3
CIVIL_MIN = 2                       # civile only on two civil markers or more
# the order a tie goes by: the specialised matters first, civile last
_RITO_ORDER = ("penale", "amministrativo", "tributario", "fallimentare", "esecuzione", "lavoro", "famiglia", "civile")
_FAMILY_ASK_MIN = 1                 # famiglia: a family marker in the request, not only in the story


def rito_rule(message: str, upload: str = "") -> Tuple[str, str]:
    """(rito, the markers) from the text's markers: each matter's weighted count (_RITO_WEIGHT), the highest wins (ties
    by _RITO_ORDER);
    famiglia only when the request itself names it (a separation told as the story's background is not the matter);
    penale before the giudice di pace is penale_gdp. ("", "") with no marker."""
    text = f"{message or ''}\n{upload or ''}"
    counts = {rito: len(pattern.findall(text)) * _RITO_WEIGHT.get(rito, 2) for rito, pattern in _RITO_MARKERS}
    for rito, pattern in _CODE_CITES:
        counts[rito] += CODE_CITE_WEIGHT * len(pattern.findall(text))
    if counts["lavoro"] and not employment(text):       # v69: «lavori», «impresa edile» -- no employment, no lavoro
        counts["lavoro"] = 0
    if counts["famiglia"]:
        family_pattern = dict(_RITO_MARKERS)["famiglia"]
        if len(family_pattern.findall(ask_part(message or upload or ""))) < _FAMILY_ASK_MIN:
            counts["famiglia"] = 0
    best = max(_RITO_ORDER, key=lambda r: (counts[r], -_RITO_ORDER.index(r)))
    if not counts[best]:
        return "", ""
    if best == "civile" and counts["civile"] < CIVIL_MIN:
        return "", ""
    rito = "penale_gdp" if best == "penale" and _GDP_RE.search(text) else best
    shown = ", ".join(f"{r} {n}" for r, n in sorted(counts.items(), key=lambda x: -x[1]) if n)
    return rito, shown


# --- the client and its role --------------------------------------------------------------------------------------------

# v65: role and stage from current-status sentences first («attualmente persona offesa», «fase delle indagini
# preliminari», «non ha ancora depositato la comparsa»); a future or hypothetical mention («potrà costituirsi parte
# civile», «qualora proponesse appello») never sets the role, nor a stage but the pre_filing (an intention is its marker)
_CURRENT_WORDS = (r"attualmente|al momento|ad oggi|allo stato(?: attuale)?|tuttora|in questa fase|nella fase|"
                  r"fase (?:delle|del|della|di)|in fase di|pende(?:nte)?|in corso|non ha ancora|non è ancora|non "
                  r"sono ancora|non si è ancora")
_CURRENT_RE = re.compile(rf"\b(?:{_CURRENT_WORDS})\b", re.IGNORECASE)
_HYPOTHETICAL_RE = re.compile(r"\b(?:potr(?:à|ebbe|anno|ebbero|ei|emmo)|dovr(?:à|ebbe|anno|ebbero)|vorr(?:à|ebbe|anno)|"
                              r"intend\w*|qualora|eventualmente|eventual\w*|nel caso in cui|laddove|ove|se\s+(?:dovesse|"
                              r"volesse|intendesse|decidesse|decide|vuole|intende)|si\s+riserva\w*|valut\w+\s+(?:se|di|"
                              r"l['’]opportunità)|pensa\s+di|medita\s+di|in\s+ipotesi|ipotizz\w*|potenzial\w*)\b",
                              re.IGNORECASE)
_FUTURE_VERB_RE = re.compile(r"\b\w+(?:erà|irà|arà|eranno|iranno|aranno|erebbe|irebbe|arebbe)\b", re.IGNORECASE)
HYPOTHETICAL_REACH = 80


def hypothetical(text: str, start: int, end: int) -> bool:
    """v65: the match text[start:end] is a future or hypothetical mention -- a modal, «qualora», «eventualmente» … in
    its clause before it (HYPOTHETICAL_REACH characters, no stop between) or inside it, or a future or conditional verb
    in it."""
    clause = max(text.rfind(c, 0, start) for c in ".;:!?\n") + 1
    return bool(_HYPOTHETICAL_RE.search(text[max(clause, start - HYPOTHETICAL_REACH):end])
                or _FUTURE_VERB_RE.search(text[start:end]))


def _actual(pattern: "re.Pattern", text: str) -> Optional["re.Match"]:
    """The first match of the pattern that is not a future or hypothetical mention."""
    return next((m for m in pattern.finditer(text or "") if not hypothetical(text, m.start(), m.end())), None)


def current_text(text: str) -> str:
    """v65: the text's current-status sentences (_CURRENT_RE), joined."""
    return "\n".join(s for s in sentences(text) if _CURRENT_RE.search(s))


_ROLE_WORDS = (
    ("imputato", r"imputat\w*|indagat\w*|accusat\w*|querelat\w*"),
    ("parte_civile", r"parte civile"),
    ("persona_offesa", r"persona offesa|persone offese|vittim\w*|querelant\w*|denunciant\w*"),
    ("attore", r"attor[ei]|attric\w*|ricorrent\w*|opponent\w*"),
    ("convenuto", r"convenut\w*|resistent\w*|intimat\w*|oppost[oa]"),
    ("terzo", r"terz[oi] (?:chiamat\w*|pignorat\w*|interveni\w*|opponent\w*)"),
)
_SELF_REF = (r"(?:\b(?:sono|sei|siamo|come|quale|in qualit[àa] di|nelle vesti di)\s+|\bvest[ei]\s+(?:di|del|della|dello|"
             r"dell['’])\s*|\bassum\w*\s+(?:la\s+)?difesa\s+(?:di|del|della|dell['’])\s*)")
_LAWYER_RE = re.compile(_SELF_REF + r"(?:l['’]\s*|il\s+|la\s+|lo\s+)?(?:avvocat[oa]\s+)?(?:legale|difensore|avvocat[oa]|"
                        r"patrocinatore|consulente)\s+(?:di fiducia\s+)?(?:del|della|dello|dell['’]|dei|degli|delle|di|dl)"
                        r"\s*((?:[^\s.,;:]+\s*){1,6})", re.IGNORECASE)
_NAME_WORD = r"[A-ZÀ-Ü][\w'’-]+"
_MY_CLIENT = r"(?:mi[oa]|nostr[oa])\s+(?:cliente|assistit[oa])"
_MY_CLIENT_NAMED_RE = re.compile(rf"\b{_MY_CLIENT},?\s+(?:(?:il\s+)?(?:sig\.(?:ra)?|signor[ae]?|dott\.(?:ssa)?)\s+)?"
                                 rf"([A-ZÀ-Ü][\w'’-]+(?:\s+[A-ZÀ-Ü][\w'’-]+)?)")
_HIS_LAWYER_RE = re.compile(r"\b(?:sono|siamo)\s+(?:il|la|i)\s+(?:su[oa]|loro)\s+(?:avvocat\w*|legal\w*|difensor\w*)\b",
                            re.IGNORECASE)
_EXAM_CLIENT_RE = re.compile(
    _SELF_REF + r"(?:l['’]\s*|il\s+|la\s+|lo\s+)?(?:avvocat[oa]\s+)?" +
    rf"(?:legale|difensore|avvocat[oa]|patrocinatore)\s+(?:di fiducia\s+)?(?:del|della|dello|dell['’]|di|dl)\s*"
    rf"(?:(?:sig\.(?:ra)?|signor[ae]?|dott\.(?:ssa)?|società|ditta)\s+)?(?:(?:[a-zà-ù]+\s+){{0,2}})?({_NAME_WORD}"
    rf"(?:\s+{_NAME_WORD})?)")


def _role_hit(text: str) -> Tuple[str, int]:
    """(the first role word's role, where that word ends) in the text, ("", -1) without one."""
    low = _low(text)
    hits = []
    for role, pattern in _ROLE_WORDS:
        m = re.search(rf"\b(?:{pattern})", low)
        if m:
            hits.append((m.start(), m.end(), role))
    if not hits:
        return "", -1
    start, end, role = min(hits)
    return role, end


def _role_of_words(text: str) -> str:
    return _role_hit(text)[0]


def _said_of(name: str, text: str) -> Tuple[str, str]:
    """(role, the words) the text gives the client -- by its name, its name's first word, or «il mio cliente»: charged by the PM, accused, arrested or condemned -> imputato;
    a querela or a denuncia of its own -> persona_offesa; a parte civile -> parte_civile; suing -> attore; sued, served
    with a citazione, an injunction or a precetto -> convenuto; a third party called in -> terzo; wanting to act ->
    attore. ("", "") when it says none."""
    words = (name or "").split()
    refs = ([re.escape(name)] if name else []) + ([re.escape(words[0])] if len(words) > 1 else [])
    if re.search(rf"\b{_MY_CLIENT}\b", text or "", re.IGNORECASE):
        refs.append(_MY_CLIENT)
    for n in refs:
        found = _said_of_ref(n, text)
        if found[0]:
            return found
    return "", ""


def _said_of_ref(n: str, text: str) -> Tuple[str, str]:
    """_said_of for one way of naming the client (a regex)."""
    checks = (
        ("imputato", rf"\b(?:p\.?\s?m\.?|pubblico ministero|procura(?: della repubblica)?)\b[^.]{{0,30}}?\b(?:contesta|"
                     rf"imputa|ha contestato|addebita|iscrive|ha iscritto|chiede il rinvio a giudizio|esercita l['’]azione"
                     rf" penale)\w*\s+(?:a\s+|di\s+|nei confronti di\s+)?(?:[^.]{{0,20}}?){n}\b"),
        ("imputato", rf"\b(?:imputat\w*|indagat\w*)\s+(?:sig\.\s+)?{n}\b"),
        ("imputato", rf"\b{n}\b,?\s+(?:già\s+)?(?:imputat|indagat|accusat|arrestat|rinviat\w* a giudizio|condannat|"
                     rf"querelat|denunciat)\w*"),
        ("imputato", rf"\b{n}\b[^.]{{0,60}}?\b(?:ricev\w*|notificat\w*)\s+(?:un|una|l['’])?\s*(?:avviso|informazione) di "
                     rf"garanzia|\b{n}\b[^.]{{0,80}}?\b(?:p\.?\s?m\.?|pubblico ministero|procura)\s+(?:lo|la|li|le)\s+"
                     rf"(?:indaga|accusa|imputa|ha iscritt\w*|iscrive)"),
        ("imputato", rf"\b{n}\b[^.]{{0,50}}?\b(?:è|viene|vengono|veniva|è stat[oa]|sono stati|era stat[oa]|viene poi|"
                     rf"risulta)\s+(?:poi\s+)?(?:imputat|indagat|accusat|arrestat|fermat|condannat|rinviat\w* a "
                     rf"giudizio|trat\w* in arresto|denunciat|querelat|iscritt\w* nel registro)\w*"),
        ("parte_civile", rf"\b{n}\b[^.]{{0,60}}?\bcostitu\w*\s+parte civile|\bcostituit\w*\s+parte civile[^.]{{0,30}}"
                         rf"\b{n}\b"),
        ("persona_offesa", rf"\b{n}\b[^.]{{0,40}}?\b(?:(?:ha|aveva)\s+)?(?:sporto|presentato|propone|proposto|presenta|"
                           rf"sporge|deposita|depositato)\s+(?:formale\s+)?(?:querela|denuncia)\b|\b{n}\b\s+(?:querela|"
                           rf"denuncia)\b"),
        ("convenuto", rf"\b{n}\b[^.]{{0,40}}?\b(?:viene|è stat[oa]|veniva|era stat[oa])\s+(?:citat|convenut|evocat)\w*|"
                      rf"\b(?:notific\w*)\s+(?:a|al|alla|all['’])\s*(?:sig\.\s+)?{n}\b[^.]{{0,60}}?(?:atto di citazione|"
                      rf"citazione|ricorso|decreto ingiuntivo|precetto)|\b{n}\b[^.]{{0,30}}?\b(?:riceve|ha ricevuto|"
                      rf"riceveva|si vede notificare|si è vist[oa] notificare)\s+(?:la notifica (?:di|del|dell['’])\s*)?"
                      rf"(?:un|una|l['’]|il)?\s*(?:atto di citazione|citazione|decreto ingiuntivo|ricorso|atto di "
                      rf"precetto|precetto)"),
        ("attore", rf"\b{n}\b[^.]{{0,40}}?\b(?:cita|conviene|ha citato|ha convenuto|evoca|ha evocato)\s+in giudizio"),
        ("convenuto", rf"\b{n}\b[^.]{{0,40}}?\b(?:è stat[oa]|viene|veniva|era stat[oa])\s+notificat[oa]\s+(?:un|una|il|"
                      rf"l['’]|la)?\s*(?:atto di citazione|citazione|ricorso|decreto ingiuntivo|atto di precetto|precetto|"
                      rf"pignoramento)|\b{n}\b[^.]{{0,30}}?\b(?:riceve|ha ricevuto|riceveva)\s+(?:una|la|un['’]|la "
                      rf"lettera di)?\s*(?:diffida|messa in mora|richiesta di pagamento)"),
        ("terzo", rf"\b{n}\b[^.]{{0,40}}?\bterz[oa] (?:chiamat|pignorat|interveni)\w*|\bpignorat\w*[^.]{{0,120}}?\b"
                  rf"(?:appartiene|apparteneva|di proprietà|intestat\w*)\s+(?:a|di)\s+{n}\b"),
        ("attore", rf"\b{n}\b[^.]{{0,40}}?\b(?:intende|vuole|vorrebbe|desidera)\s+(?:agire|ottenere|chiedere|recuperare|"
                   rf"riottenere|essere risarcit\w*|il risarcimento|proporre|far valere)"),
    )
    for i, (role, pattern) in enumerate(checks):
        intention = i == len(checks) - 1               # «intende agire»: the intention is that check's own marker
        compiled = re.compile(pattern, re.IGNORECASE)
        m = compiled.search(text or "") if intention else _actual(compiled, text or "")   # v65: never a hypothesis
        if m:
            return role, _flat(m.group(0))[:80]
    return "", ""


def _current_role(name: str, text: str) -> Tuple[str, str]:
    """v65: (role, the words) a current-status sentence gives the client («Caia, attualmente persona offesa …», «il
    mio cliente non ha ancora depositato la comparsa»): the client named before the marker and the role word after it,
    or the client the subject of a comparsa not yet filed; ("", "") with none."""
    words = (name or "").split()
    refs = ([re.escape(name)] if name else []) + ([re.escape(words[0])] if len(words) > 1 else []) + [_MY_CLIENT]
    roles = "|".join(p for _, p in _ROLE_WORDS)
    for sentence in sentences(current_text(text)):
        for n in refs:
            m = re.search(rf"\b{n}\b[^.;]{{0,40}}?\b(?:{_CURRENT_WORDS})\b[^.;]{{0,30}}?\b(?:è\s+|come\s+|in\s+qualità\s+di\s+)?"
                          rf"({roles})", sentence, re.IGNORECASE)
            if m and not hypothetical(sentence, m.start(1), m.end(1)):
                role = _role_of_words(m.group(1))
                if role:
                    return role, _flat(m.group(0))[:80]
            m = re.search(rf"\b{n}\b[^.;]{{0,40}}?\bnon (?:ha|si è) ancora (?:depositat\w*|notificat\w*|costituit\w*)"
                          rf"(?:\s+(?:la\s+)?(?:comparsa\w*(?:\s+di\s+(?:costituzione|risposta))?|in giudizio))?",
                          sentence, re.IGNORECASE)
            if m and re.search(r"comparsa|in giudizio|costituit", m.group(0), re.IGNORECASE):
                return "convenuto", _flat(m.group(0))[:80]
    return "", ""


# the client acts: the ask is to bring the claim (non-penal matters: the attore, the ricorrente)
_ACT_ASK_RE = re.compile(
    r"\b(?:come|per)\s+(?:impugnare|agire|ottenere|recuperare|riottenere|far valere|chiedere|citare|ricorrere|"
    r"contestare|insinuarsi)\b|\bcome\s+(?:ottengo|agisco|impugno|recupero|ricorro|otteniamo|agiamo)\b|"
    r"\b(?:vuole|intende|vorrebbe)\s+(?:agire|recuperare|ottenere|riottenere|impugnare|far valere)\b|\bquali azioni\b",
    re.IGNORECASE)
# another party's querela or denuncia, or the persona offesa named apart: the client is the accused
_OTHER_QUERELA_RE = r"\b({name})\b[^.]{{0,30}}?\b(?:(?:ha|aveva)\s+)?(?:sporto|presentato|sporge|presenta|deposita)\s+(?:formale\s+)?(?:denuncia|querela)"


def client_rule(message: str, upload: str = "", penal: bool = False) -> Dict[str, str]:
    """{name, role, why}: the client the text names -- the user's own words (interactive_flow.stated: «sono l'avvocato
    di X», «assisto X», «difendo l'imputato»), the exam's («assunte le vesti del legale di X», fact_signals.client_from)
    -- and its role: the role word in the lawyer's phrase («difensore dell'imputato Tizio», «legale della persona
    offesa»), else what the text says of the client (_said_of)."""
    from . import fact_signals, interactive_flow
    text = f"{message or ''}\n{upload or ''}".strip()
    said = interactive_flow.stated(message or "") if message else {}
    name = said.get("client") or ""
    role_word = said.get("client_role") or ""
    if not name:
        m = _EXAM_CLIENT_RE.search(text)
        name = m.group(1) if m and m.group(1).split()[0] not in ("Il", "La", "Lo", "Codice", "Tribunale") else ""
    if not name:
        m = _MY_CLIENT_NAMED_RE.search(text)
        name = m.group(1) if m else ""
    if not name:
        name = fact_signals.client_from(text)
    if not name and _HIS_LAWYER_RE.search(text):
        # «Sono il suo avvocato»: the one person the text names
        from .query_router import party_kinds
        named = [p for p, kind in party_kinds(text).items() if kind in ("name", "school", "company")]
        name = named[0].title() if len(named) == 1 else ""
    role, why = _current_role(name, text)          # v65: a current-status sentence first
    why = f"«{why}» (stato attuale)" if role else ""
    lawyer = _LAWYER_RE.search(text)
    if role_word and not role:
        role, why = _role_of_words(role_word), f"«{role_word}»"
    if not role and lawyer:
        role, end = _role_hit(lawyer.group(1))
        phrase = _flat(lawyer.group(0)[:lawyer.start(1) - lawyer.start(0)] + _flat(lawyer.group(1))[:end])
        why = f"«{phrase}»" if role else ""
    if not role:
        role, said_why = _said_of(name, text)
        why = f"«{said_why}»" if role else ""
    if name and not role:
        # the generic role in a «Sei l'avvocato del convenuto»-style phrase, when the name came from it
        role = _role_of_words(name)
    constituted = _actual(_CONSTITUTED_RE, text)       # v65: «potrà costituirsi parte civile» never does
    if role == "persona_offesa" and constituted:      # v62: the persona offesa who joined the trial is the parte civile
        role, why = "parte_civile", f"«{_flat(constituted.group(0))}»"
    has_client = bool(name or lawyer or re.search(rf"\b{_MY_CLIENT}\b", text, re.IGNORECASE))
    if not role and penal and name:
        other = re.search(_OTHER_QUERELA_RE.format(name=_NAME_WORD), text)
        if other and other.group(1).lower() not in name.lower():
            role, why = "imputato", f"«{_flat(other.group(0))[:60]}»: la querela è di un'altra parte"
    if not role and not penal and has_client:
        m = _ACT_ASK_RE.search(ask_part(message or ""))
        if m:
            role, why = "attore", f"«{_flat(m.group(0))}»: il cliente agisce"
    return {"name": name, "role": role, "why": why}


_CONSTITUTED_RE = re.compile(r"\bcostitu\w*\s+(?:\w+\s+){0,2}parte civile\b", re.IGNORECASE)
_SIDES = {"imputato": "accused", "convenuto": "accused", "persona_offesa": "victim", "parte_civile": "victim",
          "attore": "victim", "terzo": "terzo"}


def side_of(role: str) -> str:
    """The client's side from its role, in fact_signals.client_side's words (accused | victim), terzo, or ""."""
    return _SIDES.get(role or "", "")


# --- the stage and the prior decisions -----------------------------------------------------------------------------------

_STAGE_MARKERS = (
    ("esecuzione", re.compile(
        r"\bpignorament\w*|\bpignorat\w*|\batto di precetto\b|\bprecetto\b|\besecuzione forzata\b|\bespropriazione "
        r"(?:forzata|immobiliare|mobiliare|presso terzi)\b|\bopposizione (?:all['’]esecuzione|agli atti esecutivi|"
        r"al precetto)\b|\bufficiale giudiziario\b|\bsentenz\w* (?:passat\w* in giudicato|definitiv\w*|irrevocabil\w*)|"
        r"\bpassat\w* in giudicato\b|\bordine di esecuzione\b|\besecuzione della pena\b|\baffidamento in prova\b|"
        r"\bmisure alternative alla detenzione\b|\bmagistrato di sorveglianza\b|\btribunale di sorveglianza\b|"
        r"\battuazione (?:dell['’]|della|del)\s*(?:ordinanza|sentenza|provvedimento|decreto)\b|\bottemperanza\b|"
        r"\bvendita (?:all['’]asta|forzata)\b", re.IGNORECASE)),
    ("impugnazione", re.compile(
        r"\bappell\w*|\bimpugnazion\w*|\bimpugna\w*\s+(?:la|il|l['’])\s*(?:sentenza|decisione|ordinanza|pronuncia|"
        r"condanna|decreto penale)\b|\bricors\w* (?:per|in) cassazione\b|\bricorrere in cassazione\b|\bgravame\b|"
        r"\bmezz\w* di impugnazione\b|\breclamo\b|\bsentenza di primo grado\b|\bin primo grado\b|"
        r"\b(?:è|sono|viene|vengono|veniva|venivano|è stat[oa]|sono stat[ie]) (?:condannat|assolt|prosciolt)\w*|"
        r"\bcon sentenza\b[^.]{0,80}?\b(?:condann|assol|rigett|accol|respin|dichiar)\w*|\b(?:il tribunale|la corte|il "
        r"giudice|il gip|il gup|il tar)\b[^.]{0,60}?\b(?:ha |aveva )?(?:condannat|assolt|rigettat|respint|accolt|"
        r"dichiarat\w* inammissibil)\w*|\b(?:rigett|respin)\w* (?:la domanda|il ricorso|l['’]opposizione)\b",
        re.IGNORECASE)),
    ("primo_grado", re.compile(
        r"\budienza preliminare\b|\brinvi\w* a giudizio\b|\bdecreto che dispone il giudizio\b|\bcitazione diretta a "
        r"giudizio\b|\bgiudizio immediato\b|\bdibattiment\w*|\bprocesso (?:penale )?(?:a carico|in corso|pendente)\b|"
        r"\bprima udienza\b|\budienza (?:di comparizione|fissata|di prima comparizione)\b|\bcostituit\w* in giudizio\b|"
        r"\b(?:notificat\w*|ricevut\w*|riceve|ha ricevuto|si vede notificare)\s+(?:un|una|l['’]|il)?\s*(?:atto di "
        r"citazione|citazione|decreto ingiuntivo|ricorso ex)|\batto di citazione (?:notificat|ricevut)\w*|"
        r"\b(?:è|viene) (?:stat[oa] )?(?:citat|convenut)\w* in giudizio\b|\bgiudizio pendente\b|\bcausa pendente\b|"
        r"\bcomparsa di (?:costituzione|risposta)\b|\bopposizione a decreto ingiuntivo\b|\bdecreto penale di "
        r"condanna\b|\bprocesso\b[^.]{0,40}?\ba carico di\b|\bcostitu\w*\s+(?:\w+\s+){0,2}parte civile\b|"
        r"\bprocesso penale\b[^.]{0,30}?\bimputat\w*|\ball['’]udienza\b|\bcomparsa\b|"
        r"\bfissat\w* (?:la |l['’])?udienza\b|\b(?:ha|hanno|aveva) (?:citato|convenuto|evocato) in giudizio\b|"
        r"\bsi (?:è|sono) costituit\w*", re.IGNORECASE)),
    ("indagini", re.compile(
        r"\b(?:il|la)\s+(?:p\.?\s?m\.?|pubblico ministero|procura)\b[^.]{0,20}?\b(?:contesta|indaga|iscrive|ha "
        r"iscritto|procede)\w*|\bindagat\w*|\bindagini (?:preliminari|in corso|svolte|condotte)\b|\bindagini\b|"
        r"\bavviso (?:di conclusione delle indagini|ex art\.? ?415[- ]?bis)\b|\b415[- ]?bis\b|\bregistro (?:degli "
        r"indagati|delle notizie di reato)\b|\b(?:informazione|avviso) di garanzia\b|\bperquisizion\w*|\bsequestr\w*|"
        r"\barrestat\w*|\bfermat[oa] (?:poco dopo|dalla polizia|dai carabinieri|da una pattuglia)|\bin flagranza\b|"
        r"\b(?:(?:ha|aveva|hanno|avevano)\s+)?(?:sporto|presentato|proposto|sporge|presenta|sporgono)\s+(?:formale\s+)?"
        r"(?:denuncia(?:-querela)?|querela)\b|\bquerela (?:presentata|sporta|depositata)\b|"
        r"\b(?:denuncia|querela) (?:dopo|nei confronti|contro)\b",
        re.IGNORECASE)),
    ("pre_filing", re.compile(
        r"\b(?:intende|vuole|vorrebbe|desidera|intendono|vogliono)\s+(?:agire|proporre|presentare|sporgere|ottenere|"
        r"chiedere|recuperare|riottenere|essere risarcit|far valere|separarsi|citare|convenire)\w*|\bsi (?:rivolge|reca)"
        r"\s+(?:a|ad|da|dal|dalla|presso)\s+(?:un|una|il|la|lo|l['’]|suo|sua|proprio|propria)?\s*(?:\w+\s+)?"
        r"(?:avvocat\w*|legale|difensore|studio)|\bprima di agire\b|\bquali azioni\b|\bazioni (?:da intraprendere|"
        r"proponibili|esperibili)\b|\biniziative (?:da (?:assumere|intraprendere)|giudiziarie)\b|\bcontro chi può "
        r"agire\b|\bpuò agire\b|\bdiffid\w*", re.IGNORECASE)),
)
_STAGE_ORDER = ("esecuzione", "impugnazione", "primo_grado", "indagini", "pre_filing")   # the latest posture wins


def stage_rule(message: str, upload: str = "") -> Tuple[str, str]:
    """(stage, the words) from the text's markers, the latest posture first (esecuzione, impugnazione, primo_grado,
    indagini, pre_filing); ("", "") with none."""
    text = f"{message or ''}\n{upload or ''}"
    for where in (current_text(text), text):         # v65: the current-status sentences first
        for stage in _STAGE_ORDER:
            pattern = dict(_STAGE_MARKERS)[stage]
            m = pattern.search(where) if stage == "pre_filing" else _actual(pattern, where)
            if m:
                return stage, f"«{_flat(m.group(0))[:60]}»" + (" (stato attuale)" if where is not text else "")
    return "", ""


_COURT_RE = re.compile(
    r"\b(?:corte d['’]appello|corte di appello|corte (?:suprema )?di cassazione|cassazione|corte d['’]assise"
    r"(?: d['’]appello)?|tribunale(?: civile| penale| per i minorenni| di sorveglianza| amministrativo regionale)?|"
    r"giudice di pace|giudice (?:del lavoro|dell['’]esecuzione|monocratico|unico)|g\.?i\.?p\.?|g\.?u\.?p\.?|"
    r"t\.?a\.?r\.?|consiglio di stato)(?:\s+(?:di|del|della|dell['’])\s*[A-ZÀ-Ü]\w+)?", re.IGNORECASE)
_OUTCOMES = (
    ("condanna", r"condann\w*"), ("assoluzione", r"assol\w*|prosciolt\w*|proscioglimento"),
    ("rigetto", r"rigett\w*|respint\w*"), ("accoglimento", r"accolt\w*|accoglie\w*"),
    ("inammissibilità", r"inammissibil\w*"), ("non luogo a procedere", r"non luogo a procedere"),
    ("decreto penale di condanna", r"decreto penale di condanna"), ("decreto ingiuntivo", r"decreto ingiuntivo"),
    ("ordinanza", r"ordinanz\w*"), ("sentenza", r"sentenz\w*"),
)
_NOT_YET_RE = re.compile(r"\b(?:rischi\w*|potrebbe|potrebbero|eventual\w*|chiedere|ottenere|richiedere|vuole|intende|"
                         r"per (?:la|una)|al fine di)\b[^.]{0,25}$", re.IGNORECASE)
# v69: a date a term or an amount runs from -- the calculation needs one (else: a quesito, the term and its article)
_RELATIVE_DAY_RE = re.compile(r"\b(?:ieri|oggi|stamattina|stamani|l['’]altro ieri)\b|\b\d{1,3}\s+(?:giorni|settimane|mesi|"
                              r"anni)\s+fa\b|\b(?:dal|dall['’]anno|a partire dal|fino al|al)\s+\d{4}\b|"
                              r"\b\d{1,2}°\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|settembre|"
                              r"ottobre|novembre|dicembre)\b", re.IGNORECASE)     # «dal 2015», «1° marzo»


def decorrenza_date(text: str) -> bool:
    """v69: the text gives a date to count from -- a calendar date (_DATE_RE) or a day relative to today."""
    return bool(_DATE_RE.search(text or "") or _RELATIVE_DAY_RE.search(text or ""))


_DATE_RE = re.compile(r"\b\d{1,2}[./-]\d{1,2}[./-]\d{2,4}\b|\b\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|"
                      r"luglio|agosto|settembre|ottobre|novembre|dicembre)\s+\d{4}\b|\b(?:nel|del|in data)\s+\d{4}\b",
                      re.IGNORECASE)


# v71: a charge sentence -- «il PM contesta …», «imputato di … ai sensi dell'art. N», «chiede la condanna ai sensi …» --
# with the articles it names: its own entry («contestazione», with «articles»), the record's charge (posture.mark_sides)
_CHARGE_SENTENCE_RE = re.compile(r"\bcontest\w*|\bimput\w*|\bchied\w*\s+(?:la\s+)?condanna\b", re.IGNORECASE)


def prior_decisions(text: str) -> List[Dict[str, Any]]:
    """[{court, outcome, date}] the decisions the text states as given («il Tribunale di Alfa con sentenza del 3 marzo
    2021 ha condannato …», «vengono condannati»), one per sentence; a decision asked for or feared is not one. v71: a
    charge sentence naming its articles is a «contestazione» entry with them ({court, outcome, date, articles})."""
    from . import posture as posture_mod
    out = []
    for sentence in sentences(text):
        if _CHARGE_SENTENCE_RE.search(sentence):
            articles = posture_mod.sentence_articles(sentence)
            if articles:
                court = _COURT_RE.search(sentence)
                date = _DATE_RE.search(sentence)
                item = {"court": _flat(court.group(0)) if court else "", "outcome": posture_mod.CHARGE_OUTCOME,
                        "date": _flat(date.group(0)) if date else "", "articles": articles}
                if item not in out:
                    out.append(item)
                continue
        hit = None
        for outcome, pattern in _OUTCOMES:
            m = re.search(rf"\b(?:{pattern})", sentence, re.IGNORECASE)
            if m and not _NOT_YET_RE.search(sentence[:m.start()]):
                hit = (outcome, m)
                break
        if hit is None:
            continue
        court = _COURT_RE.search(sentence)
        date = _DATE_RE.search(sentence)
        item = {"court": _flat(court.group(0)) if court else "", "outcome": hit[0],
                "date": _flat(date.group(0)) if date else ""}
        if item not in out:
            out.append(item)
    return out[:5]


# --- the model, for the fields the rules leave empty ----------------------------------------------------------------------

_FIELD_HELP = {
    "deliverable": "deliverable: che cosa chiede -- parere (analisi di un caso concreto e strategia), atto (redigere "
                   "un atto o un documento), cenni (spiegare un istituto, senza fatti), quesito (un punto di diritto "
                   "con una risposta breve: un termine, una competenza, una pena), revisione_documento (esaminare il "
                   "documento caricato), calcolo (un importo da calcolare), ricerca_giurisprudenza (sentenze, "
                   "orientamenti della giurisprudenza), altro",
    "rito": "rito: la materia e il rito -- civile, lavoro, famiglia, esecuzione (esecuzione forzata), penale, "
            "penale_gdp (penale davanti al giudice di pace), amministrativo, tributario, fallimentare",
    "role": "role: la posizione della parte che l'avvocato assiste -- attore, convenuto, imputato (anche indagato), "
            "persona_offesa, parte_civile, terzo",
    "stage": "stage: la fase -- pre_filing (prima di agire o di un processo), indagini (indagini preliminari), "
             "primo_grado (giudizio di primo grado in corso), impugnazione (dopo una decisione, per impugnarla), "
             "esecuzione (esecuzione forzata o della pena)",
}
_ALLOWED = {"deliverable": DELIVERABLES, "rito": RITI, "role": ROLES, "stage": STAGES}
_MODEL_SYSTEM = (
    "Sei l'accoglienza di un assistente legale italiano: leggi la richiesta dell'utente e dici che cosa vuole, senza "
    "rispondere alla richiesta. Rispondi SOLO con un oggetto JSON con queste chiavi, ciascuna con uno dei valori "
    "elencati, oppure \"sconosciuto\" se il testo non lo dice:\n{fields}")


def _view(message: str, upload: str, history: Sequence[Dict[str, Any]]) -> str:
    message = message or ""
    if len(message) > MODEL_VIEW_HEAD + MODEL_VIEW_TAIL:
        message = f"{message[:MODEL_VIEW_HEAD]}\n[…]\n{message[-MODEL_VIEW_TAIL:]}"
    parts = [f"RICHIESTA:\n{message}"]
    if upload:
        parts.append(f"DOCUMENTO CARICATO (inizio):\n{_flat(upload)[:UPLOAD_VIEW]}")
    turns = [h for h in history or [] if isinstance(h, dict) and h.get("content")][-HISTORY_TURNS:]
    if turns:
        parts.append("CONVERSAZIONE PRECEDENTE:\n" + "\n".join(
            f"{h.get('role', '')}: {_flat(h['content'])[:HISTORY_CHARS]}" for h in turns))
    return "\n\n".join(parts)


def ask_model(message: str, upload: str, history: Sequence[Dict[str, Any]], fields: Sequence[str]) -> Dict[str, str]:
    """One call for the fields asked: {field: value} for the values it names among the allowed ones ({} on any failure,
    logged). Never a field it was not asked, never «sconosciuto»."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from .ai_chat import _call_chat
    system = _MODEL_SYSTEM.format(fields="\n".join(f"- {_FIELD_HELP[f]}" for f in fields))
    try:
        reply = _call_chat([SystemMessage(content=system), HumanMessage(content=_view(message, upload, history))],
                           max_tokens=MODEL_TOKENS, temperature=0)
    except Exception as exc:
        logger.warning("intake: the model call failed, the defaults: %s", exc)
        return {}
    m = re.search(r"\{.*\}", reply or "", re.DOTALL)
    try:
        raw = json.loads(m.group(0)) if m else {}
    except (ValueError, TypeError):
        logger.warning("intake: the model's reply is not JSON: %r", (reply or "")[:200])
        return {}
    out = {}
    for field in fields:
        value = str((raw or {}).get(field) or "").strip().lower().replace(" ", "_")
        if value in _ALLOWED[field] and value != UNKNOWN:
            out[field] = value
    return out


# --- the intake -------------------------------------------------------------------------------------------------------------

def route_of(result: Dict[str, Any], has_upload: bool = False) -> str:
    """What the API runs for the intake: the deliverable's route (ROUTES); an upload to fill is the document route, a
    review without an upload the plain answer; a parere on an out-of-corpus matter the rules found, the out-of-scope
    copy -- a question or a calculation on it keeps the plain answer, whose calculation gate reaches the tax
    calculators."""
    deliverable = result.get("deliverable") or "altro"
    if result.get("rito") in OUT_OF_CORPUS and (result.get("confidence") or {}).get("rito") == RULE \
            and deliverable == "parere":
        return OUT_OF_SCOPE
    if deliverable == "atto" and has_upload and (result.get("why") or {}).get("deliverable", "").startswith(
            "il documento caricato"):
        return DOCUMENT
    if deliverable == "revisione_documento" and not has_upload:
        return RAG
    return ROUTES.get(deliverable, RAG)


def doc_intent(result: Dict[str, Any]) -> str:
    """The document route's own intent for the intake (api._chat_impl's document branch): analyse, generate (the
    upload filled), defensive_draft (a parere on the upload, when the case-first flow is off or failed), or rag (the
    branch falls through to the routes after it)."""
    if result.get("route") == DOCUMENT and result.get("deliverable") == "atto":
        return "generate"
    if result.get("route") != CASE_FIRST and result.get("deliverable") == "parere":
        return "rag"                    # an out-of-corpus matter: the copy, after the branch
    if result.get("deliverable") == "parere" and result.get("case_in") != "upload":
        return "rag"                    # the case is the message: its own path, after the branch
    return DOC_INTENTS.get(result.get("deliverable") or "", "rag")


# v79 (C5): the text shows someone suing, serving or prosecuting the client -- the only ground for «convenuto»
_SUED_RE = re.compile(
    r"\batto\s+di\s+citazione|\bcitat[oaie]\s+(?:in\s+giudizio|davanti)|\b(?:ha|hanno|aveva|avevano)\s+(?:citato|convenuto)\b|"
    r"\bconvenut[oaie]\b|\b(?:fatto|fa|fare|intentat\w*|promoss\w*|avviat\w*)\s+(?:una\s+)?causa\b|\bagit\w*\s+in\s+giudizio|"
    r"\bnotificat\w*|\bnotifica\w*\s+(?:di\s+)?(?:un|una|il|la|l['’])|\bdecreto\s+ingiuntivo|\bingiunzion\w+\s+di\s+pagamento|"
    r"\bprecetto\b|\bpignorament\w*|\bsfratt\w*|\bricorso\s+(?:contro|nei\s+confronti)|\bimputat\w*|\bindagat\w*|"
    r"\brinviat\w*\s+a\s+giudizio|\b(?:ha|hanno)\s+(?:denunciato|querelato)\b|\bquerela\s+(?:contro|nei\s+confronti)",
    re.IGNORECASE)
CONVENUTO_UNSUPPORTED = "il modello: «convenuto», ma il testo non mostra un'azione contro il cliente (attore)"


def sued(text: str) -> bool:
    return bool(_SUED_RE.search(text or ""))


def detect(message: str, upload: str = "", history: Sequence[Dict[str, Any]] = (), *, model: bool = True,
           refers_to_upload: bool = False, upload_role: str = "") -> Dict[str, Any]:
    """The intake of one request (see the module's doc). message: the user's words; upload: the text of the document
    the request carries ("" without one); history: the conversation's earlier messages [{role, content}], read by the
    model only; model False: the rules and the defaults alone (scripts/intake_check.py --rules-only)."""
    message = message or ""
    upload = upload or ""
    sig = _signals(message)
    facts = narrates_facts(message, sig) or bool(upload and narrates_facts(upload))
    values: Dict[str, str] = {}
    confidence: Dict[str, str] = {}
    why: Dict[str, str] = {}

    def take(field: str, value: str, how: str, reason: str) -> None:
        if value and field not in values:
            values[field], confidence[field], why[field] = value, how, reason

    deliverable, reason = deliverable_rule(message, upload, refers_to_upload, sig)
    take("deliverable", deliverable, RULE, reason)
    rito, reason = rito_rule(message, upload)
    take("rito", rito, RULE, reason)
    client = client_rule(message, upload, penal=(rito or "").startswith("penale"))
    take("role", client["role"], RULE, client["why"])
    stage, reason = stage_rule(message, upload)
    case = values.get("deliverable") in CASE_DELIVERABLES or (not values.get("deliverable") and facts)
    if case:
        take("stage", stage, RULE, reason)

    wanted = []
    if "deliverable" not in values:
        wanted.append("deliverable")
    if case or "deliverable" not in values:
        wanted += [f for f in ("rito", "role", "stage") if f not in values]
    if model and wanted:
        answer = ask_model(message, upload, history, wanted) or {}
        for field in wanted:
            value = str(answer.get(field) or "")
            if field == "rito" and value == "lavoro" and not employment(f"{message}\n{upload}"):
                continue                # v69: the model's «lavoro» needs an employment relationship in the text
            if field == "role" and value == "convenuto" and not sued(f"{message}\n{upload}"):
                take(field, "attore", MODEL, CONVENUTO_UNSUPPORTED)      # v79 (C5): nobody acts against the client
                continue
            if value in _ALLOWED[field] and value != UNKNOWN:
                take(field, value, MODEL, "il modello")
    if "deliverable" not in values:
        default = ("revisione_documento" if upload and refers_to_upload and upload_role in ("", "document")
                   else "parere" if facts else "altro")
        take("deliverable", default, DEFAULT, "nessuna regola, nessuna risposta del modello")
    case = values["deliverable"] in CASE_DELIVERABLES
    take("stage", "pre_filing" if case else UNKNOWN, DEFAULT, "il testo non dice la fase")
    take("rito", UNKNOWN, DEFAULT, "l'area del caso la decide (frame)")
    take("role", UNKNOWN, DEFAULT, "la logica del cliente la decide")
    if not case and confidence.get("role") != RULE:
        values["role"], confidence["role"] = UNKNOWN, DEFAULT
    from .query_router import party_kinds
    try:
        kinds = party_kinds(f"{message}\n{upload}")
    except Exception:
        kinds = {}
    result = {
        "deliverable": values["deliverable"],
        "requested_blocks": requested_blocks(message),
        "named_articles": named_articles(message, client["name"]),        # v65: with their verb and use
        "rito": values["rito"], "role": values["role"], "stage": values["stage"],
        "prior_decisions": prior_decisions(f"{message}\n{upload}"),
        "client": {"name": client["name"], "side": side_of(values["role"]),
                   "parties": [p for p, kind in kinds.items() if kind != "role" and p != "io"][:8]},
        "facts": facts,
        "upload": bool(upload),
        # where the case is: the upload when the request is about it or the message alone narrates none (today's
        # _interactive_upload_case), else the message (a narrative beside an unrelated attachment)
        "case_in": "upload" if upload and values["deliverable"] == "parere" and (
            refers_to_upload or not _narrative(message)) else "message",
        "confidence": {f: confidence[f] for f in FIELDS},
        "why": {f: why.get(f, "") for f in FIELDS},
    }
    result["route"] = route_of(result, bool(upload))
    return result


def safe_detect(message: str, upload: str = "", history: Sequence[Dict[str, Any]] = (), **kwargs) -> Dict[str, Any]:
    """detect, never raising: on a failure (logged), the defaults with today's narrative check deciding parere."""
    try:
        return detect(message, upload, history, **kwargs)
    except Exception as exc:
        logger.error("intake failed, the defaults: %s", exc, exc_info=True)
        try:
            narrative = _narrative(message)
        except Exception:
            narrative = False
        deliverable = "parere" if narrative else "altro"
        result = {"deliverable": deliverable, "requested_blocks": [], "named_articles": [], "rito": UNKNOWN,
                  "role": UNKNOWN,
                  "stage": "pre_filing" if narrative else UNKNOWN, "prior_decisions": [],
                  "client": {"name": "", "side": "", "parties": []}, "facts": narrative, "upload": bool(upload),
                  "case_in": "upload" if upload and not narrative else "message",
                  "confidence": {f: DEFAULT for f in FIELDS}, "why": {f: "intake fallita" for f in FIELDS},
                  "error": f"{type(exc).__name__}: {exc}"[:200]}
        result["route"] = route_of(result, bool(upload))
        return result


def request_intake(message: str, upload: str = "", history: Sequence[Dict[str, Any]] = (), *,
                   refers_to_upload: bool = False, upload_role: str = "") -> Dict[str, Any]:
    """v73: the request's intake as the API reads it -- one function for the API (api._request_intake), the case-first
    flow when no intake is handed to it (interactive_flow.take_intake) and the sims (interactive_flow.request_case_first):
    safe_detect on the message, the upload's text, the last HISTORY_TURNS messages, whether the message is about the
    upload and the upload's role; the [VERBOSE:intake] line logged here, timed."""
    import time
    t0 = time.time()
    result = safe_detect(message, upload or "", list(history or [])[-HISTORY_TURNS:], refers_to_upload=refers_to_upload,
                         upload_role=upload_role or "")
    vlog("intake", log_view(result), (time.time() - t0) * 1000)
    return result


def log_view(result: Dict[str, Any]) -> Dict[str, Any]:
    """The [VERBOSE:intake] line: the fields, their confidence and reasons, the blocks, the route."""
    return {k: result.get(k) for k in ("deliverable", "route", "rito", "role", "stage", "requested_blocks",
                                        "named_articles", "prior_decisions", "client", "facts", "upload", "case_in", "confidence", "why",
                                        "error")
            if k in result}


# --- the atto's context -------------------------------------------------------------------------------------------------

def case_record(message: str, upload: str = "", lang: str = "it") -> str:
    """The case-first record of a narrated case for an atto's drafting (the generation route's context): the first read
    (case_read.read, one bounded model call) as lines -- the client, the goal, the facts, what the case does not say; ""
    on any failure or no read (logged)."""
    from . import case_read
    try:
        read = case_read.read(upload or message, message if upload else "", lang)
    except Exception as exc:
        logger.warning("intake: the case record failed, the atto without it: %s", exc)
        return ""
    if not read:
        return ""
    lines = []
    client = read.get("client") or {}
    if client.get("name"):
        lines.append(f"Cliente: {client['name']}" + (f" ({client['role']})" if client.get("role") else ""))
    if read.get("goal"):
        lines.append(f"Obiettivo: {read['goal']}")
    if read.get("facts"):
        lines.append("Fatti:\n" + "\n".join(f"- {f}" for f in read["facts"]))
    if read.get("unknown"):
        lines.append("Il caso non dice:\n" + "\n".join(f"- {u}" for u in read["unknown"]))
    return "\n".join(lines)


# --- downstream: the defaults the analysis decides --------------------------------------------------------------------------

def resolve(result: Optional[Dict[str, Any]], analysis: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The intake with its default role and rito filled as the existing logic reads them once the case is analysed: the
    rito from the frame's area (interactive_flow.frame_families), the role from the client's side
    (fact_signals.client_side: accused -> imputato in penale, convenuto otherwise; victim -> persona_offesa in penale,
    attore otherwise). Confidence stays default; a rule's or the model's value is never changed."""
    if not result:
        return result
    from . import fact_signals, interactive_flow
    out = dict(result)
    out["confidence"] = dict(result.get("confidence") or {})
    families = interactive_flow.frame_families(analysis or {})
    family = families[0] if families else ""
    if out.get("rito") == UNKNOWN and family in ("civile", "penale", "amministrativo"):
        out["rito"] = family
    if out.get("role") == UNKNOWN:
        facts = ((analysis or {}).get("fact_signals") or {}).get("facts") or {}
        side = fact_signals.client_side(str(facts.get("client_side") or (analysis or {}).get("parte_assistita") or ""))
        penal = (out.get("rito") or "").startswith("penale") or family == "penale"
        role = {"accused": "imputato" if penal else "convenuto",
                "victim": "persona_offesa" if penal else "attore"}.get(side, "")
        if role == "convenuto" and not sued(str((analysis or {}).get(interactive_flow.CASE_KEY) or "")):
            role = "attore"             # v79 (C5): «convenuto» only when the text shows an action against the client
        if role:
            out["role"] = role
            out["client"] = {**(out.get("client") or {}), "side": side}
    return out
