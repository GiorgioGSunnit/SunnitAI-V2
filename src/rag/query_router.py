"""ROUTER_ENABLED: which path a chat request takes, by the kind of request it is (26 Oct, the source-tiered
retrieval architecture).

  narrative  a fact pattern -- several sentences, two or more parties (named, Tizio/Caio, companies, roles,
             the user), a sequence of events -- with a request for action, an opinion, a strategy or a
             parere ("il candidato", "quali azioni", "strategia", "difesa"). The strategy pipeline
             (run_case_analysis_pipeline), whose rulings may act first as a map (HYBRID_NARRATIVE_ENABLED).
  direct     a question about a norm, an institution, a term or a procedure, without a fact pattern. The
             chat's graph, its articles from primary documents first (PRIMARY_FIRST_ENABLED).
  ccnl       a collective agreement named, or an employment question that needs one. Tagged only: today's
             path.
  document   a question about an uploaded document: its own path, unchanged.

Deterministic first (signals); one short LLM call breaks a tie only when the signals disagree. Every
decision is logged with its reasons ("route_chosen"). The routes the chat had before -- defensive draft,
calculation, document, generation -- keep precedence where they apply today: the router only adds where
they did not decide.
"""

import logging
import re
import threading
from typing import Any, Dict, List, Optional

from . import feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

ROUTER_FLAG = "ROUTER_ENABLED"
ROUTES = ("narrative", "direct", "ccnl", "document")


def enabled() -> bool:
    return feature_flags.enabled(ROUTER_FLAG)


# --- signals ----------------------------------------------------------------------------------------

_SCHOOL_NAMES = r"tizi[oa]|cai[oa]|sempronio|sempronia|mevio|mevia|filano|filana|calpurni[oa]|primo|secondo"
_COMPANY_RE = re.compile(r"\b[A-Z][\w&'.-]*(?:\s+[A-Z][\w&'.-]*)*\s+(?:s\.?r\.?l\.?s?|s\.?p\.?a\.?|s\.?n\.?c\.?|"
                         r"s\.?a\.?s\.?|s\.?c\.?a\.?r\.?l\.?|soc\.\s*coop)\b", re.IGNORECASE)
_SCHOOL_RE = re.compile(rf"\b({_SCHOOL_NAMES})\b", re.IGNORECASE)
_ROLE_RE = re.compile(
    r"\b(?:il|la|lo|l'|i|gli|le|un|una|uno|suo|sua|mio|mia|miei|nostro|nostra|del|della|dal|dalla|al|alla)\s*"
    r"(venditor\w*|acquirent\w*|compratore|locator\w*|conduttor\w*|inquilin\w*|proprietari\w*|vicin\w*|"
    r"datore di lavoro|dipendent\w*|lavorator\w*|committent\w*|appaltator\w*|impresa|societ[àa]|ditta|banca|"
    r"assicurazion\w*|compagnia|comune|amministrazion\w*|ente|medic\w*|ospedal\w*|struttura|condomini\w*|"
    r"amministratore|creditor\w*|debitor\w*|fideiussor\w*|mutuatari\w*|cliente|fornitor\w*|gestor\w*|"
    r"marito|moglie|coniuge|figli\w*|padre|madre|genitor\w*|fratell\w*|sorell\w*|erede|eredi|nonn\w*|"
    r"socio|soci|amministrator\w*|imputat\w*|indagat\w*|persona offesa|vittima|agent\w*|carabinier\w*|"
    r"polizia|pubblico ufficiale|notaio|avvocato|architett\w*|ingegner\w*|costruttor\w*|"
    r"istruttor\w*|bagnin\w*|albergator\w*|titolare)\b", re.IGNORECASE)
_FIRST_PERSON_RE = re.compile(r"\b(?:io|ho|abbiamo|mi|mio|mia|miei|mie|sono stat[oa]|nostr[oaie])\b", re.IGNORECASE)
# A capitalised word that is not a sentence's first one: a person's or a firm's name ("Marco Rossi", "Marex").
_NAME_RE = re.compile(r"(?<=[a-zàèéìòù,;:] )([A-Z][a-zàèéìòù]+(?:\s+[A-Z][a-zàèéìòù]+)?)")
_NOT_NAMES = {"Cassazione", "Corte", "Tribunale", "Stato", "Repubblica", "Costituzione", "Codice", "Legge",
              "Comune", "Regione", "Provincia", "Italia", "Europa", "Unione", "Ministero", "Consiglio", "Tar",
              "Procura", "Giudice", "Pubblico", "Ministero", "Euro", "Gennaio", "Febbraio", "Marzo", "Aprile",
              "Maggio", "Giugno", "Luglio", "Agosto", "Settembre", "Ottobre", "Novembre", "Dicembre", "Art",
              "Cod", "Ccnl", "Inps", "Inail", "Iva", "Irpef", "Ires", "Candidato", "Il", "La", "Lo", "Gli", "Le"}

# Past events: compound past, remote past, past participle with an auxiliary; dates and sequence words.
_PAST_RE = re.compile(
    r"\b(?:ha|hanno|ho|abbiamo|avete|aveva|avevano|è|sono|era|erano|fu|furono|venne|vennero|viene|vengono)\s+"
    r"(?:\w+\s+)?\w+(?:ato|ata|ati|ate|uto|uta|uti|ute|ito|ita|iti|ite|so|sa|si|se|tto|tta|tti|tte)\b"
    r"|\b\w{3,}(?:ò|ì|arono|irono|erono)\b", re.IGNORECASE)
_DATE_RE = re.compile(
    r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b|\b\d{1,2}\s+(?:gennaio|febbraio|marzo|aprile|maggio|giugno|luglio|agosto|"
    r"settembre|ottobre|novembre|dicembre)\b|\bnel\s+(?:19|20)\d{2}\b|\bin data\b", re.IGNORECASE)
_SEQUENCE_RE = re.compile(r"\b(?:successivamente|in seguito|dopo (?:alcun|qualche|due|tre|pochi|un|una)|"
                          r"poco dopo|il giorno (?:dopo|seguente)|nel frattempo|pochi giorni|alcuni (?:giorni|mesi)|"
                          r"a distanza di|quindi|infine|allora)\b", re.IGNORECASE)
# Present-tense narration, the exams' own ("Tizio vende a Caio ..."): a party followed by a verb.
_PRESENT_ACTION_RE = re.compile(rf"\b(?:{_SCHOOL_NAMES})\b,?\s+(?:\w+\s+){{0,6}}?\w+(?:a|e|ano|ono)\b",
                                re.IGNORECASE)

_ASK_RE = re.compile(
    r"\bil candidato\b|\bquali azioni\b|\bquali (?:rimedi|tutele|strumenti|difese|eccezioni|iniziative)\b|"
    r"\bstrategi\w*\b|\bdifes[ao]\b|\bdifender\w*\b|\bparere\b|\bcosa (?:posso|può|possono|possiamo|devo|deve) fare\b|"
    r"\bcome (?:posso|può|possono|possiamo|devo|deve) (?:agire|tutelar\w*|difender\w*|comportar\w*|procedere)\b|"
    r"\bche (?:possibilità|chance|probabilità)\b|\bha ragione\b|\bpuò (?:ottenere|chiedere|agire|pretendere)\b|"
    r"\bposso (?:ottenere|chiedere|agire|pretendere|fare causa|denunciare)\b|\bassist\w+ (?:il|la) client\w*\b|"
    r"\brediga\b|\bprospetti\b|\bindividui\b|\bvalut\w+ (?:la|le) posizion\w*\b", re.IGNORECASE)
_QUESTION_RE = re.compile(
    r"\?|\b(?:cos'è|cosa (?:è|sono|si intende)|che cos|quali sono|qual è|come funziona|in cosa consiste|"
    r"quando si|entro quanto|entro quale|qual è il termine|che differenza|differenza tra|requisiti|presupposti|"
    r"disciplina|definizione|si prescrive|quanto dura|chi può|è possibile)\b", re.IGNORECASE)
_NORM_RE = re.compile(r"\bart(?:icol[oi])?\.?\s*\d+|\bc\.\s*[cp]\.|\bcodice\b|\blegge\b|\bd\.\s*lgs\b|\bnorma\b|"
                      r"\bistitut\w*\b|\btermine\b|\bprocedura\b|\bprocediment\w*\b|\bricorso\b|\bappello\b",
                      re.IGNORECASE)
_CCNL_RE = re.compile(r"\bccnl\b|\bc\.c\.n\.l\.?|\bcontratto collettivo\b|\baccordo collettivo\b", re.IGNORECASE)
_EMPLOYMENT_RE = re.compile(r"\blavorat\w*|\bdipendent\w*|\bdatore di lavoro\b|\bassunt\w*|\bimpiegat\w*|\boperai\w*|"
                            r"\bapprendist\w*|\bcolf\b|\bbadant\w*", re.IGNORECASE)
_CCNL_TOPIC_RE = re.compile(r"\bretribuzion\w*|\bstipendi\w*|\bminim[io] (?:tabellar|contrattual)\w*|\bferie\b|"
                            r"\bpermessi\b|\bpreavviso\b|\blivell\w* di inquadramento|\binquadrament\w*|"
                            r"\bstraordinari\w*|\btredicesim\w*|\bquattordicesim\w*|\bscatti di anzianità|"
                            r"\bperiodo di prova\b|\bcomporto\b|\bmalattia\b|\bmansion\w*|\borario di lavoro\b|"
                            r"\bindennità\b|\btrasferta\b|\breperibilità\b", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^.!?\n]+[.!?]?")


_CALC_RE = re.compile(r"\bcalcol\w*|\bquanto (?:mi |gli |le |ci )?spett\w*|\bimporto (?:dovuto|spettante)\b|"
                      r"\binteressi legali\b|\brivalutazion\w* monetari\w*", re.IGNORECASE)


def asks_calculation(text: str) -> bool:
    """The message asks for a number to be computed: the calculation path keeps precedence over a narrative."""
    return bool(_CALC_RE.search(text or ""))


def _sentences(text: str) -> List[str]:
    return [s.strip() for s in _SENTENCE_RE.findall(text or "") if len(s.split()) >= 3]


def parties(text: str) -> List[str]:
    """The distinct parties the text names: school names, companies, roles, capitalised names, the user."""
    found: List[str] = []

    def add(label: str) -> None:
        key = label.lower().strip()
        if key and key not in found:
            found.append(key)

    for m in _SCHOOL_RE.finditer(text or ""):
        add(m.group(1))
    for m in _COMPANY_RE.finditer(text or ""):
        add(m.group(0))
    for m in _ROLE_RE.finditer(text or ""):
        add(m.group(1)[:6])
    for m in _NAME_RE.finditer(text or ""):
        words = m.group(1).split()
        if words[0] not in _NOT_NAMES and not _SCHOOL_RE.fullmatch(words[0]):
            add(m.group(1))
    if _FIRST_PERSON_RE.search(text or ""):
        add("io")
    return found


def events(text: str) -> int:
    """How many events the text narrates: past verbs, dates, sequence words, a party's present-tense action
    in a sentence that is not a question."""
    narrated = " ".join(s for s in _sentences(text) if not s.endswith("?"))
    return (len(_PAST_RE.findall(narrated)) + len(_DATE_RE.findall(narrated))
            + len(_SEQUENCE_RE.findall(narrated)) + len(_PRESENT_ACTION_RE.findall(narrated)))


def signals(text: str) -> Dict[str, Any]:
    text = text or ""
    found = parties(text)
    return {"sentences": len(_sentences(text)), "words": len(text.split()), "parties": found,
            "events": events(text), "ask": bool(_ASK_RE.search(text)), "question": bool(_QUESTION_RE.search(text)),
            "norm": bool(_NORM_RE.search(text)), "ccnl_named": bool(_CCNL_RE.search(text)),
            "employment": bool(_EMPLOYMENT_RE.search(text)), "ccnl_topic": bool(_CCNL_TOPIC_RE.search(text))}


# --- the decision --------------------------------------------------------------------------------------

def decide(sig: Dict[str, Any]) -> tuple:
    """(route or None when ambiguous, reasons, the leaning for a tie) from the signals alone."""
    n_parties, n_events = len(sig["parties"]), sig["events"]
    fact_pattern = n_parties >= 2 and n_events >= 2 and (sig["sentences"] >= 3 or sig["words"] >= 40)
    reasons = [f"{sig['sentences']} sentences", f"{n_parties} parties ({', '.join(sig['parties'][:5])})",
               f"{n_events} events"]
    if sig["ccnl_named"]:
        return "ccnl", reasons + ["a collective agreement named"], "ccnl"
    if sig["employment"] and sig["ccnl_topic"]:
        return "ccnl", reasons + ["an employment question on a matter a CCNL governs"], "ccnl"
    if fact_pattern and sig["ask"]:
        return "narrative", reasons + ["a request for action/opinion/strategy"], "narrative"
    if n_parties < 2 and n_events < 2:
        return "direct", reasons + ["no fact pattern"], "direct"
    if not fact_pattern and sig["sentences"] <= 2 and (sig["question"] or sig["norm"]) and not sig["ask"]:
        return "direct", reasons + ["a short question about a norm or institution"], "direct"
    lean = "narrative" if (sig["ask"] or (fact_pattern and sig["question"])) else "direct"
    why = ("a fact pattern without an explicit request" if fact_pattern
           else "a request with too few parties or events for a fact pattern" if sig["ask"]
           else "some parties or events, no request")
    return None, reasons + [why], lean


_TIE_SYSTEM = (
    "Sei un classificatore per un assistente legale italiano. Il messaggio è una NARRATIVA se racconta i fatti "
    "di un caso concreto (parti, vicende) e chiede che cosa fare, un parere, una strategia o una valutazione di "
    "quel caso; è una domanda DIRETTA se chiede di una norma, di un istituto, di un termine o di una procedura "
    "in generale, anche con un esempio. Rispondi SOLO con una parola: narrativa o diretta.")
_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()
_CACHE_MAX = 256


def _tie_break(text: str, lean: str) -> tuple:
    """(route, how): one short call; the leaning on any failure or an unreadable reply."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from .ai_chat import _call_chat
    try:
        view = text if len(text) <= 1200 else f"{text[:400]}\n[…]\n{text[-800:]}"
        reply = _call_chat([SystemMessage(content=_TIE_SYSTEM), HumanMessage(content=view)], max_tokens=5)
        word = (reply or "").strip().lower()
        if word.startswith("narrativ"):
            return "narrative", "llm tie-break: narrativa"
        if word.startswith("dirett"):
            return "direct", "llm tie-break: diretta"
        return lean, f"llm tie-break unreadable ({word[:20]!r}): leaning"
    except Exception as exc:
        logger.warning("router tie-break failed, leaning %s: %s", lean, exc)
        return lean, "llm tie-break failed: leaning"


def classify(text: str, has_document: bool = False, tie_break: bool = True, log: bool = True) -> Dict[str, Any]:
    """{"route", "reasons", "signals", "tie_break"}. The same text is classified once per process (the chat
    and the pipeline both ask)."""
    if has_document:
        result = {"route": "document", "reasons": ["a document is attached"], "signals": {}, "tie_break": False}
        if log:
            vlog("route_chosen", result)
        return result
    key = f"{int(tie_break)}:{text or ''}"
    with _cache_lock:
        cached = _cache.get(key)
    if cached is not None:
        return cached
    sig = signals(text)
    route, reasons, lean = decide(sig)
    tied = False
    if route is None:
        if tie_break:
            route, how = _tie_break(text or "", lean)
            tied = True
        else:
            route, how = lean, "no tie-break: leaning"
        reasons = reasons + [how]
    result = {"route": route, "reasons": reasons, "signals": sig, "tie_break": tied}
    with _cache_lock:
        if len(_cache) >= _CACHE_MAX:
            _cache.clear()
        _cache[key] = result
    if log:
        vlog("route_chosen", {"route": route, "reasons": reasons, "tie_break": tied})
        logger.info("router: %s (%s)", route, "; ".join(reasons))
    return result


def frame_confirms(analysis: Optional[Dict[str, Any]], text: str) -> Optional[bool]:
    """With the case frame built: does it confirm a narrative -- two or more parties (the typed facts' parties,
    else the text's) and two or more events (the frame's facts, else the typed facts' events)? None without a
    frame."""
    analysis = analysis or {}
    frame = analysis.get("frame") or {}
    typed = analysis.get("fact_signals") or {}
    typed = typed.get("facts", typed) if isinstance(typed, dict) else {}
    if not frame and not typed:
        return None
    n_parties = len([p for p in (typed.get("parties") or []) if isinstance(p, dict) and (p.get("name") or p.get("role"))]) \
        or len(parties(text))
    n_events = len([f for f in (frame.get("facts") or []) if str(f).strip()]) \
        or len([e for e in (typed.get("events") or []) if isinstance(e, dict) and e.get("what")])
    return n_parties >= 2 and n_events >= 2


def route_for_pipeline(text: str, analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The route a strategy-pipeline run goes by. Without the router, today's routing decided: the strategy
    pipeline is the narrative path. With it, the router's route, and a narrative the frame does not confirm
    (fewer than two parties or events) is logged as such and not treated as one."""
    if not enabled():
        return {"route": "narrative", "reasons": ["router off: the strategy pipeline is the narrative path"]}
    result = dict(classify(text))
    if result["route"] == "narrative":
        confirmed = frame_confirms(analysis, text)
        result["frame_confirms"] = confirmed
        if confirmed is False:
            result = {**result, "route": "direct",
                      "reasons": result["reasons"] + ["the case frame has fewer than two parties or events"]}
        vlog("route_frame_check", {"confirmed": confirmed, "route": result["route"]})
    return result
