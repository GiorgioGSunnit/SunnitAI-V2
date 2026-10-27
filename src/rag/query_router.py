"""ROUTER_ENABLED: which path a chat request takes, by the kind of request it is (26 Oct, the source-tiered
retrieval architecture).

  narrative  two or more parties (named, Tizio/Caio, companies, public bodies, roles, the user) AND two or
             more events or a request for action, an opinion, a strategy or a parere ("quali azioni",
             "strategia", "difesa"); an exam's markers ("il candidato", "assunte le vesti") make it one
             regardless. A one-sentence question whose parties are only generic roles ("il conduttore può
             chiedere al locatore ...?") is not one. The strategy pipeline (run_case_analysis_pipeline), whose
             rulings may act first as a map (HYBRID_NARRATIVE_ENABLED).
  direct     a question about a norm, an institution, a term or a procedure, without a fact pattern. The
             chat's graph, its articles from primary documents first (PRIMARY_FIRST_ENABLED).
  ccnl       a collective agreement named, or an employment relationship that is the question's subject
             (employer and worker, a question on pay, leave, dismissal or contract terms). Labour facts inside a
             wider narrative stay narrative, unless its question itself names the agreement. Tagged only:
             today's path.
  document   a question about an uploaded document: its own path, unchanged.

Deterministic first (signals); one short LLM call breaks a tie only when the signals disagree. Every
decision is logged with its reasons ("route_chosen"). In the strategy pipeline the case frame may turn a
narrative into a direct question only when it finds fewer than two parties AND fewer than two events
(route_for_pipeline), never an exam's. The routes the chat had before -- defensive draft,
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
# The firm's name capitalised (case-sensitive: "presso la Gamma S.r.l.", not the whole phrase), its form in
# any case.
_COMPANY_RE = re.compile(r"\b[A-Z][\w&'.-]*(?:\s+[A-Z][\w&'.-]*)*\s+(?i:s\.?r\.?l\.?s?|s\.?p\.?a\.?|s\.?n\.?c\.?|"
                         r"s\.?a\.?s\.?|s\.?c\.?a\.?r\.?l\.?|soc\.\s*coop)\b")
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
# Words that open a sentence capitalised without being anyone's name.
_SENTENCE_OPENERS = {
    "Il", "La", "Lo", "Gli", "Le", "I", "L", "Un", "Una", "Uno", "Nel", "Nella", "Nei", "Negli", "Dopo", "Prima",
    "Quali", "Qual", "Quale", "Come", "Cosa", "Che", "Chi", "Dove", "Quando", "Quanto", "Quanti", "Perché",
    "Entro", "In", "Per", "Con", "Da", "Dal", "Dalla", "Di", "Del", "Della", "Al", "Alla", "Ai", "Sul", "Sulla",
    "Tra", "Fra", "Successivamente", "Inoltre", "Infine", "Poi", "Quindi", "Tuttavia", "Ma", "E", "Ed", "Si",
    "Non", "Questo", "Questa", "Tale", "Secondo", "Sono", "Ho", "Mi", "Mio", "Mia", "Lei", "Lui", "Egli", "Essa",
    "Loro", "Ora", "Oggi", "Ieri", "Durante", "Mentre", "Se", "Anche", "Pertanto", "Allora", "Vorrei", "Posso",
    "Può", "Devo", "Deve", "Ci", "Vi", "Ne", "A", "O", "Già", "Sempre", "Ogni", "Tutti", "Alcuni", "Altri"}
_CAPITALISED_RE = re.compile(r"(?<![\w'])([A-Z][a-zàèéìòù]{2,})\b")
# "Primo"/"Secondo" are school names only as a capitalised party, never "secondo il CCNL", "in primo grado".
_ORDINAL_NOT_PARTY_RE = re.compile(r"(?:primo|secondo)\s+(?:il|lo|la|l'|i|gli|le|me|te|noi|voi|cui|quanto|"
                                   r"grado|comma|periodo|piano|luogo|caso|semestre|anno|motivo)\b", re.IGNORECASE)

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
# Public bodies as parties ("il Comune di Rimini", "l'Agenzia delle Entrate", "la ASL").
_PA_RE = re.compile(r"\b(?:comune di \w+|regione \w+|provincia di \w+|ministero (?:del|della|dell'|degli|per)\s*\w+|"
                    r"agenzia delle entrate|agenzia delle dogane|a\.?s\.?l\.?(?=\s|$)|azienda sanitaria|"
                    r"azienda ospedaliera|inps|inail|prefettur\w*|questur\w*|pubblica amministrazione|"
                    r"(?<![.\w])p\.\s?a\.(?!\w))", re.IGNORECASE)
_POSSESSIVE = ("suo", "sua", "mio", "mia", "miei", "nostro", "nostra")
# An exam's trace: the candidate asked to take a party's side.
_EXAM_RE = re.compile(r"\b(?:il|la) candidat[oa]\b|\bassunt[eioa] (?:le|la) vest[ei]\b|"
                      r"\bpremess[io] (?:brevi )?cenni\b", re.IGNORECASE)
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
_EMPLOYER_RE = re.compile(r"\bdatore di lavoro\b|\bdatrice di lavoro\b|\bazienda\b|\bimpresa\b|\bditta\b|"
                          r"\bsociet[àa]\b|\bprincipale\b|\btitolare\b", re.IGNORECASE)
_WORKER_RE = re.compile(r"\blavorat\w*|\bdipendent\w*|\bimpiegat\w*|\boperai\w*|\bapprendist\w*|\bcolf\b|"
                        r"\bbadant\w*|\bcollaborat\w*|\bassunt[oa]\b", re.IGNORECASE)
# The user as the worker: the employer is implied ("sono un dipendente", "lavoro presso").
_SELF_WORKER_RE = re.compile(r"\bsono (?:un|una|un')\s*(?:dipendent|lavorat|impiegat|operai|apprendist|colf|badant)\w*|"
                             r"\blavoro (?:come|presso|per|da)\b|\bmi (?:hanno|ha) (?:assunt|licenziat)\w*",
                             re.IGNORECASE)
_CCNL_TOPIC_RE = re.compile(r"\bretribuzion\w*|\bstipendi\w*|\bpaga\b|\bbusta paga\b|\blicenziament\w*|\blicenzia\w*|"
                            r"\bdimission\w*|\bdimett\w*|\btfr\b|\btrattamento di fine rapporto\b|\bcongedo\b|"
                            r"\bmaternità\b|\bpaternità\b|\bcontratto a (?:tempo determinato|termine)\b|"
                            r"\bpart[- ]time\b|\brinnovo del contratto\b|\bminim[io] (?:tabellar|contrattual)\w*|\bferie\b|"
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


def party_kinds(text: str) -> Dict[str, str]:
    """{party: kind} for the distinct parties the text names, in order: "school" (Tizio, Caio), "company",
    "pa" (a public body), "name" (a capitalised name), "user" (the first person), "own role" (a role with a
    possessive: "mia madre", "il suo datore"), "role" (a generic one: "il conduttore"). All but "role" are
    concrete: a real case's parties."""
    text = text or ""
    found: Dict[str, str] = {}

    def add(label: str, kind: str) -> None:
        key = label.lower().strip()
        if key and (key not in found or (found[key] == "role" and kind != "role")):
            found[key] = kind

    def inside_a_body(label: str) -> bool:
        """Part of a company's or a public body's name already found ("Comune" of "Comune di Ferrara")."""
        word = label.lower().strip()
        return any(kind in ("company", "pa") and re.search(rf"\b{re.escape(word)}\b", key)
                   for key, kind in found.items())

    for m in _SCHOOL_RE.finditer(text):
        if m.group(1).lower() in ("primo", "secondo") and (
                not m.group(1)[0].isupper() or _ORDINAL_NOT_PARTY_RE.match(text, m.start())):
            continue
        add(m.group(1), "school")
    for m in _COMPANY_RE.finditer(text):
        add(m.group(0), "company")
    for m in _PA_RE.finditer(text):
        add(m.group(0), "pa")
    for m in _ROLE_RE.finditer(text):
        if inside_a_body(m.group(1)):
            continue
        determiner = m.group(0)[:m.start(1) - m.start(0)].strip().lower()
        add(m.group(1)[:6], "own role" if determiner in _POSSESSIVE else "role")
    for m in _NAME_RE.finditer(text):
        words = m.group(1).split()
        if words[0] not in _NOT_NAMES and not _SCHOOL_RE.fullmatch(words[0]) and m.group(1).lower() not in found \
                and not inside_a_body(words[0]):
            add(m.group(1), "name")
    # A name that opens its sentences ("Marco affitta ... Marco vuole ..."): a capitalised word, not a usual
    # opener, written twice or more.
    counts: Dict[str, int] = {}
    for m in _CAPITALISED_RE.finditer(text):
        counts[m.group(1)] = counts.get(m.group(1), 0) + 1
    for word, n in counts.items():
        if n >= 2 and word not in _SENTENCE_OPENERS and word not in _NOT_NAMES and not _SCHOOL_RE.fullmatch(word) \
                and not any(word.lower() == k or k.startswith(word.lower() + " ") for k in found) \
                and not inside_a_body(word):
            add(word, "name")
    if _FIRST_PERSON_RE.search(text):
        add("io", "user")
    return found


def parties(text: str) -> List[str]:
    """The distinct parties the text names: school names, companies, public bodies, roles, capitalised names,
    the user."""
    return list(party_kinds(text))


def _concrete(kinds: Dict[str, str]) -> List[str]:
    return [p for p, kind in kinds.items() if kind != "role"]


def events(text: str) -> int:
    """How many events the text narrates: past verbs, dates, sequence words, a party's present-tense action
    in a sentence that is not a question; and a statement naming a concrete party with none of those ("Marco
    affitta a Luca un negozio.")."""
    concrete = [p for p in _concrete(party_kinds(text)) if p != "io"]
    count = 0
    for sentence in _sentences(text):
        if sentence.endswith("?"):
            continue
        found = (len(_PAST_RE.findall(sentence)) + len(_DATE_RE.findall(sentence))
                 + len(_SEQUENCE_RE.findall(sentence)) + len(_PRESENT_ACTION_RE.findall(sentence)))
        about = any(re.search(rf"\b{re.escape(p)}\b", sentence, re.IGNORECASE) for p in concrete)
        count += found or (1 if about and len(sentence.split()) >= 4 else 0)
    return count


def _question_part(text: str) -> str:
    """The text's question: its sentences ending in "?" and its last sentence (the whole of a short text)."""
    sentences = _sentences(text)
    if len(sentences) <= 1:
        return text or ""
    asked = [s for s in sentences[:-1] if s.endswith("?")]
    return " ".join(asked + [sentences[-1]])


def signals(text: str) -> Dict[str, Any]:
    text = text or ""
    kinds = party_kinds(text)
    question = _question_part(text)
    employment = bool(_EMPLOYMENT_RE.search(text))
    relationship = bool((_EMPLOYER_RE.search(text) and _WORKER_RE.search(text)) or _SELF_WORKER_RE.search(text))
    return {"sentences": len(_sentences(text)), "words": len(text.split()), "parties": list(kinds),
            "concrete_parties": _concrete(kinds), "events": events(text), "exam": bool(_EXAM_RE.search(text)),
            "ask": bool(_ASK_RE.search(text)), "question": bool(_QUESTION_RE.search(text)),
            "norm": bool(_NORM_RE.search(text)), "ccnl_named": bool(_CCNL_RE.search(text)),
            "ccnl_in_question": bool(_CCNL_RE.search(question)), "employment": employment,
            "ccnl_topic": bool(_CCNL_TOPIC_RE.search(text)),
            "employment_subject": relationship and bool(_CCNL_TOPIC_RE.search(question))}


# --- the decision --------------------------------------------------------------------------------------

def decide(sig: Dict[str, Any]) -> tuple:
    """(route or None when ambiguous, reasons, the leaning for a tie) from the signals alone:
      1. an exam's markers -> narrative;
      2. two or more parties AND (two or more events OR a request for action/strategy/parere), unless it is a
         one-sentence question with only generic roles and no events -> narrative (ccnl when its question
         itself names the collective agreement: labour facts inside a narrative stay narrative);
      3. a collective agreement named, or an employment relationship the question is about -> ccnl;
      4. no fact pattern (fewer than two parties and events, a short question on a norm, a general
         question) -> direct;
      5. otherwise a tie, with a leaning."""
    n_parties, n_events = len(sig["parties"]), sig["events"]
    concrete = sig.get("concrete_parties", sig["parties"])
    reasons = [f"{sig['sentences']} sentences",
               f"{n_parties} parties ({', '.join(sig['parties'][:5])}; concrete {len(concrete)})",
               f"{n_events} events"]
    if sig.get("exam"):
        return "narrative", reasons + ["an exam's markers (il candidato / assunte le vesti)"], "narrative"
    general = not concrete and n_events < 2 and sig["sentences"] <= 1
    if n_parties >= 2 and (n_events >= 2 or sig["ask"]) and not general:
        if sig.get("ccnl_in_question"):
            return "ccnl", reasons + ["a fact pattern whose question names the collective agreement"], "ccnl"
        why = "two or more events" + (" and a request for action/opinion/strategy" if sig["ask"] else "") \
            if n_events >= 2 else "a request for action/opinion/strategy"
        return "narrative", reasons + [f"two or more parties and {why}"], "narrative"
    if sig["ccnl_named"]:
        return "ccnl", reasons + ["a collective agreement named"], "ccnl"
    if sig.get("employment_subject"):
        return "ccnl", reasons + ["an employment relationship the question is about (pay, leave, dismissal, "
                                  "contract terms)"], "ccnl"
    if n_parties < 2 and n_events < 2:
        return "direct", reasons + ["no fact pattern"], "direct"
    if sig["sentences"] <= 2 and n_events < 2 and (sig["question"] or sig["norm"]):
        return "direct", reasons + ["a short question about a norm or institution"], "direct"
    if general:
        return "direct", reasons + ["a general question: generic roles only, no events"], "direct"
    lean = "narrative" if sig["ask"] or (n_parties >= 2 and n_events >= 1) else "direct"
    why = ("parties without enough events or a request" if n_parties >= 2
           else "events or a request with fewer than two parties")
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


def frame_counts(analysis: Optional[Dict[str, Any]], text: str) -> Optional[tuple]:
    """(parties, events) the case frame finds: the typed facts' parties, else the text's; the frame's facts,
    else the typed facts' events. None without a frame."""
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
    return n_parties, n_events


def frame_confirms(analysis: Optional[Dict[str, Any]], text: str) -> Optional[bool]:
    """With the case frame built: does it leave the narrative standing? Only a frame with fewer than two
    parties AND fewer than two events turns it down -- one of the two is enough to keep it. None without a
    frame."""
    counts = frame_counts(analysis, text)
    if counts is None:
        return None
    n_parties, n_events = counts
    return not (n_parties < 2 and n_events < 2)


def route_for_pipeline(text: str, analysis: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The route a strategy-pipeline run goes by. Without the router, today's routing decided: the strategy
    pipeline is the narrative path. With it, the router's route; a narrative whose frame finds fewer than
    two parties and fewer than two events is logged as such and not treated as one (never an exam's)."""
    if not enabled():
        return {"route": "narrative", "reasons": ["router off: the strategy pipeline is the narrative path"]}
    result = dict(classify(text))
    if result["route"] == "narrative":
        counts = frame_counts(analysis, text)
        confirmed = frame_confirms(analysis, text)
        exam = bool((result.get("signals") or {}).get("exam"))
        result["frame_confirms"] = confirmed
        if confirmed is False and not exam:
            result = {**result, "route": "direct",
                      "reasons": result["reasons"] + [f"the case frame has fewer than two parties and fewer "
                                                      f"than two events ({counts[0]}, {counts[1]})"]}
        vlog("route_frame_check", {"confirmed": confirmed, "counts": counts, "exam": exam, "route": result["route"]})
        logger.info("router: frame check %s -> %s", counts, result["route"])
    return result
