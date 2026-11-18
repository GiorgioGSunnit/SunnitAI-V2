"""v60 (C): the procedural facts of an article, read from its own text, never written by the model.

For a c.p. offence (rule_quote.offence): procedibilità (a querela / d'ufficio, with the comma that says so, and the
commi that make it d'ufficio again), the querela's term (the article's own, else art. 124 c.p.), its remissione (the
article's «irrevocabile», else art. 152 c.p.), the competenza (Corte d'assise, giudice di pace, tribunale collegiale or
monocratico, from the pena the first comma states) and the pena range. For a c.p.p., c.p.a. or c.p.c. article: the
term it states («entro … giorni»), with its comma.

The facts are computed once per article and stored on its Section node (PROPERTY, JSON; VERSION) by
scripts/fill_procedural_store.py; at run time attach() reads the store for the topics' rows, and computes from the row's
text what the store does not hold (source "computed"). §4 renders them (render_computed.procedure_lines) and every
writer sentence on these points is checked against them (consistency()).
"""

import json
import logging
import re
import unicodedata
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROPERTY = "proc_facts"
VERSION_PROPERTY = "proc_version"
VERSION = 4                     # v60h (1): the d'ufficio conditions and the querela's comma; v65: termini per comma;
                                # v67: each term's subject the act that owns its number, a «1-bis» comma its own
READ_VERSIONS = (1, 2, 3, VERSION)  # a v1 entry is read, its conditions computed from the row's text (attach); v2 / v3,
                                    # its termini computed from the row's text (by_number)
CODES = ("c.p.", "c.p.p.", "c.p.a.", "c.p.c.")
QUERELA_DEFAULT = "tre mesi (art. 124 c.p.)"
REMISSIONE_DEFAULT = "remissibile (art. 152 c.p.)"
UFFICIO_DEFAULT = "d'ufficio (regola generale: l'articolo non prevede la querela)"
# d.lgs. 274/2000, art. 4, co. 1, lett. a: the c.p. delitti of the giudice di pace (within that article's limits) --
# v76: by the exact article (612-bis, 582-bis … are not in it)
PEACE_JUDGE = {"581", "582", "590", "592", "595", "612", "626", "627", "631", "632", "633", "635", "636", "637", "638",
               "639", "647"}

_NUMBERS = {"un": 1, "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "sette": 7, "otto": 8,
            "nove": 9, "dieci": 10, "undici": 11, "dodici": 12, "tredici": 13, "quattordici": 14, "quindici": 15,
            "sedici": 16, "diciassette": 17, "diciotto": 18, "diciannove": 19, "venti": 20, "ventuno": 21,
            "ventidue": 22, "ventitre": 23, "ventitré": 23, "ventiquattro": 24, "venticinque": 25, "ventisei": 26,
            "trenta": 30, "sessanta": 60, "novanta": 90, "centoventi": 120, "centottanta": 180, "quarantacinque": 45,
            "quindicina": 15, "trentasei": 36,
            # v65: the terms of the procedural codes (641 c.p.c.'s «quaranta giorni»)
            "quaranta": 40, "cinquanta": 50, "settanta": 70, "ottanta": 80, "cento": 100, "centocinquanta": 150}
_QTY = r"(\d+|" + "|".join(sorted(_NUMBERS, key=len, reverse=True)) + r")"
_QUERELA_RE = re.compile(r"\ba querela (?:della persona offesa|di parte)|\bpunibile a querela|\bsi procede a querela",
                         re.IGNORECASE)
_UFFICIO_RE = re.compile(r"\bsi procede,?\s*(?:(?:tuttavia|comunque|però|sempre|altresì|invece)\s*,?\s*)?d['’]ufficio",
                         re.IGNORECASE)          # v60h: «Si procede, tuttavia, d'ufficio» too
# v60h (1): the conditions that make a querela offence d'ufficio -- the clause after «si procede d'ufficio» («se …»,
# «quando …», «nei casi …»), split at «ovvero se»; in the querela's own sentence, its «salvo che / salvo nei casi …»
_OPENER_RE = re.compile(r"^(se|quando|qualora|ove|allorché|nei casi|nel caso|per i fatti|per il fatto)\b\s*", re.IGNORECASE)
_SALVO_RE = re.compile(r"\bsalvo(?:\s+che|\s+(?=nei casi|nel caso|il caso|quando|se\b))\s*(.+)$", re.IGNORECASE)
_OR_IF_RE = re.compile(r",?\s+(?:ovvero|oppure|o)\s+(?=(?:se|quando|qualora)\b)", re.IGNORECASE)
CONDITION_CHARS = 260
_QUERELA_TERM_RE = re.compile(r"termine per (?:la )?(?:proposi|presenta)zione della querela è di " + _QTY + r" (giorni|mesi)"
                              r"|querela (?:è proposta|può essere proposta|si propone) entro (?:il termine di )?" + _QTY
                              + r" (giorni|mesi)", re.IGNORECASE)
_IRREVOCABLE_RE = re.compile(r"querela (?:proposta )?è irrevocabile|remissione (?:della querela )?(?:non è ammessa|è esclusa)"
                             r"|la remissione può essere soltanto processuale", re.IGNORECASE)
_PENA_RE = re.compile(
    r"(ergastolo)"
    r"|reclusione da " + _QTY + r"(?: (mesi|anni|giorni))? a " + _QTY + r" (mesi|anni)"
    r"|reclusione fino a " + _QTY + r" (mesi|anni)"
    r"|reclusione non inferiore (?:a " + _QTY + r" (mesi|anni)|ad? (anni|mesi) " + _QTY + r")"
    r"|(arresto) (?:da|fino a) [^,.;]{1,40}"
    r"|(multa|ammenda) (?:da|fino a) (?:euro )?[\d.]+(?: a (?:euro )?[\d.]+)?", re.IGNORECASE)
MAX_RECLUSIONE = 24             # art. 23 c.p.: the reclusione's general maximum
_TERM_RE = re.compile(r"(?:entro|nel termine (?:perentorio )?di|termine (?:perentorio )?di|termine di decadenza di) "
                      r"(?:il termine di )?" + _QTY + r" (giorni|mesi|anni)", re.IGNORECASE)


def _n(word: str) -> Optional[int]:
    word = (word or "").lower()
    return int(word) if word.isdigit() else _NUMBERS.get(word)


def _years(qty: str, unit: str) -> Optional[float]:
    n = _n(qty)
    if n is None:
        return None
    return n if unit.lower().startswith("ann") else n / 12 if unit.lower().startswith("mes") else n / 365


def _code(code: Any) -> str:
    from .plan_then_write import _code_key
    return _code_key(" ".join(str(code or "").split()))


def _label(k: Optional[int]) -> str:
    return f"co. {k}" if k else "testo"


def pena(text: str) -> Tuple[str, Optional[float], bool]:
    """(the first pena the text states, as written; its maximum in years -- «non inferiore a» has the reclusione's
    general maximum, art. 23 c.p.; ergastolo)."""
    m = _PENA_RE.search(text or "")
    if not m:
        return "", None, False
    whole = " ".join(m.group(0).split())
    if m.group(1):
        return "ergastolo", None, True
    if m.group(2):                  # da X [unit] a Y unit
        return whole, _years(m.group(4), m.group(5)), False
    if m.group(6):                  # fino a X unit
        return whole, _years(m.group(6), m.group(7)), False
    if m.group(8) or m.group(11):   # non inferiore a X unit / ad anni X
        return whole, float(MAX_RECLUSIONE), False
    return whole, None, False


# v62: art. 5 c.p.p. decides first -- the Corte d'assise's offences by article whatever their pena (lett. b: 579, 580,
# 584; lett. d-bis: 600, 601, 602), the delitti contro la personalità dello Stato (Libro II, Titolo I: artt. 241-313)
# punished up to ten years or more (lett. d), and by pena (lett. a: ergastolo or a maximum of 24 years or more, rapina,
# estorsione and the mafia association excepted); then the giudice di pace, then 33-bis / 33-ter
ASSISE = "Corte d'assise (art. 5 c.p.p.)"
ASSISE_ARTICLES = frozenset({"579", "580", "584", "600", "601", "602"})
ASSISE_EXCEPTED = frozenset({"628", "629", "416-bis"})
STATE_OFFENCES = (241, 313)


def assise(number: str, max_years: Optional[float], ergastolo: bool) -> bool:
    """The offence goes to the Corte d'assise under art. 5 c.p.p. (see ASSISE_ARTICLES)."""
    number = str(number or "").strip().lower().replace(" ", "-")
    base = re.match(r"\d+", number)
    if not base:
        return False
    if base.group(0) in ASSISE_ARTICLES and number == base.group(0):
        return True
    if (ergastolo or (max_years is not None and max_years >= 24)) and number not in ASSISE_EXCEPTED \
            and base.group(0) not in ASSISE_EXCEPTED:
        return True
    return STATE_OFFENCES[0] <= int(base.group(0)) <= STATE_OFFENCES[1] and max_years is not None and max_years >= 10


def court(number: str, facts: Dict[str, Any]) -> str:
    """v62: the offence's competenza as read now -- art. 5 c.p.p. first (assise, the stored pena read again), else the
    store's (computed when it was filled)."""
    whole, max_years, ergastolo = pena(str((facts or {}).get("pena") or ""))
    if assise(number, max_years, ergastolo):
        return ASSISE
    return str((facts or {}).get("competenza") or "")


def competenza(number: str, max_years: Optional[float], ergastolo: bool) -> str:
    base = re.match(r"\d+", str(number))
    if assise(str(number), max_years, ergastolo):
        return ASSISE
    if " ".join(str(number or "").lower().split()).replace(" ", "-") in PEACE_JUDGE:      # v76: the article itself --
        return "giudice di pace (art. 4 d.lgs. 274/2000, nei limiti ivi previsti)"           # 612-bis is not 612
    if max_years is None:
        return ""
    if max_years > 10:
        return "tribunale in composizione collegiale (art. 33-bis c.p.p.)"
    return "tribunale in composizione monocratica (art. 33-ter c.p.p.)"


def _clause(text: str) -> Tuple[str, str]:
    """(opener, clause): «se ricorre …» -> («se», «ricorre …»); «nei casi previsti …» -> («», «nei casi previsti …»)."""
    text = " ".join(str(text or "").split()).strip(" ,.;:")
    m = _OPENER_RE.match(text)
    if m and m.group(1).lower() in ("se", "quando", "qualora", "ove", "allorché"):
        opener, text = m.group(1).lower(), text[m.end():]
    else:
        opener = ""
    if len(text) > CONDITION_CHARS:
        text = text[:CONDITION_CHARS].rsplit(" ", 1)[0] + " …"
    return opener, text


def _pieces(clause: str) -> List[str]:
    return [p for p in _OR_IF_RE.split(clause) if p.strip()]


def conditions(commi: Sequence[Tuple[Optional[int], str]]) -> List[Dict[str, str]]:
    """v60h (1): [{"comma", "opener", "text"}] -- each condition under which a querela offence is procedibile d'ufficio,
    as its text states it: the clause of a «si procede d'ufficio» sentence (after it, else before it: «Nei casi previsti
    dal secondo comma si procede d'ufficio»), each «ovvero se» a condition of its own; a «salvo che» of the querela's
    own sentence («punibile a querela della persona offesa, salvo che ricorra …»)."""
    out: List[Dict[str, str]] = []
    for k, text in commi or []:
        for sentence in re.split(r"(?<=[.;])\s+(?=[A-ZÀ-Ü])", " ".join(str(text or "").split())):
            m = _UFFICIO_RE.search(sentence)
            if m:
                after = sentence[m.end():].strip(" ,.;")
                before = sentence[:m.start()].strip(" ,.;")
                before = re.sub(r"\s*(?:,\s*)?e\s*$", "", before)
                clause = after if after else (before if _OPENER_RE.match(before) else "")
                for piece in _pieces(clause):
                    opener, cond = _clause(piece)
                    if cond:
                        out.append({"comma": _label(k), "opener": opener, "text": cond})
                continue
            if _QUERELA_RE.search(sentence):
                salvo = _SALVO_RE.search(sentence[_QUERELA_RE.search(sentence).end():])
                if salvo:
                    che = bool(re.match(r"salvo\s+che\b", salvo.group(0), re.IGNORECASE))
                    for piece in _pieces(salvo.group(1)):
                        opener, cond = _clause(piece)
                        if cond:
                            out.append({"comma": _label(k), "opener": opener or ("se" if che else ""), "text": cond})
    seen, kept = set(), []
    for c in out:
        key = (c["comma"], c["text"].lower())
        if key not in seen:
            seen.add(key)
            kept.append(c)
    return kept


def extract(number: str, code: str, commi: Sequence[Tuple[Optional[int], str]]) -> Dict[str, Any]:
    """The article's procedural facts from its commi [(k, text)] -- {} when its text states none it should hold."""
    from . import rule_quote
    code = _code(code)
    commi = [(k, " ".join(str(t or "").split())) for k, t in commi or [] if str(t or "").strip()]
    if not commi:
        return {}
    out: Dict[str, Any] = {"version": VERSION}
    if code == "c.p." and rule_quote.offence((str(number), "c.p.")):
        querela = [k for k, t in commi if _QUERELA_RE.search(t)]
        ufficio = [k for k, t in commi if _UFFICIO_RE.search(t)]
        if querela:
            out["procedibilita"] = "a querela della persona offesa (" + _label(querela[0]) + ")" + (
                "; d'ufficio nei casi indicati al " + ", ".join(_label(k) for k in ufficio) if ufficio else "")
            out["querela"] = True
            out["querela_comma"] = _label(querela[0])
            out["ufficio_condizioni"] = conditions(commi)
            term = next((m for _, t in commi for m in [_QUERELA_TERM_RE.search(t)] if m), None)
            if term:
                qty, unit = (term.group(1), term.group(2)) if term.group(1) else (term.group(3), term.group(4))
                k = next(k for k, t in commi if _QUERELA_TERM_RE.search(t))
                out["querela_termine"] = f"{qty} {unit} ({_label(k)})"
            else:
                out["querela_termine"] = QUERELA_DEFAULT
            irrevocable = next((k for k, t in commi if _IRREVOCABLE_RE.search(t)), None)
            out["remissione"] = (f"querela irrevocabile ({_label(irrevocable)})" if irrevocable is not None
                                 else REMISSIONE_DEFAULT)
        else:
            out["procedibilita"] = UFFICIO_DEFAULT
            out["querela"] = False
        stated, max_years, ergastolo = pena(commi[0][1])
        if stated:
            out["pena"] = f"{stated} ({_label(commi[0][0])})"
        court = competenza(str(number), max_years, ergastolo)
        if court:
            out["competenza"] = court
        return out
    if code in ("c.p.p.", "c.p.a.", "c.p.c."):
        found = termini(commi)              # v65: every term, per comma, with its subject -- never the first number alone
        since = decorrenze(commi)           # v66: where they run from, the law's own words
        if found or since:
            out["termini"] = found
            if since:
                out["decorrenze"] = since
            ambiguous = any(t.get("ambiguous") for t in found)
            # as a v2 entry wrote it, for its readers; v66: none when a comma gives one act two numbers
            out["termine"] = None if ambiguous or not found else f"{found[0]['text']} ({found[0]['comma']})"
            if ambiguous:
                out["termine_ambiguo"] = True
            return out
        return {}
    return {}


# v65: the procedural acts a term is the term of -- its subject, the noun nearest its number in its sentence
SUBJECT_NOUNS = ("opposizione", "opposizioni", "appello", "ricorso", "reclamo", "impugnazione", "revocazione",
                 "denuncia", "querela", "remissione", "istanza", "richiesta", "memoria", "memorie", "comparsa",
                 "costituzione", "notificazione", "notifica", "deposito", "citazione", "riassunzione", "dichiarazione",
                 "domanda", "eccezione", "eccezioni", "decreto", "sentenza", "ordinanza", "motivazione", "motivi",
                 "precetto", "pignoramento", "udienza", "comparizione", "convalida", "riesame", "interrogatorio",
                 "indagini", "archiviazione", "nomina", "rinuncia", "integrazione", "chiamata", "intervento",
                 "pagamento", "adempimento", "esecuzione", "proroga", "avviso", "giudizio", "prova", "prove",
                 "consulenza", "perizia", "liquidazione", "trascrizione", "vendita", "assegnazione")
_SUBJECT_RE = re.compile(r"\b(" + "|".join(sorted(SUBJECT_NOUNS, key=len, reverse=True)) + r")\b", re.IGNORECASE)
_CLAUSE_STOP_RE = re.compile(r"[;:]|\.\s+(?=[A-ZÀ-Ü])")
# a noun after a preposition is a complement («dalla notificazione», «atto di citazione»), never the term's subject
_PREPOSITION_RE = re.compile(r"(?:\b(?:di|a|da|in|con|su|per|tra|fra|del|dello|della|dei|degli|delle|dell|al|allo|alla|"
                             r"ai|agli|alle|all|dal|dallo|dalla|dai|dagli|dalle|dall|nel|nello|nella|nei|negli|nelle|"
                             r"nell|sul|sullo|sulla|sui|sugli|sulle|sull|o|e|ed|od|oppure|ovvero)['’]?\s*)$",
                             re.IGNORECASE)          # a conjunction: the noun after it shares the complement's place


# v66 (store v3): every form a code states a term in, one finder each, the first one holding a number keeping it
_Q = _QTY[1:-1]                     # the number words' alternation, without its group
# v67: «è» as the corpus writes it too -- «E'» for «È» («E' ammessa la separazione personale dei coniugi», art. 150 c.c.),
# «e'», a curly or grave apostrophe
_IS = r"(?:è|é|e['’`´])"
_ARE = r"(?:" + _IS + r"|sono)"
# v67: an act's words run to its sentence's end, never to an abbreviation's dot («di cui all'art. 404», «nn. 1 e 2»)
_ABBR_DOT = r"(?:(?<=\bart)|(?<=\bartt)|(?<=\bn)|(?<=\bnn)|(?<=\bco)|(?<=\blett)|(?<=\bcpv))\."
_ACT = r"(?:[^.;:]|" + _ABBR_DOT + r")"


def _qu(p: str) -> str:
    """A quantity and its unit, either order («trenta giorni», «giorni sessanta»), as named groups p+q1/u1, p+u2/q2."""
    return (rf"(?:(?P<{p}q1>{_Q})\s+(?P<{p}u1>giorni|mesi|anni)|(?P<{p}u2>giorni|mesi|anni)\s+(?P<{p}q2>{_Q}))")


# «Il termine per proporre l'appello … è di trenta giorni», «… è di giorni sessanta»
_PER_RE = re.compile(r"\b(?:il\s+|i\s+)?termin[ei]\s+per\s+(?P<act>" + _ACT + r"{1,200}?)\s+" + _ARE
                     + r"\s+(?:anche\s+)?(?:di\s+)?" + _qu("a"), re.IGNORECASE)
# «È anche di trenta giorni il termine per proporre la revocazione …»
_INVERTED_RE = re.compile(r"\b" + _ARE + r"\s+(?:anche\s+)?di\s+" + _qu("b") + r"\s+il\s+termine\s+per\s+(?P<act>" + _ACT
                          + r"{1,200})", re.IGNORECASE)
# «Il termine per proporre impugnazione, per ciascuna delle parti, è: a) di quindici giorni, per i provvedimenti …;»
_LIST_HEAD_RE = re.compile(r"\b(?:il\s+|i\s+)?termin[ei]\s+per\s+(?P<act>" + _ACT + r"{1,200}?)\s+" + _ARE + r"\s*:",
                           re.IGNORECASE)
# v67: an item's condition ends with its «;» or its sentence -- never runs into the comma after it
_LIST_ITEM_RE = re.compile(r"(?:^|\s)(?P<letter>[a-z])\)\s*di\s+" + _qu("c") + r"\s*,?\s*(?P<when>(?:[^;.]|" + _ABBR_DOT
                           + r"|\.(?=\S))*)", re.IGNORECASE)
# «Se l'intimato risiede …, il termine è di cinquanta giorni»: a term with no act of its own
_BARE_RE = re.compile(r"\b(?:il\s+)?termine\s+" + _IS + r"\s+(?:di\s+)?" + _qu("d"), re.IGNORECASE)
# «non possono proporsi dopo decorsi sei mesi dalla pubblicazione»
_ELAPSED_RE = re.compile(r"\bdecors[io]\s+" + _qu("e"), re.IGNORECASE)
# v69: a term counted back from an event -- «deve costituirsi … almeno settanta giorni prima dell'udienza» (166 c.p.c.)
_BEFORE_TERM_RE = re.compile(r"\balmeno\s+" + _qu("f") + r"(?=\s+prima\b)", re.IGNORECASE)
# «I termini previsti dal comma 1 decorrono: a) …», «… decorrono dalla notificazione della sentenza …»
_FROM_LIST_RE = re.compile(r"\bdecorr\w*\s*:", re.IGNORECASE)
_FROM_ITEM_RE = re.compile(r"(?:^|\s)([a-z])\)\s*(da\w*[^;]*)", re.IGNORECASE)
_FROM_SENTENCE_RE = re.compile(r"\bdecorr\w*\s+(?:dal|dalla|dallo|dall['’]|dai|dagli|dalle)\b", re.IGNORECASE)
_SENTENCE_STOP_RE = re.compile(r"(?<=[.?!])\s+(?=[A-ZÀ-Ü«])")
_WHEN_RE = re.compile(r"^\s*((?:se|quando|qualora|ove|laddove|nel caso|nei casi)\b[^,]{3,200}),", re.IGNORECASE)
# «dalla notificazione della sentenza», «dalla esecuzione o notificazione del provvedimento»: a noun phrase and its «di …»
# chain, never the clause after it
_W = r"[a-zà-ü][\wà-ü]*"
_SINCE_RE = re.compile(r"\s*,?\s*((?:dal|dalla|dallo|dai|dagli|dalle)\s+" + _W + r"|dall['’]\s*" + _W
                       + r"|prima\s+(?:del|della|dello|dei|degli|delle)\s+" + _W + r"|prima\s+dell['’]\s*" + _W   # v69
                       + r")((?:\s+(?:del|della|"
                       r"dello|dei|degli|delle|di|o|e|od|ed)\s+(?:(?:il|lo|la|i|gli|le|un|una|uno)\s+)?" + _W
                       + r"|\s+(?:dell['’]|d['’])\s*" + _W + r")*)", re.IGNORECASE)
_ACT_CUT_RE = re.compile(r"\b(?:(?:di cui|contro|avverso|sopra|nei confronti|nel caso|per ciascun\w*|che)\b"
                         r"|per\s+(?:il|lo|la|i|gli|le)\b|per\s+l['’])", re.IGNORECASE)    # v67: «per i motivi …»
_COMPLEMENT_RE = re.compile(r"\s+(?:(?:di|del|della|dello|dell['’])\s*(" + "|".join(sorted(SUBJECT_NOUNS + ("terzo",
                            "risposta"), key=len, reverse=True)) + r")\b|per\s+(cassazione)\b)", re.IGNORECASE)


def _quantity(m: "re.Match", p: str) -> Tuple[str, str]:
    q = m.group(f"{p}q1") or m.group(f"{p}q2")
    u = m.group(f"{p}u1") or m.group(f"{p}u2")
    return q, u.lower()


def acts(phrase: str) -> str:
    """v66: the acts a «termine per …» names, in order and joined («l'appello, la revocazione e l'opposizione di terzo
    di cui all'articolo 404» -> «appello, revocazione, opposizione di terzo»): the free nouns of SUBJECT_NOUNS before
    the phrase's first qualifier (_ACT_CUT_RE), each with its «di …» or «per cassazione»."""
    cut = _ACT_CUT_RE.search(phrase or "")
    phrase = (phrase or "")[:cut.start()] if cut else (phrase or "")
    out = []
    free = None                     # v67: a noun after «e / o» shares the place of the noun before it («dalla
    for m in _SUBJECT_RE.finditer(phrase):  # esecuzione o notificazione»: both complements)
        before = phrase[max(0, m.start() - 8):m.start()]
        joined = re.search(r"\b(?:e|o|ed|od)\s+(?:l['’]|il|la|lo|i|gli|le)?\s*$", before)
        free = free if joined and free is not None else (bool(joined) or not _PREPOSITION_RE.search(before))
        if not free:
            continue
        of = _COMPLEMENT_RE.match(phrase, m.end())
        name = (m.group(1) + (f" di {of.group(1)}" if of and of.group(1) else " per cassazione" if of else "")).lower()
        if name not in out:
            out.append(name)
    return ", ".join(out)


# v67: the act a number is the term of -- the noun phrase that owns its «entro», «nel termine di», «decorsi», read from
# the verb that governs it: «decreto … da emettere entro trenta giorni» -> «emissione del decreto», never the
# «opposizione» the sentence names later. A verb's act ("": its object is the act -- «di proporre appello»: appello)
VERB_ACTS = {"emettere": "emissione", "pronunciare": "pronuncia", "depositare": "deposito", "notificare": "notificazione",
             "comunicare": "comunicazione", "pagare": "pagamento", "consegnare": "consegna", "restituire": "restituzione",
             "presentare": "presentazione", "iscrivere": "iscrizione", "trascrivere": "trascrizione",
             "eseguire": "esecuzione", "compiere": "compimento", "riassumere": "riassunzione", "rinnovare": "rinnovazione",
             "integrare": "integrazione", "impugnare": "impugnazione", "convalidare": "convalida",
             "adempiere": "adempimento", "costituire": "costituzione", "comparire": "comparizione",
             "opporre": "opposizione", "proporre": "", "fare": "", "promuovere": "", "esperire": ""}
PARTICIPLE_ACTS = {"emess": "emissione", "pronunciat": "pronuncia", "depositat": "deposito", "notificat": "notificazione",
                   "comunicat": "comunicazione", "pagat": "pagamento", "consegnat": "consegna", "restituit": "restituzione",
                   "presentat": "presentazione", "iscritt": "iscrizione", "trascritt": "trascrizione",
                   "eseguit": "esecuzione", "compiut": "compimento", "riassunt": "riassunzione", "rinnovat": "rinnovazione",
                   "integrat": "integrazione", "impugnat": "impugnazione", "convalidat": "convalida",
                   "propost": "", "fatt": "", "promoss": "", "espletat": "espletamento"}


def _reflexive(verb: str) -> str:
    return verb[:-3] + "rsi" if verb.endswith("rre") else verb[:-2] + "rsi"     # proporre -> proporsi, notificarsi


_VERB_FORMS = {**{v: v for v in VERB_ACTS}, **{_reflexive(v): v for v in VERB_ACTS}}
_VERB_ALT = "|".join(sorted(_VERB_FORMS, key=len, reverse=True))
# «decreto motivato da emettere entro …», «atto di citazione da notificarsi nel termine …»: the verb right before it
_DA_VERB_RE = re.compile(r"\b(?P<noun>[a-zà-ü]{4,})(?:\s+[a-zà-ü]+)?\s+da\s+(?P<verb>" + _VERB_ALT + r")\s*,?\s*$",
                         re.IGNORECASE)
# «il ricorso deve essere depositato … entro», «la richiesta è presentata … entro»
_PASSIVE_RE = re.compile(r"\b(?:essere|è|sono|viene|vengono|va|vanno|stat[oaie])\s+(?P<stem>"
                         + "|".join(sorted(PARTICIPLE_ACTS, key=len, reverse=True)) + r")[oaie]\b", re.IGNORECASE)
# «ingiunge … di pagare la somma o di consegnare la cosa … nel termine di …», «il convenuto deve costituirsi entro …»
_INFINITIVE_RE = re.compile(r"\b(?:di|a|ad|deve|devono|può|possono|dovrà|dovranno)\s+(?P<verb>" + _VERB_ALT + r")\b(?:\s+(?P<obj>(?:il|lo|la|i|gli|le)\s+"
                            + _W + r"|l['’]\s*" + _W + r"|(?:" + "|".join(SUBJECT_NOUNS) + r")\b))?", re.IGNORECASE)
# «… nello stesso termine può essere fatta opposizione»: one more act of the term before it
_SAME_TERM_RE = re.compile(r"\b(?:nello|nel|entro\s+lo|entro\s+il)\s+(?:stesso|medesimo)\s+termine\b", re.IGNORECASE)
_ARTICLES = {"il": "del", "lo": "dello", "la": "della", "i": "dei", "gli": "degli", "le": "delle", "l'": "dell'",
             "l’": "dell'"}
_FEMININE_PLURALS = ("memorie", "prove", "indagini", "comparse", "somme", "cose", "parti", "spese")


def _of(noun: str, article: str = "") -> str:
    """«del decreto», «della somma», «dell'appello», «dei motivi», «delle memorie»: the article the text gives the noun,
    else the one its ending asks for."""
    noun = noun.lower().strip()
    if article:
        prep = _ARTICLES.get(article.lower().strip(), "di")
        return f"{prep}{noun}" if prep.endswith("'") else f"{prep} {noun}"
    head = noun.split()[0] if noun else ""
    vowel = head[:1] in "aeiouàèéìòù"
    if head.endswith("ioni") or head in _FEMININE_PLURALS:
        return f"delle {noun}"
    if head.endswith("i"):
        return f"{'degli' if vowel or re.match(r's[^aeiou]|z|gn|ps', head) else 'dei'} {noun}"
    if vowel:
        return f"dell'{noun}"
    if head.endswith(("a", "ione", "tà")):
        return f"della {noun}"
    return f"{'dello' if re.match(r's[^aeiou]|z|gn|ps', head) else 'del'} {noun}"


def _object(phrase: str) -> Tuple[str, str]:
    """(article, noun) of an infinitive's object («la somma» -> («la», «somma»))."""
    m = re.match(r"\s*(?:(il|lo|la|i|gli|le)\s+|(l['’])\s*)?(" + _W + r")", phrase or "", re.IGNORECASE)
    return ((m.group(1) or m.group(2) or ""), m.group(3)) if m else ("", "")


def owner(t: str, floor: int, at: int, ceiling: int, right: int = -1) -> str:
    """v67: the act the number at t[at] is the term of -- the noun phrase that owns its marker, read leftwards in
    t[floor:at] (floor: its clause's start, or the term before it in the sentence):
      «<noun> [adj] da <verb> entro N»          -> «<the verb's act> del <noun>» (decreto … da emettere: emissione del
                                                   decreto)
      «<act> … essere <participle> … entro N»   -> «<the participle's act> del <act>» (deposito del ricorso); the act itself
                                                   for «proposta / fatta»
      «di <verb> <object> … nel termine di N»   -> «<the verb's act> della <object>», each (pagamento della somma)
      «<acts> … entro / decorsi N»              -> the acts (acts())
    with none on its left, the acts after its own words (right: past its «dal …») up to ceiling (a fronted or set-off
    term: «…, entro trenta giorni dalla ricezione, richiede l'archiviazione»). "" when none."""
    left = t[floor:at]
    m = _DA_VERB_RE.search(left)
    if m:
        act = VERB_ACTS[_VERB_FORMS[m.group("verb").lower()]]
        return f"{act} {_of(m.group('noun'))}" if act else m.group("noun").lower()
    passive = list(_PASSIVE_RE.finditer(left))
    if passive:
        last = passive[-1]
        act, named = PARTICIPLE_ACTS[last.group("stem").lower()], acts(left[:last.start()])
        if named:
            return ", ".join(f"{act} {_of(a)}" for a in named.split(", ")) if act else named
        if act:
            return act
    out = []
    for i in _INFINITIVE_RE.finditer(left):
        act = VERB_ACTS[_VERB_FORMS[i.group("verb").lower()]]
        article, noun = _object(i.group("obj") or "")
        if act:
            name = f"{act} {_of(noun, article)}" if noun else act
        else:
            name = acts(i.group("obj") or "")
        if name and name not in out:
            out.append(name)
    if out:
        return ", ".join(out)
    return acts(left) or acts(t[max(at, right):ceiling])


def _sentence_of(t: str, at: int) -> Tuple[int, int]:
    starts = [m.end() for m in _SENTENCE_STOP_RE.finditer(t, 0, at)]
    after = _SENTENCE_STOP_RE.search(t, at)
    return (starts[-1] if starts else 0), (after.start() if after else len(t))


def termini(commi: Sequence[Tuple[Optional[Any], str]]) -> List[Dict[str, Any]]:
    """v65, v66: every term the article's commi state, each {comma, subject, qty, unit, text, sentence, when, since,
    letter?, ambiguous?}, in each finder's form -- «entro / nel termine di N …» (_TERM_RE), «il termine per <atto> è di
    N …» (subject: the acts, acts()), «è anche di N … il termine per <atto>», a lettered list «è: a) di N …, <when>; b)
    …» (one term per letter, its condition clause its when), «il termine è di N …» (no act), «decorsi N …». v67: a
    subject found by the clause otherwise is the act that owns the number (owner(); «nello stesso termine» adds the act
    it names to the term before it); «when» the condition the term holds under, «since» the «dal …»
    after its number; «sentence» its sentence (a lettered item: the list's head and the item). Two different numbers
    for the same subject in one comma, with no condition telling them apart: each «ambiguous» (marked_ambiguous)."""
    out: List[Dict[str, Any]] = []
    for k, raw in commi or []:
        t = " ".join(unicodedata.normalize("NFC", str(raw or "")).split())
        hits: Dict[int, Dict[str, Any]] = {}
        # v67: where each number's own words end -- the next one's owner is read after them
        finders = (_TERM_RE, _ELAPSED_RE, _BEFORE_TERM_RE, _PER_RE, _INVERTED_RE, _BARE_RE)
        ends = sorted(m.end() for r in finders for m in r.finditer(t))
        starts = sorted(m.start() for r in finders for m in r.finditer(t))

        def floor_of(at: int, clause: int) -> int:
            return max([clause] + [e for e in ends if e <= at])

        def ceiling_of(at: int, clause_end: int) -> int:
            return min([clause_end] + [s for s in starts if s > at])

        def clause_of(at: int) -> Tuple[int, int]:
            stops = [x.end() for x in _CLAUSE_STOP_RE.finditer(t, 0, at)]
            after = _CLAUSE_STOP_RE.search(t, at)
            return (stops[-1] if stops else 0), (after.start() if after else len(t))

        def owned(m: "re.Match") -> str:            # v67: the act that owns the number of a match (owner())
            c0, c1 = clause_of(m.start())
            since = _SINCE_RE.match(t, m.end())
            return owner(t, floor_of(m.start(), c0), m.start(), ceiling_of(m.end(), c1), since.end() if since else m.end())

        def put(at: int, q: str, u: str, subject: str, sentence: str, when: str = "", letter: str = "",
                end: int = -1) -> None:
            if at in hits:
                return
            since = _SINCE_RE.match(t, end) if end >= 0 else None
            qty = _n(q)
            hits[at] = {"comma": _label(k), "subject": subject, "qty": qty if qty is not None else q, "unit": u,
                        "text": f"{q} {u}", "sentence": sentence.strip(), "when": when.strip().rstrip(".;,"),
                        "since": (since.group(1) + since.group(2)).strip() if since else "",
                        **({"letter": letter} if letter else {})}
        for head in _LIST_HEAD_RE.finditer(t):
            s0, _ = _sentence_of(t, head.start())
            nxt = _LIST_HEAD_RE.search(t, head.end())
            body = t[head.end():nxt.start() if nxt else len(t)]
            for item in _LIST_ITEM_RE.finditer(body):
                q, u = _quantity(item, "c")
                at = head.end() + item.start(item.group("cq1") and "cq1" or "cu2")
                put(at, q, u, acts(head.group("act")), f"{t[s0:head.end()]} {item.group(0).strip()}",
                    when=item.group("when"), letter=item.group("letter"))
        for pattern, p, act in ((_PER_RE, "a", True), (_INVERTED_RE, "b", True), (_BARE_RE, "d", False),
                                (_ELAPSED_RE, "e", False), (_BEFORE_TERM_RE, "f", False)):
            for m in pattern.finditer(t):
                q, u = _quantity(m, p)
                at = m.start(f"{p}q1") if m.group(f"{p}q1") else m.start(f"{p}q2")
                s0, s1 = _sentence_of(t, m.start())
                when = _WHEN_RE.match(t[s0:s1])
                if act:
                    subject = acts(m.group("act"))
                elif p in ("e", "f"):                       # v67: «l'appello … non possono proporsi dopo decorsi»;
                    subject = owned(m)                      # v69: «il convenuto deve costituirsi … almeno N … prima»
                else:
                    subject = ""
                put(at, q, u, subject, t[s0:s1], when.group(1) if when else "", end=m.end())
                if p == "f" and not hits[at]["text"].startswith("almeno "):
                    hits[at]["text"] = f"almeno {hits[at]['text']}"      # v69: a minimum, never the date itself
        for m in _TERM_RE.finditer(t):
            s0, s1 = _sentence_of(t, m.start())
            when = _WHEN_RE.match(t[s0:s1])
            put(m.start(1), m.group(1), m.group(2).lower(), owned(m), t[s0:s1], when.group(1) if when else "",
                end=m.end())
        for m in _SAME_TERM_RE.finditer(t):            # v67: «nello stesso termine può essere fatta opposizione»
            s0, _ = _sentence_of(t, m.start())
            before = [at for at in hits if s0 <= at < m.start()]
            named = acts(t[m.end():clause_of(m.end())[1]])
            if before and named:
                term = hits[max(before)]
                extra = [a for a in named.split(", ") if a not in term["subject"].split(", ")]
                term["subject"] = ", ".join(([term["subject"]] if term["subject"] else []) + extra)
        out += [hits[at] for at in sorted(hits)]
    return marked_ambiguous(out)


def marked_ambiguous(terms: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v66: two different numbers for the same subject in one comma, no condition (when) telling them apart (641 c.p.c.,
    co. 1: thirty and forty days, one sentence and its one «Se …»): each of them «ambiguous» -- no reader states its
    number, each quotes its sentence."""
    groups: Dict[Tuple[str, str, str, str], List[Dict[str, Any]]] = {}
    for t in terms:                 # v69: two numbers run from two events («prima dell'udienza …», «dal …») are two terms
        groups.setdefault((t["comma"], t["subject"], t.get("when") or "", t.get("since") or ""), []).append(t)
    for group in groups.values():
        if len({(x["qty"], x["unit"]) for x in group}) > 1:
            for x in group:
                x["ambiguous"] = True
    return terms


def decorrenze(commi: Sequence[Tuple[Optional[Any], str]]) -> List[Dict[str, Any]]:
    """v66: where the article's terms run from -- a lettered list after «decorrono:» (one entry per letter), or a
    sentence «… decorr… dal / dalla …» -- each {comma, letter?, text}, the text the law's own. v67: a sentence once,
    however many «decorr…» it holds (326 c.p.c., co. 1: «decorrono dalla notificazione …», «il termine decorre dal
    giorno …»)."""
    out: List[Dict[str, Any]] = []
    for k, raw in commi or []:
        t = " ".join(unicodedata.normalize("NFC", str(raw or "")).split())
        head = _FROM_LIST_RE.search(t)
        if head:
            s0, _ = _sentence_of(t, head.start())
            out.append({"comma": _label(k), "text": t[s0:head.end()].strip()})
            out += [{"comma": _label(k), "letter": m.group(1), "text": m.group(2).strip().rstrip(".;")}
                    for m in _FROM_ITEM_RE.finditer(t[head.end():])]
            continue
        for m in _FROM_SENTENCE_RE.finditer(t):
            s0, s1 = _sentence_of(t, m.start())
            entry = {"comma": _label(k), "text": t[s0:s1].strip()}
            if entry not in out:
                out.append(entry)
    return out


def term_text(term: Dict[str, Any]) -> str:
    """v65: a term as one line -- «appello: trenta giorni (co. 1)»; v66: its «dal …» and its condition with it."""
    head = f"{term.get('subject')}: " if term.get("subject") else ""
    since = f" {term['since']}" if term.get("since") else ""
    when = f", {term['when'][:1].lower() + term['when'][1:]}" if term.get("when") else ""
    letter = f", lett. {term['letter']})" if term.get("letter") else ")"
    return f"{head}{term.get('text')}{since}{when} ({term.get('comma')}{letter}"


# --- v66: the one accessor every reader of a stored term calls -------------------------------------------------------------

def terms_of(facts: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The facts' terms (store v3); [] for none."""
    return [t for t in (facts or {}).get("termini") or [] if isinstance(t, dict)]


def _words(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-zà-ü]{4,}", str(text or "").lower())}


def terms_for(facts: Optional[Dict[str, Any]], subject: str = "") -> List[Dict[str, Any]]:
    """The article's terms for a subject: those whose subject holds its first word, its act («opposizione al decreto
    ingiuntivo» -> 641 c.p.c.'s opposizione, never the «emissione del decreto» -- v67), else those sharing any word
    with it («impugnazione della sentenza penale» -> the 585 c.p.p. impugnazione's three); with none such, or no
    subject asked, every term when they all say one number, else []."""
    terms = terms_of(facts)
    words = _words(subject)
    head = next(iter(w[:6] for w in re.findall(r"[a-zà-ü]{4,}", str(subject or "").lower())), "")
    hits = [t for t in terms if head and head in _words(t.get("subject"))] or \
        ([t for t in terms if words & _words(t.get("subject"))] if words else [])
    if hits:
        return hits
    return terms if len({(t["qty"], t["unit"]) for t in terms}) == 1 else []


def term_lines(facts: Optional[Dict[str, Any]], label: str, subject: str = "") -> Tuple[List[str], Dict[str, Any]]:
    """v66: (the lines a reader shows for the article's terms -- for the subject when given --, {"ambiguous", "numbers",
    "terms"}): one line per term, «appello: trenta giorni (art. 325, co. 1, c.p.c.)»; an ambiguous one never states its
    number -- its sentence quoted once, with AMBIGUOUS_TERM. Every reader (the quesito, §4's «Termini», the impugnazione
    closed list) calls this, never the numbers of its own table."""
    from . import interaction_copy as C
    terms = terms_for(facts, subject) if subject else terms_of(facts)
    number, _, code = str(label or "").partition(" ")
    lines, quoted = [], set()
    for t in terms:
        where = f"art. {number}, {t.get('comma')}" + (f", lett. {t['letter']})" if t.get("letter") else "") + f", {code}"
        if t.get("ambiguous"):
            if t.get("sentence") and t["sentence"] not in quoted:
                quoted.add(t["sentence"])
                lines.append(C.fill(C.AMBIGUOUS_TERM, sentence=t["sentence"], where=where) or "")
            continue
        head = f"{t['subject']}: " if t.get("subject") else ""
        since = f" {t['since']}" if t.get("since") else ""
        when = f", {t['when'][:1].lower() + t['when'][1:]}" if t.get("when") else ""
        lines.append(f"{C.capital(head + t['text'])}{since}{when} ({where})")
    numbers = {(t["qty"], t["unit"]) for t in terms if not t.get("ambiguous")}
    return [x for x in lines if x], {"ambiguous": any(t.get("ambiguous") for t in terms), "numbers": numbers,
                                     "terms": terms}


def from_lines(facts: Optional[Dict[str, Any]], label: str) -> List[str]:
    """v66: the article's «decorrenza» lines from the store (decorrenze): a list's head and its letters, or a
    sentence, quoted as the law gives it."""
    from . import interaction_copy as C
    number, _, code = str(label or "").partition(" ")
    out = []
    for d in (facts or {}).get("decorrenze") or []:
        where = f"art. {number}, {d.get('comma')}" + (f", lett. {d['letter']})" if d.get("letter") else "") + f", {code}"
        out.append(C.fill(C.DECORRENZA_LINE, text=d.get("text"), where=where) or "")
    return [x for x in out if x]


def from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    from . import rule_quote
    ref = row.get("_article")
    if not ref or _code(ref[1]) not in CODES:
        return {}
    return extract(str(ref[0]), str(ref[1]), rule_quote.row_commi(row, suffixed=True))   # v67: «1-bis» its own


# v60s (6): the markers the corpus leaves in an article's text -- «(abrogato)», «(soppresso)», footnotes «(3)», «[196]» --
# never reach the store: every string of the facts cleaned (rule_quote.clean_text and these), a list item left empty out
_STORE_MARKER_RE = re.compile(r"\s*[(\[]\s*(?:abrogat[oaie]|soppress[oaie]|omissis|\d{1,4})\s*[)\]]", re.IGNORECASE)


def clean_facts(value: Any) -> Any:
    """The facts with every string free of the corpus' markers (_STORE_MARKER_RE, rule_quote.clean_text); a dict's or a
    list's entry whose text is left empty goes."""
    from . import rule_quote
    if isinstance(value, str):
        return " ".join(rule_quote.clean_text(_STORE_MARKER_RE.sub("", value)).split())
    if isinstance(value, list):
        out = [clean_facts(v) for v in value]
        return [v for v in out if not (v == "" or (isinstance(v, dict) and "text" in v and not v["text"]))]
    if isinstance(value, dict):
        return {k: clean_facts(v) for k, v in value.items()}
    return value


LOOKUP_QUERY = ("UNWIND $items AS it MATCH (d:Document {id: it.doc})-[:CONTAINS]->(s:Section) "
                "WHERE s.name = it.num OR s.name = it.num + '.0.0' "
                f"RETURN it.doc AS doc, it.num AS num, s.{PROPERTY} AS facts, s.{VERSION_PROPERTY} AS version")


def lookup(session, rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """{(doc id, number): facts} the store holds at the current VERSION."""
    from .topic_groups import unit_key
    items = sorted({unit_key(r) for r in rows if all(unit_key(r))})
    if not items:
        return {}
    out = {}
    for rec in session.run(LOOKUP_QUERY, items=[{"doc": d, "num": n} for d, n in items]).data():
        if rec.get("facts") and rec.get("version") in READ_VERSIONS:
            try:
                out[(rec["doc"], rec["num"])] = json.loads(rec["facts"])
            except (TypeError, ValueError):
                continue
    return out


def attach(session, rows: List[Dict[str, Any]]) -> Dict[str, int]:
    """In place: each row of CODES its "_proc" -- the store's, else computed from its text. {"store", "computed"}."""
    from .topic_groups import unit_key
    mine = [r for r in rows if r.get("_article") and _code(r["_article"][1]) in CODES]
    stored = lookup(session, mine) if session is not None and mine else {}
    counts = {"store": 0, "computed": 0}
    for row in mine:
        facts = stored.get(unit_key(row))
        if facts:
            if facts.get("querela") and "ufficio_condizioni" not in facts:     # a v1 entry: its conditions from the text
                computed = from_row(row)
                facts = {**facts, "ufficio_condizioni": computed.get("ufficio_condizioni") or [],
                         "querela_comma": computed.get("querela_comma") or ""}
            row["_proc"] = {**facts, "source": "store"}
            counts["store"] += 1
            continue
        facts = from_row(row)
        if facts:
            row["_proc"] = {**facts, "source": "computed"}
            counts["computed"] += 1
    from .verbose_logger import vlog
    vlog("procedural_facts", {"rows": len(mine), **counts})
    return counts


def by_number(rows: Iterable[Dict[str, Any]], code: str = "c.p.") -> Dict[str, Dict[str, Any]]:
    """{article number: facts} of the rows of one code ("_proc", else computed now)."""
    out = {}
    for row in rows or []:
        ref = row.get("_article")
        if ref and _code(ref[1]) == code:
            facts = row.get("_proc") or from_row(row)
            if facts and code != "c.p." and facts.get("version") != VERSION:     # v65: an older entry, read now
                from . import rule_quote
                commi = rule_quote.row_commi(row, suffixed=True)
                found, since = termini(commi), decorrenze(commi)
                facts = {**facts, **({"termini": found} if found else {}), **({"decorrenze": since} if since else {})}
            if facts:
                out.setdefault(str(ref[0]), facts)
    return out


# --- §4 and the writer's sentences --------------------------------------------------------------------------------------

PROVATO, DA_PROVARE, NON_RISULTA = "provato", "da provare", "non risulta"


def ufficio_verdict(statuses: Sequence[str]) -> str:
    """v60h (1): the line's verdict from its conditions' record statuses -- one provato: d'ufficio; every one non
    risulta: a querela; else «da provare» (a querela unless a condition is proved). "" with no status."""
    statuses = [s for s in statuses or [] if s]
    if not statuses:
        return ""
    if PROVATO in statuses:
        return PROVATO
    if all(s == NON_RISULTA for s in statuses):
        return NON_RISULTA
    return DA_PROVARE


def with_statuses(facts: Dict[str, Any], recorded: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """The offence's facts with each d'ufficio condition's record status (the record's "procedibilita" entries, by
    their text; a condition the record does not hold: da provare) and the verdict ("ufficio_esito")."""
    conds = list(facts.get("ufficio_condizioni") or [])
    if not conds:
        return facts
    by_text = {" ".join(str(r.get("text") or "").lower().split()): r for r in recorded or []}
    out = []
    for c in conds:
        r = by_text.get(" ".join(c["text"].lower().split())) or {}
        out.append({**c, "status": r.get("status") or DA_PROVARE, "why": r.get("why") or ""})
    return {**facts, "ufficio_condizioni": out, "ufficio_esito": ufficio_verdict([c["status"] for c in out])}


def _querela_comma(facts: Dict[str, Any]) -> str:
    if facts.get("querela_comma"):
        return str(facts["querela_comma"])
    m = re.search(r"\((co\. \d+|testo)\)", str(facts.get("procedibilita") or ""))
    return m.group(1) if m else "testo"


def procedibilita_text(facts: Dict[str, Any]) -> str:
    """The «procedibile …» part: with d'ufficio conditions, «a querela della persona offesa (co. N); d'ufficio se … (co.
    M: status); oppure … — esito: …» (interaction_copy's PROCEDURE_CONDITION*); else the facts' own text."""
    from . import interaction_copy as C
    conds = facts.get("ufficio_condizioni") or []
    if not (facts.get("querela") and conds):
        return C.fill(C.PROCEDURE_PROCEDIBILITA, text=facts.get("procedibilita")) or ""
    items = []
    for c in conds:
        where = c["comma"] + (f": {c['status']}" if c.get("status") else "")
        items.append(C.fill(C.PROCEDURE_CONDITION, condition=f"{c.get('opener') or ''} {c['text']}".strip(),
                            where=where) or c["text"])
    text = C.fill(C.PROCEDURE_CONDITIONAL, comma=_querela_comma(facts),
                  conditions=C.PROCEDURE_CONDITION_JOIN.join(items)) or ""
    verdict = C.PROCEDURE_VERDICTS.get(facts.get("ufficio_esito") or "")
    return text + (C.fill(C.PROCEDURE_VERDICT, verdict=verdict) or "" if verdict else "")


def line(number: str, rubrica: str, facts: Dict[str, Any], procedibilita: bool = True) -> str:
    """The «Procedibilità e competenza» line of one offence, from interaction_copy's PROCEDURE_* parts. v60h (1): with
    d'ufficio conditions, each named with its record status and the line's verdict (procedibilita_text). v62: the
    competenza by art. 5 c.p.p. first (court); procedibilita False (a querela the case states, at the trial): no
    procedibilità nor querela term."""
    from . import interaction_copy as C
    name = " ".join(str(rubrica or "").split()).rstrip(" .") or "Reato"
    parts = [C.fill(C.PROCEDURE_HEAD, name=name[:1].upper() + name[1:], number=number),
             procedibilita_text(facts) if procedibilita else None,
             C.fill(C.PROCEDURE_QUERELA, term=facts.get("querela_termine"), remissione=facts.get("remissione"))
             if facts.get("querela") and procedibilita else None,
             C.fill(C.PROCEDURE_COMPETENZA, text=court(number, facts)),
             C.fill(C.PROCEDURE_PENA, text=facts.get("pena"))]
    head, rest = parts[0], [p for p in parts[1:] if p]
    if not rest:
        rest = [C.PROCEDURE_UNKNOWN]
    return f"- {head}: " + C.PROCEDURE_JOIN.join(rest) + "."


_CLAIM_UFFICIO = re.compile(r"procedibil\w* d['’]ufficio|si procede d['’]ufficio|perseguibil\w* d['’]ufficio", re.IGNORECASE)
_CLAIM_QUERELA = re.compile(r"(?:procedibil|perseguibil|punibil)\w* a querela|si procede a querela", re.IGNORECASE)
_EXCEPTION = re.compile(r"\b(?:aggravant|salvo|tranne|eccett|nei casi)", re.IGNORECASE)
# v79.10.3 (C1, review): a sentence offering the remissione della querela or the 162-ter extinction -- only a querela
# offence's, the store's (lawyer_method.procedibility_out on the method's path); one denying it is left alone
_CLAIM_REMISSION = re.compile(r"\bremission\w*\s+(?:della\s+|di\s+)?querel\w*|\b162-ter\b|\bcondotte\s+riparatorie\b",
                              re.IGNORECASE)
_REMISSION_DENIED = re.compile(r"\bnon\b|\besclus\w*|\bpreclus\w*|\binammissibil\w*|\birrevocabil\w*", re.IGNORECASE)
_COURTS = (("assise", "Corte d'assise"), ("giudice di pace", "giudice di pace"), ("tribunale", "tribunale"))
_TERM_CLAIM = re.compile(r"querela[^.]{0,60}?entro (?:il termine di )?" + _QTY + r" (giorni|mesi)", re.IGNORECASE)
_SENTENCE_RE = re.compile(r"[^.!?\n]+(?:[.!?]+|$)")


def _conflict(sentence: str, facts: Dict[str, Any]) -> str:
    """Why the sentence contradicts the facts ("" when it does not)."""
    procedibilita = str(facts.get("procedibilita") or "")
    if procedibilita:
        proved = facts.get("querela") and facts.get("ufficio_esito") == PROVATO     # v60h (1): a condition proved
        if proved and _CLAIM_QUERELA.search(sentence) and not _EXCEPTION.search(sentence):
            return "a querela, il record: d'ufficio (condizione provata)"
        if facts.get("querela") and not proved and _CLAIM_UFFICIO.search(sentence) and not _EXCEPTION.search(sentence):
            return "d'ufficio, il testo: a querela"
        if not facts.get("querela") and _CLAIM_QUERELA.search(sentence):
            return "a querela, il testo: d'ufficio"
        irrevocable = str(facts.get("remissione") or "").startswith("querela irrevocabile")
        if (not facts.get("querela") or proved or irrevocable) and _CLAIM_REMISSION.search(sentence) \
                and not _REMISSION_DENIED.search(sentence):          # v79.10.3 (C1, review): the prose path too
            return "remissione o 162-ter, il testo: " + ("querela irrevocabile" if irrevocable and not proved
                                                         and facts.get("querela") else "d'ufficio")
    court = str(facts.get("competenza") or "").lower()
    if court and re.search(r"competen", sentence, re.IGNORECASE):
        said = next((name for key, name in _COURTS if key in sentence.lower()), "")
        if said and said.lower().split()[-1] not in court:
            return f"competenza {said}, il testo: {facts['competenza']}"
    term = _TERM_CLAIM.search(sentence)
    stated = str(facts.get("querela_termine") or "")
    if term and stated:
        said = f"{term.group(1)} {term.group(2)}".lower()
        if _n(term.group(1)) is not None and not stated.lower().startswith(said) and not re.match(
                rf"{_n(term.group(1))}\b", stated):
            return f"termine di querela {said}, il testo: {stated}"
    return ""


def consistency(sections: List[Tuple[str, str]], facts: Dict[str, Dict[str, Any]], rubriche: Dict[str, str],
                log: List[Dict[str, Any]], skip: Sequence[str] = ()) -> List[Tuple[str, str]]:
    """Each writer sentence on procedibilità, querela, remissione or competenza of an offence the facts hold, checked
    against them: a contradicting one becomes the offence's line in a sentence (PROCEDURE_FIXED, once per section), a
    second one in the same section is dropped. The article is the one the sentence cites, else -- one offence only --
    that one. skip: section titles left alone (§4, rendered from the facts)."""
    from . import interaction_copy as C
    if not facts:
        return sections
    out = []
    for title, text in sections:
        if title in skip or not text:
            out.append((title, text))
            continue
        from .plan_then_write import _sentence_spans
        given: set = set()
        pieces = []
        last = 0
        for start, end in _sentence_spans(text):
            sentence = text[start:end]
            cited = [n for n in re.findall(r"\bart(?:icolo|\.)?\s*(\d+(?:-[a-z]+)?)\s*(?:,\s*co\.\s*\d+,?\s*)?c\.\s*p\.(?!\s*p)",
                                           sentence, re.IGNORECASE) if n in facts]
            number = cited[0] if cited else (next(iter(facts)) if len(facts) == 1 else "")
            why = _conflict(sentence, facts[number]) if number else ""
            if not why:
                continue
            fixed = ""
            if number not in given:
                given.add(number)
                body = line(number, rubriche.get(number, ""), facts[number])[2:].rstrip(".")
                fixed = C.fill(C.PROCEDURE_FIXED, text=body) or ""
            pieces.append((start, end, fixed))
            log.append({"section": title, "article": f"{number} c.p.", "why": why, "sentence": sentence.strip()[:200],
                        "now": fixed[:200]})
        if pieces:
            rebuilt = []
            for s, e, now in pieces:
                rebuilt.append(text[last:s])
                rebuilt.append(now)
                last = e
            rebuilt.append(text[last:])
            text = re.sub(r"[ \t]{2,}", " ", "".join(rebuilt)).replace(" \n", "\n").strip(" ")
        out.append((title, text))
    return out
