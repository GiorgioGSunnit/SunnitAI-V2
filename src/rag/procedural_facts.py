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
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

PROPERTY = "proc_facts"
VERSION_PROPERTY = "proc_version"
VERSION = 2                     # v60h (1): the d'ufficio conditions (ufficio_condizioni) and the querela's comma
READ_VERSIONS = (1, VERSION)    # a v1 entry is read, its conditions computed from the row's text (attach)
CODES = ("c.p.", "c.p.p.", "c.p.a.", "c.p.c.")
QUERELA_DEFAULT = "tre mesi (art. 124 c.p.)"
REMISSIONE_DEFAULT = "remissibile (art. 152 c.p.)"
UFFICIO_DEFAULT = "d'ufficio (regola generale: l'articolo non prevede la querela)"
# d.lgs. 274/2000, art. 4, co. 1, lett. a: the c.p. delitti of the giudice di pace (within that article's limits)
PEACE_JUDGE = {"581", "582", "590", "592", "595", "612", "626", "627", "631", "632", "633", "635", "636", "637", "638",
               "639", "647"}

_NUMBERS = {"un": 1, "uno": 1, "una": 1, "due": 2, "tre": 3, "quattro": 4, "cinque": 5, "sei": 6, "sette": 7, "otto": 8,
            "nove": 9, "dieci": 10, "undici": 11, "dodici": 12, "tredici": 13, "quattordici": 14, "quindici": 15,
            "sedici": 16, "diciassette": 17, "diciotto": 18, "diciannove": 19, "venti": 20, "ventuno": 21,
            "ventidue": 22, "ventitre": 23, "ventitré": 23, "ventiquattro": 24, "venticinque": 25, "ventisei": 26,
            "trenta": 30, "sessanta": 60, "novanta": 90, "centoventi": 120, "centottanta": 180, "quarantacinque": 45,
            "quindicina": 15, "trentasei": 36}
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


def competenza(number: str, max_years: Optional[float], ergastolo: bool) -> str:
    base = re.match(r"\d+", str(number))
    if ergastolo or (max_years is not None and max_years >= 24):
        return "Corte d'assise (art. 5 c.p.p.)"
    if base and base.group(0) in PEACE_JUDGE:
        return "giudice di pace (art. 4 d.lgs. 274/2000, nei limiti ivi previsti)"
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
        for k, t in commi:
            m = _TERM_RE.search(t)
            if m:
                out["termine"] = f"{m.group(1)} {m.group(2)} ({_label(k)})"
                return out
        return {}
    return {}


def from_row(row: Dict[str, Any]) -> Dict[str, Any]:
    from . import rule_quote
    ref = row.get("_article")
    if not ref or _code(ref[1]) not in CODES:
        return {}
    return extract(str(ref[0]), str(ref[1]), rule_quote.row_commi(row))


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


def line(number: str, rubrica: str, facts: Dict[str, Any]) -> str:
    """The «Procedibilità e competenza» line of one offence, from interaction_copy's PROCEDURE_* parts. v60h (1): with
    d'ufficio conditions, each named with its record status and the line's verdict (procedibilita_text)."""
    from . import interaction_copy as C
    name = " ".join(str(rubrica or "").split()).rstrip(" .") or "Reato"
    parts = [C.fill(C.PROCEDURE_HEAD, name=name[:1].upper() + name[1:], number=number),
             procedibilita_text(facts),
             C.fill(C.PROCEDURE_QUERELA, term=facts.get("querela_termine"), remissione=facts.get("remissione"))
             if facts.get("querela") else None,
             C.fill(C.PROCEDURE_COMPETENZA, text=facts.get("competenza")),
             C.fill(C.PROCEDURE_PENA, text=facts.get("pena"))]
    head, rest = parts[0], [p for p in parts[1:] if p]
    if not rest:
        rest = [C.PROCEDURE_UNKNOWN]
    return f"- {head}: " + C.PROCEDURE_JOIN.join(rest) + "."


_CLAIM_UFFICIO = re.compile(r"procedibil\w* d['’]ufficio|si procede d['’]ufficio|perseguibil\w* d['’]ufficio", re.IGNORECASE)
_CLAIM_QUERELA = re.compile(r"(?:procedibil|perseguibil|punibil)\w* a querela|si procede a querela", re.IGNORECASE)
_EXCEPTION = re.compile(r"\b(?:aggravant|salvo|tranne|eccett|nei casi)", re.IGNORECASE)
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
