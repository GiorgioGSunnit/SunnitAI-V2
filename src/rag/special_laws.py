"""v65: the special-law codes (data/special_laws.json: the codice della strada, the codice del consumo, the testi
unici) a case admits only with their domain signal in the first read -- vehicles and driving for the codice della
strada; a professional seller and a consumer for the codice del consumo; …

  signals(text)               {law id: the signal is in the text} for every law of the table
  law_of(code)                the table's law a row's code or document name belongs to; None for an ordinary code
  gate(rows, text, named)     (the rows admitted, the labels dropped): a special law's article without its signal is no
                              candidate -- the user's named ones aside
  term_allowed(term, found)   a computed term line of a special law only with its signal (the consumer code's terms)
  read_text(read)             the first read's words the signals are read in: its facts, goal, hypotheses, parties,
                              alternatives
  labour_gate(rows, labour, named)  v69: the labour rito's articles (c.p.c. 409-447) only when the case's rito is
                              «lavoro» -- the user's named ones aside, the way a special law needs its signal

No case-specific logic: the table holds the codes and their domains; every rule reads the first read's own words.
"""

import json
import logging
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LAWS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                         "special_laws.json")
_LAWS: Dict[str, Any] = {}


def laws() -> List[Dict[str, Any]]:
    """The table's laws, each with its compiled expressions (read once; [] when missing, logged)."""
    if "list" not in _LAWS:
        out = []
        try:
            with open(LAWS_FILE, encoding="utf-8") as f:
                raw = json.load(f).get("laws") or []
        except (OSError, ValueError) as exc:
            logger.warning("special_laws: %s not read: %s", LAWS_FILE, exc)
            raw = []
        for law in raw:
            try:
                signals = [law["signal"]] if law.get("signal") else list(law.get("signal_all") or [])
                out.append({**law, "_code": re.compile(law["code"], re.IGNORECASE),
                            "_signals": [re.compile(x, re.IGNORECASE) for x in signals]})
            except (KeyError, re.error) as exc:
                logger.warning("special_laws: %r skipped: %s", law.get("id"), exc)
        _LAWS["list"] = out
    return _LAWS["list"]


def law_of(code: Any) -> Optional[Dict[str, Any]]:
    text = " ".join(str(code or "").split())
    return next((law for law in laws() if text and law["_code"].search(text)), None)


def signals(text: str) -> Dict[str, bool]:
    """{law id: every one of its signal expressions matches the text}."""
    return {law["id"]: bool(law["_signals"]) and all(r.search(text or "") for r in law["_signals"]) for law in laws()}


def read_text(read: Optional[Dict[str, Any]]) -> str:
    """The first read's words: its facts, goal, hypotheses (claims, institutions), parties and their roles, alternatives
    and the client's role."""
    read = read or {}
    parts: List[str] = [str(read.get("goal") or "")] + [str(f) for f in read.get("facts") or []]
    for h in read.get("hypotheses") or []:
        parts += [str(h.get("claim") or "")] + [str(x) for x in h.get("institutions") or []]
    for p in read.get("parties") or []:
        parts += [str(p.get("name") or ""), str(p.get("role") or "")]
    for a in read.get("alternatives") or []:
        parts += [str(a.get("name") or ""), str(a.get("why") or "")]
    parts.append(str((read.get("client") or {}).get("role") or ""))
    return " \n".join(x for x in parts if x)


# v75: an EU (or national) regulation by its number -- «Reg. 2022/2554», «Regolamento (UE) 2016/679»
_REGULATION_RE = re.compile(r"\breg(?:olamento)?\.?\s*(?:\((?:ue|ce|cee)\)\s*)?(?:n\.\s*)?\d{1,4}\s*/\s*\d{2,5}\b|"
                            r"\bregolamento\s+\((?:ue|ce|cee)\)", re.IGNORECASE)
UNLISTED_REGULATION = "regolamento senza un segnale proprio"


def regulation(*names: Any) -> bool:
    """v75: the row's code or document name is a regulation's."""
    return any(_REGULATION_RE.search(" ".join(str(n or "").split())) for n in names)


def gate(rows: Sequence[Dict[str, Any]], text: str, named: set) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(the rows admitted, the labels dropped): a row of a special law whose signal the text lacks goes -- unless the
    user names its article (named: keys). v75: the law is read from the row's code, else its document's name; a
    regulation needs an entry of its own (its own domain signal, never the sector's: DORA's is ICT risk, not «banca») --
    one with no entry goes too."""
    from .interactive_flow import _key, _label
    found = signals(text)
    kept, dropped = [], []
    for r in rows or []:
        ref = r.get("_article")
        doc = str((r.get("d") or {}).get("name") or "")
        law = (law_of(ref[1]) or law_of(doc)) if ref and len(ref) == 2 else None
        unlisted = law is None and ref and len(ref) == 2 and regulation(ref[1], doc)
        if (law and not found.get(law["id"]) or unlisted) and _key(ref) not in named:
            label = f"{_label(ref)} ({law['name'] if law else UNLISTED_REGULATION})"
            if label not in dropped:
                dropped.append(label)
            continue
        kept.append(r)
    return kept, dropped


# v69: «Delle controversie in materia di lavoro» -- an appalto's «lavori» never brings the labour rito's articles
LABOUR_CODE, LABOUR_RANGE = "c.p.c.", (409, 447)


def labour_article(ref: Any) -> bool:
    """v69: the reference is one of c.p.c. 409-447 (their -bis … too)."""
    if not ref or len(ref) != 2:
        return False
    from .interactive_flow import _key
    number, code = _key(ref)
    base = re.match(r"\d+", str(number))
    return code == LABOUR_CODE and bool(base) and LABOUR_RANGE[0] <= int(base.group(0)) <= LABOUR_RANGE[1]


def labour_gate(rows: Sequence[Dict[str, Any]], labour: bool, named: set) -> Tuple[List[Dict[str, Any]], List[str]]:
    """v69: (the rows admitted, the labels dropped): with the case's rito not «lavoro», a c.p.c. 409-447 row goes --
    unless the user names its article (named: keys)."""
    from .interactive_flow import _key, _label
    if labour:
        return list(rows or []), []
    kept, dropped = [], []
    for r in rows or []:
        ref = r.get("_article")
        if labour_article(ref) and _key(ref) not in named:
            if _label(ref) not in dropped:
                dropped.append(_label(ref))
            continue
        kept.append(r)
    return kept, dropped


def term_allowed(term: Dict[str, Any], found: Optional[Dict[str, bool]]) -> bool:
    """A computed term line: kept unless its reference is a special law's and its signal is missing (found None: no
    first read -- kept, today's behaviour)."""
    if found is None:
        return True
    law = law_of(term.get("ref"))
    return not law or bool(found.get(law["id"]))
