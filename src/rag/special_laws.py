"""v65: the special-law codes (data/special_laws.json: the codice della strada, the codice del consumo, the testi
unici) a case admits only with their domain signal in the first read -- vehicles and driving for the codice della
strada; a professional seller and a consumer for the codice del consumo; …

  signals(text)               {law id: the signal is in the text} for every law of the table
  law_of(code)                the table's law a row's code or document name belongs to; None for an ordinary code
  law_of_row(code, doc)       v79.10 (A4, review): a row's -- an exact entry by the document's name only
  gate(rows, text, named)     (the rows admitted, the labels dropped): a special law's article without its signal is no
                              candidate -- the user's named ones aside
  term_allowed(term, found)   a computed term line of a special law only with its signal (the consumer code's terms)
  read_text(read)             the first read's words the signals are read in: its facts, goal, hypotheses, parties,
                              alternatives
  labour_gate(rows, labour, named)  v69: the labour rito's articles (c.p.c. 409-447) only when the case's rito is
                              «lavoro» -- the user's named ones aside, the way a special law needs its signal

No case-specific logic: the table holds the codes and their domains; every rule reads the first read's own words.

v79.10 (A4): a family's own signal (the table's «families»: strong expressions, one suffices, or weak ones, «weak_min»
different ones needed) -- a law without a signal of its own takes its family's, «signal_families» adds one to a law's own:

  families()                  the compiled family signals
  family_signal(family, text) the words holding the family's signal ([] when none: «il Comune» alone is no signal)

v79.10 (A4, archive names): an entry with «expected» (the data/corpus_expected.json law it is) or its own
«archive_names» is found on a row's code or document name only by its archive documents' exact names (whitespace-
normalised equality, corpus_laws.norm), by the sigla the pipeline labels such a document with («c.d.s.» for «Codice
della strada 2026») or by its own citation as the whole text («d.P.R. 309/1990»); its «code»
expression and its number/year read the user's words only (family_gate.naming). The other entries match as before.

  exact(law)                  the entry is matched by its archive names
  law_named(text)             the entry a citation in words names (its expression: a computed term's reference)
  archive_names()             every archive name the table maps (corpus_gaps.py: the names no law maps)
"""

import json
import logging
import os
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

LAWS_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                         "special_laws.json")
_LAWS: Dict[str, Any] = {}


def laws() -> List[Dict[str, Any]]:
    """The table's laws, each with its compiled expressions (read once; [] when missing, logged). v79.10 (A4): each
    with «_families» -- the families whose signal (families()) admits it: its «signal_families», and its own family
    when it has no signal of its own and the family has one."""
    if "list" not in _LAWS:
        out = []
        try:
            with open(LAWS_FILE, encoding="utf-8") as f:
                data = json.load(f)
            raw = data.get("laws") or []
        except (OSError, ValueError) as exc:
            logger.warning("special_laws: %s not read: %s", LAWS_FILE, exc)
            data, raw = {}, []
        _LAWS["families"] = _compile_families(data.get("families") or {})
        for law in raw:
            try:
                signals = [law["signal"]] if law.get("signal") else list(law.get("signal_all") or [])
                fams = list(law.get("signal_families") or [])
                if not signals and law.get("family") in _LAWS["families"] and law["family"] not in fams:
                    fams.append(law["family"])
                out.append({**law, "_code": re.compile(law["code"], re.IGNORECASE),
                            "_signals": [re.compile(x, re.IGNORECASE) for x in signals],
                            "_families": [x for x in fams if x in _LAWS["families"]],
                            "_archive": _archive(law)})
            except (KeyError, re.error) as exc:
                logger.warning("special_laws: %r skipped: %s", law.get("id"), exc)
        _LAWS["list"] = out
    return _LAWS["list"]


def _archive(law: Dict[str, Any]) -> Optional[List[str]]:
    """v79.10 (A4): the entry's archive names, normalised -- its «expected» law's (data/corpus_expected.json) and its
    own «archive_names»; None for an entry with neither (matched by its expression, as before)."""
    if "expected" not in law and "archive_names" not in law:
        return None
    from . import corpus_laws
    names = corpus_laws.archive_names(str(law["expected"])) if law.get("expected") else []
    names += [corpus_laws.norm(x) for x in law.get("archive_names") or [] if corpus_laws.norm(x)]
    return list(dict.fromkeys(names))


def exact(law: Optional[Dict[str, Any]]) -> bool:
    """v79.10 (A4): the entry is matched to documents by its archive names only."""
    return bool(law) and law.get("_archive") is not None


def archive_names() -> List[str]:
    """v79.10 (A4): every archive name the table's entries map."""
    return list(dict.fromkeys(n for law in laws() for n in law.get("_archive") or []))


def _compile_families(raw: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """v79.10 (A4): {family: {"strong": [compiled], "weak": [compiled], "weak_min": n}} -- an entry that does not compile
    is skipped, logged."""
    out: Dict[str, Dict[str, Any]] = {}
    for name, spec in (raw or {}).items():
        try:
            out[name] = {"strong": [re.compile(x, re.IGNORECASE) for x in spec.get("strong") or []],
                         "weak": [re.compile(x, re.IGNORECASE) for x in spec.get("weak") or []],
                         "weak_min": max(1, int(spec.get("weak_min") or 2))}
        except (AttributeError, TypeError, ValueError, re.error) as exc:
            logger.warning("special_laws: family %r skipped: %s", name, exc)
    return out


def families() -> Dict[str, Dict[str, Any]]:
    """v79.10 (A4): the table's family signals (data/special_laws.json «families»), compiled."""
    laws()
    return _LAWS.get("families") or {}


def family_signal(family: str, text: str, act: bool = True) -> List[str]:
    """v79.10 (A4): the words of the text that hold the family's signal -- one strong expression's match, else the
    weak expressions' matches when at least weak_min different ones match; [] otherwise (or no such family). v79.14.1
    (item 3): a family that needs the administrative act (data/administrative_act.json «families»: amministrativo) only
    with one in the same text (family_gate.administrative_act) -- «Comune», «difformità edilizie», «permesso di
    costruire» alone never; act False: the signal as before (what a refusal for lack of the act counts)."""
    spec = families().get(family)
    if not spec or not text:
        return []
    got: List[str] = []
    for r in spec["strong"]:
        m = r.search(text)
        if m:
            got = [m.group(0)]
            break
    if not got:
        weak = [m.group(0) for m in (r.search(text) for r in spec["weak"]) if m]
        got = weak if len(weak) >= spec["weak_min"] else []
    if got and act:
        from . import family_gate
        if family in family_gate.act_table()["families"] and not family_gate.administrative_act(text):
            return []
    return got


def law_of(code: Any, name_only: bool = False, exact_entries: bool = True) -> Optional[Dict[str, Any]]:
    """The table's law of a row's code or document name. v79.10 (A4): an exact entry by its archive names or its own
    citation as the whole text; the others by their expression. v79.10 (A4, review): name_only -- the text is a
    document's name: an exact entry by its archive names only (never a sigla, never a citation: a document named
    «D.lgs. 81/2008» is not the entry); exact_entries False -- the exact entries left out (law_of_row: a row's code
    beside its document's name, which alone says whether the row is the exact entry's)."""
    text = " ".join(str(code or "").split())
    return next((law for law in laws() if text and (exact_entries or not exact(law)) and _matches(law, text, name_only)),
                None)


def law_of_row(code: Any, doc: Any) -> Optional[Dict[str, Any]]:
    """v79.10 (A4, review): the table's law of a row -- by its document's name when it has one (an exact entry by that
    name only: a row coded «c.d.s.» of «Codice della strada commentato 2024» is not the codice della strada's), its code
    for the other entries (first, as before); by its code alone (law_of) when it has no document's name."""
    from . import corpus_laws
    code, doc = corpus_laws.norm(code), corpus_laws.norm(doc)
    if not doc:
        return law_of(code)
    return (law_of(code, exact_entries=False) if code != doc else None) or law_of(doc, name_only=True)


def _matches(law: Dict[str, Any], text: str, name_only: bool = False) -> bool:
    if not exact(law):
        return bool(law["_code"].search(text))
    from . import corpus_laws
    if text in law["_archive"]:
        return True
    if name_only:
        return False
    if corpus_laws.sigla_of(text, law["_archive"]):
        return True
    number, year = corpus_laws.cited(text)
    return bool(number) and (number, year) == (str(law.get("number") or ""), str(law.get("year") or ""))


def signals(text: str) -> Dict[str, bool]:
    """{law id: every one of its signal expressions matches the text}. v79.10 (A4): or one of its families' signal
    (family_signal: a strong expression, or weak_min weak ones)."""
    return {law["id"]: (bool(law["_signals"]) and all(r.search(text or "") for r in law["_signals"]))
            or any(family_signal(f, text or "") for f in law.get("_families") or []) for law in laws()}


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


def gate(rows: Sequence[Dict[str, Any]], text: str, named: set, keep: Optional[Callable[[Dict[str, Any]], bool]] = None
         ) -> Tuple[List[Dict[str, Any]], List[str]]:
    """(the rows admitted, the labels dropped): a row of a special law whose signal the text lacks goes -- unless the
    user names its article (named: keys). v75: the law is read from the row's code, else its document's name; a
    regulation needs an entry of its own (its own domain signal, never the sector's: DORA's is ICT risk, not «banca») --
    one with no entry goes too. v79.10 (A5, review): keep -- a row it returns True for stays all the same (interactive_
    flow: the family gate's override doors, family_gate.rescued, so the safety valve reaches the table's laws)."""
    from .interactive_flow import _key, _label
    found = signals(text)
    kept, dropped = [], []
    for r in rows or []:
        ref = r.get("_article")
        doc = str((r.get("d") or {}).get("name") or "")
        law = law_of_row(ref[1], doc) if ref and len(ref) == 2 else None      # v79.10 (A4, review): by the doc
        unlisted = law is None and ref and len(ref) == 2 and regulation(ref[1], doc)
        if (law and not found.get(law["id"]) or unlisted) and _key(ref) not in named and not (keep and keep(r)):
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
    law = law_of(term.get("ref")) or law_named(term.get("ref"))
    return not law or bool(found.get(law["id"]))


def law_named(text: Any) -> Optional[Dict[str, Any]]:
    """v79.10 (A4): the table's law a citation written in words names («art. 132 Codice del Consumo», a computed term's
    reference) -- every entry by its expression: words, never a document's name (law_of reads those)."""
    text = " ".join(str(text or "").split())
    return next((law for law in laws() if text and law["_code"].search(text)), None)
