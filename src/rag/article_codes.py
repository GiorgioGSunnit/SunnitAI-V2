"""Every article a chat answer cites carries its code or law ("art. 585 c.p.p.", not "art. 585"), so a reader
can tell c.p. from c.p.p. The synthesis prompt asks for it (prompts.synthesis_system_message); this is the
post-check (synthesize_answer): an article cited bare gets the code of the retrieved rows when exactly one
code or law among them has that article -- otherwise it is left as written and logged as ambiguous or
unknown ("article_code_check").
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

from .article_suffix import SUFFIX, normalize_number

_NUMBER = rf"\d+(?:\s*-?\s*{SUFFIX})?(?![0-9A-Za-z])"
# "art. 585", "artt. 585 e 593", "articoli 52, 54 e 55", "dell'art.585"
_RUN_RE = re.compile(rf"\b(?:artt?\.|art\b|articol[oi])\s*{_NUMBER}(?:\s*(?:,|e|ed|o|&)\s*{_NUMBER})*",
                     re.IGNORECASE)
_NUMBER_RE = re.compile(_NUMBER, re.IGNORECASE)
# A code or a law written next to the article.
_CODE = (r"c\.?\s?[cp]\.?(?:\s?[cpa]\.?)?(?![a-z])|c\.?\s?d\.?\s?s\.?(?![a-z])|cost\.|cod\.|codic[ei]|costituzione|"
         r"legge|l\.(?=\s*(?:n\.\s*)?\d)|d\.?\s?lgs|d\.?\s?l\.(?=\s*(?:n\.\s*)?\d)|d\.?\s?p\.?\s?r|r\.?\s?d\.(?=\s*\d)|"
         r"t\.?\s?u\.?(?![a-z])|testo unico|regolament\w*|reg\.|direttiva|dir\.|ccnl|tuel|tuir|disp\.?\s*att|"
         r"preleggi|statuto|convenzione|cedu|trattato|tfue|carta|decreto")
# After the article: optional "comma 2", "lett. b)", then (del/della ...) the code.
_AFTER_RE = re.compile(rf"^(?:\s*,?\s*(?:commi?|co\.|lett\.|lettera|n\.|numero)\s*[\w\-]+\)?)*\s*,?\s*"
                       rf"(?:(?:del|della|dello|dei|delle|dal|dalla|di|in|ex)\s+)?(?:\(\s*)?(?:{_CODE})",
                       re.IGNORECASE)
# Before the article: "c.p.p. art. 585", "(c.p., art. 52)", "Codice penale, art. 52" -- a code's name of a few
# words needs a comma, colon or dash before the article ("art. 1490 c.c. e l'art. 52": 52 is bare).
_BEFORE_RE = re.compile(rf"(?:(?:{_CODE})\.?\s*\(?\s*|(?:{_CODE})[\w.\s']{{0,30}}?\s*[,:–—(-]\s*\(?\s*)$",
                        re.IGNORECASE)

_LAW_KIND = [(re.compile(r"d\.?\s?lgs|decreto legislativo", re.I), "d.lgs."),
             (re.compile(r"d\.?\s?p\.?\s?r|decreto del presidente della repubblica", re.I), "d.p.r."),
             (re.compile(r"d\.?\s?l\.|decreto[- ]legge", re.I), "d.l."),
             (re.compile(r"\blegge\b|\bl\.", re.I), "l.")]


def law_label(document_name: str) -> str:
    """How an article of this document is cited: "l. 241/1990", "d.lgs. 36/2023", else the document's name
    without a trailing edition year, in brackets ("(Testo unico enti locali)")."""
    name = (document_name or "").strip()
    for kind_re, kind in _LAW_KIND:
        if not kind_re.search(name):
            continue
        m = re.search(r"\b(\d{1,4})\s*/\s*((?:19|20)\d{2})\b", name)
        if m:
            return f"{kind} {m.group(1)}/{m.group(2)}"
        number = re.search(r"\bn\.?\s*(\d{1,4})\b", name)
        year = re.search(r"\b((?:19|20)\d{2})\b", name)
        if number and year:
            return f"{kind} {number.group(1)}/{year.group(1)}"
    bare = re.sub(r"\s+(?:19|20)\d{2}$", "", name)
    return f"({bare})" if bare else ""


def _label(document_name: str) -> Optional[str]:
    from .defensive_generation import _CODE_PREFIXES
    name = (document_name or "").strip()
    if not name:
        return None
    for abbr, prefix in _CODE_PREFIXES.items():
        if name.lower().startswith(prefix.lower()) and "commentat" not in name.lower():
            return abbr
    if name.lower().startswith("costituzione"):
        return "Cost."
    return law_label(name)


def codes_by_article(rows: Iterable[Dict[str, Any]]) -> Dict[str, Set[str]]:
    """{article number: the codes or laws among the rows that have it}, from the rows of codes and laws only
    (rulings, books and rows with no article-shaped section are skipped)."""
    from .defensive_generation import _NOT_A_PRIMARY_SOURCE_RE, _article_num
    out: Dict[str, Set[str]] = {}
    for row in rows or []:
        d = row.get("d") if isinstance(row.get("d"), dict) else {}
        s = row.get("s") if isinstance(row.get("s"), dict) else {}
        if d.get("document_type") in ("interpretation", "special") or row.get("_source") == "ruling_support":
            continue
        tagged = row.get("_article")
        if tagged and len(tagged) == 2:          # (number, "c.c.") or (number, the document's name)
            number = normalize_number(str(tagged[0]))
            label = str(tagged[1]) if str(tagged[1]).endswith(".") else _label(str(tagged[1]))
        else:
            name, d_name = str(s.get("name") or ""), str(d.get("name") or "")
            if not re.match(r"^\d+(?:-[a-z]+)?(?:\.|$)", name, re.IGNORECASE) or not d_name \
                    or _NOT_A_PRIMARY_SOURCE_RE.search(d_name):
                continue
            number, label = _article_num(name), _label(d_name)
        if number and label:
            out.setdefault(number, set()).add(label)
    return out


def _has_code(text: str, start: int, end: int) -> bool:
    if _AFTER_RE.match(text[end:end + 80]):
        return True
    return bool(_BEFORE_RE.search(text[max(0, start - 50):start]))


def complete_codes(answer: str, rows: Iterable[Dict[str, Any]]) -> Tuple[str, Dict[str, Any]]:
    """(the answer with each bare article's code added where the rows make it unambiguous, a report:
    {"bare", "completed": ["585 c.p.p.", ...], "ambiguous": ["52: c.p., c.p.p."], "unknown": ["9"]})."""
    report: Dict[str, Any] = {"bare": 0, "completed": [], "ambiguous": [], "unknown": []}
    if not answer:
        return answer, report
    table = codes_by_article(rows)
    inserts: List[Tuple[int, str]] = []
    for run in _RUN_RE.finditer(answer):
        if _has_code(answer, run.start(), run.end()):
            continue
        head = re.match(r"\b(?:artt?\.|art\b|articol[oi])\s*", run.group(0), re.IGNORECASE).end()
        numbers = [(run.start() + head + m.start(), run.start() + head + m.end(), normalize_number(m.group(0)))
                   for m in _NUMBER_RE.finditer(run.group(0)[head:])]
        report["bare"] += len(numbers)
        labels = []
        for _, _, number in numbers:
            found = table.get(number) or set()
            labels.append(next(iter(found)) if len(found) == 1 else None)
            if len(found) > 1:
                report["ambiguous"].append(f"{number}: {', '.join(sorted(found))}")
            elif not found:
                report["unknown"].append(number)
        if labels and all(labels) and len(set(labels)) == 1:
            inserts.append((run.end(), labels[0]))            # "artt. 585 e 593 c.p.p."
            report["completed"] += [f"{n} {labels[0]}" for _, _, n in numbers]
            continue
        for (_, end, number), label in zip(numbers, labels):
            if label:
                inserts.append((end, label))
                report["completed"].append(f"{number} {label}")
    for at, label in sorted(inserts, reverse=True):
        if label.endswith(".") and answer[at:at + 1] == ".":     # "... 593 c.p.p." ends the sentence too
            label = label[:-1]
        answer = answer[:at] + " " + label + answer[at:]
    return answer, report
