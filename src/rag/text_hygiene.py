"""v79.12.7 (item 7): the text residue the reader must never see, and the procedural acts the facts narrate. The tables
are data/text_hygiene.json (read once).

  stance_out(text, log)        a parenthesis holding only a ruling's stance on an element («(contro)», «(a favore)»,
                               «(verso l'integrazione: contro)») -- the registry's and the thesis call's wording --
                               dropped outside the tables and the quoted passages; stance_hits lists those left.
  codes_out(text, known, log)  an element code (R1, C2, P3) outside the tables and the quoted passages: its parenthesis
                               gone, a bare code its element's words; a bare R-code no record holds drops its sentence
                               (as a registry label no ruling holds does). codes_hits lists those left.
  internal_notes_out(text,log) a sentence holding a list's draft status («elenco interno», «ancora in bozza») removed.
  orphan_dates(text)           a date left before a closing parenthesis with no head («…, 11.04.2025)»): the spans the
                               citation check removes (lawyer_method.whole_citations, rulings.printed_problems).
  procedural_act(sentence)     the act a sentence narrates (the PM's contestazione, an avviso, a notifica, the rinvio
                               a giudizio, a decreto) -- issue_records.certainty: the act is a certain fact unless what
                               is left once its own words are out is attributed or hedged; act_named(element, act): the
                               element names the act (the only element its sentence grounds).
  cenni_check(text, texts, src, log)
                               a «Cenni» section's rule statement (rule_statements.RULE_VERBS and the table's
                               cenni_rule_verbs) must be grounded in the article it cites -- one comma of its text, or
                               an element of its record --, the section's articles when it names none; else removed.
  main_line_count(text, tesi)  how many times a section prints the main line (tables included): at most once.
  residue_out(text, known, log)
                               the last pass over the delivered answer (interactive_flow.text_residue): codes, stances,
                               internal notes and headless citations, wherever a late step or a path that bypassed
                               compose's checks left them.
"""

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                         "text_hygiene.json")


@lru_cache(maxsize=1)
def data() -> Dict[str, Any]:
    """data/text_hygiene.json ({} when missing or unreadable, logged)."""
    try:
        with open(DATA_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError) as exc:
        logger.warning("text_hygiene: %s not read: %s", DATA_FILE, exc)
        return {}


def _join(patterns: Sequence[str]) -> Optional["re.Pattern"]:
    good = []
    for p in patterns or []:
        try:
            re.compile(p)
            good.append(p)
        except re.error as exc:
            logger.warning("text_hygiene: pattern %r skipped: %s", p, exc)
    return re.compile("|".join(f"(?:{p})" for p in good), re.IGNORECASE) if good else None


_QUOTE_RE = re.compile(r"«[^«»\n]*»")


def _table(line: str) -> bool:
    return line.lstrip().startswith("|")


def _outside_quotes(line: str, fn: Callable[[str], str]) -> str:
    """fn applied to the line's text outside its «…» quotes (a ruling's passage, an element's or a fact's words)."""
    out, at = [], 0
    for m in _QUOTE_RE.finditer(line):
        out += [fn(line[at:m.start()]), m.group(0)]
        at = m.end()
    out.append(fn(line[at:]))
    return "".join(out)


def _tidy(line: str) -> str:
    """A line a removal changed: no space before punctuation, no double space inside it (its indentation kept), no
    empty parenthesis."""
    line = re.sub(r"\(\s*\)", "", line)
    line = re.sub(r"[ \t]+([,.;:)])", r"\1", line)
    line = re.sub(r"([,;:])\s*([.;:])", r"\2", line)
    return re.sub(r"(?<=\S)[ \t]{2,}", " ", line).rstrip()


def _by_line(text: str, fn: Callable[[str], str]) -> str:
    """fn on every line outside the tables, outside the quotes; a changed line tidied; the text unchanged otherwise."""
    if not text:
        return text or ""
    lines = []
    for line in text.split("\n"):
        if _table(line):
            lines.append(line)
            continue
        done = _outside_quotes(line, fn)
        lines.append(_tidy(done) if done != line else line)
    out = "\n".join(lines)
    return out if out != text else text


# --- (a) a ruling's stance on an element ----------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def stance_re() -> Optional["re.Pattern"]:
    d = data()
    words = [w for w in d.get("stance_words") or [] if w]
    if not words:
        return None
    prefixes = "|".join(f"(?:{p})" for p in d.get("stance_prefixes") or [] if p)
    head = rf"(?:(?:{prefixes})\s*:\s*)?" if prefixes else ""
    return re.compile(r"[ \t]*\(\s*" + head + "(?:" + "|".join(rf"{w}" for w in words) + r")\s*\)", re.IGNORECASE)


@lru_cache(maxsize=1)
def stance_part_res() -> Tuple[Optional["re.Pattern"], Optional["re.Pattern"]]:
    """(review) A stance heading or closing a parenthesis that holds more («(contro, Cass. pen., …)», «(…; a favore)»):
    the stance and its separator alone go."""
    d = data()
    words = [w for w in d.get("stance_words") or [] if w]
    if not words:
        return None, None
    prefixes = "|".join(f"(?:{p})" for p in d.get("stance_prefixes") or [] if p)
    head = rf"(?:(?:{prefixes})\s*:\s*)?" if prefixes else ""
    alt = "(?:" + "|".join(words) + ")"
    return (re.compile(r"(?<=\()\s*" + head + alt + r"\s*[,;:]\s*(?=[^)\s])", re.IGNORECASE),
            re.compile(r"\s*[,;]\s*" + head + alt + r"\s*(?=\))", re.IGNORECASE))


def stance_out(text: str, log: Optional[List[str]] = None) -> str:
    rx = stance_re()
    if rx is None or not text:
        return text or ""
    lead, trail = stance_part_res()

    def strip(seg: str) -> str:
        def gone(m: "re.Match") -> str:
            if log is not None:
                log.append(m.group(0).strip())
            return ""
        seg = rx.sub(gone, seg)
        for part in (lead, trail):
            if part is not None:
                seg = part.sub(gone, seg)
        return seg
    return _by_line(text, strip)


def stance_hits(text: str) -> List[str]:
    found: List[str] = []
    stance_out(text, found)
    return found


# --- (a) an element code ----------------------------------------------------------------------------------------------------

_CODE_PAREN_RE = re.compile(r"[ \t]*\(\s*(?:elementi?\s+)?([RCP]\d{1,3}(?:\s*(?:,|e|ed|-|–|/)\s*[RCP]\d{1,3})*)\s*\)")
# (review) «R1-R3», «R1–3», «R1/R2»: a range or a pair of codes, expanded to «R1, R2 e R3» before the strip
_CODE_RANGE_RE = re.compile(r"(?<![\w./«-])([RCP])(\d{1,3})\s*([-–/])\s*([RCP])?(\d{1,3})(?![\w/-])")
_CODE_TOKEN_RE = re.compile(r"(?<![\w./«-])([RCP]\d{1,3})(?![\w/-])")
_R_CODE_RE = re.compile(r"(?<![\w./«-])(R\d{1,3})(?![\w/-])")


def _expand_ranges(seg: str, known: Dict[str, Dict[str, Any]]) -> str:
    """«R1-R3» -> «R1, R2 e R3», «R1/R2» -> «R1 e R2» -- R-codes, or known codes of one letter; a range over twelve
    codes, or mixed letters, as it is."""
    def one(m: "re.Match") -> str:
        a, b, letter = int(m.group(2)), int(m.group(5)), m.group(1)
        if m.group(4) and m.group(4) != letter:
            return m.group(0)
        ids = [f"{letter}{a}", f"{letter}{b}"] if m.group(3) == "/" else (
            [f"{letter}{i}" for i in range(a, b + 1)] if 0 < b - a <= 12 else [])
        if not ids or not (letter == "R" or all(x in known for x in ids)):
            return m.group(0)
        return ", ".join(ids[:-1]) + " e " + ids[-1]
    return _CODE_RANGE_RE.sub(one, seg)


@lru_cache(maxsize=1)
def code_context_re() -> Optional["re.Pattern"]:
    return _join(data().get("code_context_patterns") or [])


def codes_out(text: str, known: Dict[str, Dict[str, Any]], log: Optional[List[str]] = None) -> str:
    """The text's element codes outside the tables and the quotes: a parenthesis of codes gone (each known, or an
    R-code: a requisite's id is never the reader's), a bare known code its element's words («…»); a sentence holding a
    bare R-code no record holds removed -- only when the answer has codes of its own (known), so a text the method did
    not write is read as it is. (review) A range or a pair («R1-R3», «R1/R2») is read as its codes. With no known code
    (a text no record wrote), a parenthesis of R-codes still goes and a sentence using a bare R-code as an element
    (data «code_context_patterns»: «contesta R2», «l'elemento R2») is removed -- an R-code is never the reader's."""
    if not text:
        return text or ""
    if not known:
        return _codes_out_unknown(text, log)
    from .lawyer_method import _by_sentence, _short

    def paren(m: "re.Match") -> str:
        ids = re.findall(r"[RCP]\d{1,3}", m.group(1))
        if ids and all(x in known or x.startswith("R") for x in ids):
            if log is not None:
                log.append(m.group(0).strip())
            return ""
        return m.group(0)

    def element(m: "re.Match") -> str:
        if m.group(1) in known:
            if log is not None:
                log.append(m.group(1))
            return f"«{_short(known[m.group(1)].get('text'), 80)}»"
        return m.group(0)

    def segment(seg: str) -> str:
        return _CODE_TOKEN_RE.sub(element, _CODE_PAREN_RE.sub(paren, _expand_ranges(seg, known)))
    return unknown_codes_out(_by_line(text, segment), known, log, original=text)


def _codes_out_unknown(text: str, log: Optional[List[str]] = None) -> str:
    """codes_out with no known code: a parenthesis of R-codes gone; a sentence using a bare R-code as an element
    (code_context_re) removed -- a line's bold head kept."""
    from .lawyer_method import _by_sentence

    def paren(m: "re.Match") -> str:
        ids = re.findall(r"[RCP]\d{1,3}", m.group(1))
        if ids and all(x.startswith("R") for x in ids):
            if log is not None:
                log.append(m.group(0).strip())
            return ""
        return m.group(0)
    out = _by_line(text, lambda seg: _CODE_PAREN_RE.sub(paren, _expand_ranges(seg, {})))
    ctx_re = code_context_re()
    if ctx_re is None:
        return out if out != text else text

    def review(sentence: str) -> str:
        bare = _R_CODE_RE.findall(_QUOTE_RE.sub("", sentence))
        if bare and ctx_re.search(_QUOTE_RE.sub("", sentence)):
            if log is not None:
                log.append(f"{', '.join(bare)}: {sentence.strip()[:120]}")
            head = _BOLD_LEAD_RE.match(sentence)
            return head.group(0).rstrip() if head else ""
        return sentence
    lines = []
    for line in out.split("\n"):
        if _table(line) or not _R_CODE_RE.search(_QUOTE_RE.sub("", line)):
            lines.append(line)
            continue
        indent = line[:len(line) - len(line.lstrip())]
        kept = _by_sentence(line, review)
        if kept.strip():
            lines.append(indent + kept.strip() if kept != line else line)
    joined = "\n".join(lines)
    return joined if joined != text else text


_BOLD_LEAD_RE = re.compile(r"^\s*(?:[-•]\s+)?\*\*[^*\n]{1,80}\*\*\s*")


def unknown_codes_out(text: str, known: Dict[str, Dict[str, Any]], log: Optional[List[str]] = None,
                      original: Optional[str] = None) -> str:
    """The sentences holding a bare R-code no record holds (the thesis call's invented «R9»), removed outside the
    tables and the quotes -- a line's bold head («**Risposta.**») kept; a parenthesis of such codes alone dropped.
    Nothing when known is empty (the method wrote no element)."""
    original = text if original is None else original
    if not text or not known:
        return text or ""
    from .lawyer_method import _by_sentence

    def paren(m: "re.Match") -> str:
        ids = re.findall(r"[RCP]\d{1,3}", m.group(1))
        if ids and all(x in known or x.startswith("R") for x in ids) and any(x not in known for x in ids):
            if log is not None:
                log.append(m.group(0).strip())
            return ""
        return m.group(0)
    text = _by_line(text, lambda seg: _CODE_PAREN_RE.sub(paren, _expand_ranges(seg, known)))

    def review(sentence: str) -> str:
        bare = [x for x in _R_CODE_RE.findall(_QUOTE_RE.sub("", sentence)) if x not in known]
        if bare:
            if log is not None:
                log.append(f"{', '.join(bare)}: {sentence.strip()[:120]}")
            head = _BOLD_LEAD_RE.match(sentence)
            return head.group(0).rstrip() if head else ""
        return sentence
    lines = []
    for line in text.split("\n"):
        if _table(line) or not _R_CODE_RE.search(_QUOTE_RE.sub("", line)):
            lines.append(line)
            continue
        indent = line[:len(line) - len(line.lstrip())]
        kept = _by_sentence(line, review)
        if kept.strip():
            lines.append(indent + kept.strip() if kept != line else line)
    joined = "\n".join(lines)
    return joined if joined != original else original


def codes_hits(text: str, known: Dict[str, Dict[str, Any]]) -> List[str]:
    found: List[str] = []
    codes_out(text, known, found)
    return found


# --- (d) a list's draft status ----------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def internal_re() -> Optional["re.Pattern"]:
    return _join(data().get("internal_notes") or [])


def internal_notes_out(text: str, log: Optional[List[str]] = None) -> str:
    rx = internal_re()
    if rx is None or not text or not rx.search(text):
        return text or ""
    from .lawyer_method import _by_sentence

    def review(sentence: str) -> str:
        if rx.search(sentence):
            if log is not None:
                log.append(sentence.strip()[:200])
            return ""
        return sentence
    lines = []
    for line in text.split("\n"):
        if _table(line) or not rx.search(line):
            lines.append(line)
            continue
        indent = line[:len(line) - len(line.lstrip())]
        kept = _by_sentence(line, review).strip()
        if kept:
            lines.append(indent + kept)
    return "\n".join(lines)


def internal_note_hits(text: str) -> List[str]:
    rx = internal_re()
    return [m.group(0) for m in rx.finditer(text or "")] if rx is not None else []


# --- (b) a date left without its ruling's head -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def date_pattern() -> str:
    """The table's date forms as one regex source («11.04.2025», «11 aprile 2025»)."""
    forms = [p for p in data().get("headless_dates") or [] if p]
    return "(?:" + "|".join(f"(?:{p})" for p in forms) + ")" if forms else r"(?:\d{1,2}[./-]\d{1,2}[./-]\d{4})"


@lru_cache(maxsize=1)
def _orphan_re() -> "re.Pattern":
    return re.compile(r"[ \t]*(?:[,;]\s*)?(?:(?:del|in\s+data|ud\.|udienza(?:\s+del)?|dep\.|depositat\w*(?:\s+il)?)\s*)?" + date_pattern() + r"\s*\)",
                      re.IGNORECASE)


def orphan_dates(text: str) -> List[Tuple[int, int]]:
    """The spans of a date before a closing parenthesis that closes nothing on its line («… 11.04.2025)» once its
    reference went) -- a date in its own parenthesis («(11.04.2025)»), or inside one opened before, is none."""
    spans = []
    at = 0
    for line in (text or "").split("\n"):
        if not _table(line):
            for m in _orphan_re().finditer(line):
                before = line[:m.start()]
                if before.count("(") - before.count(")") <= 0:
                    spans.append((at + m.start(), at + m.end()))
        at += len(line) + 1
    return spans


def _masked(text: str) -> str:
    """The text with each «…» quote's inside blanked (same length): a passage's own citations are no residue."""
    return _QUOTE_RE.sub(lambda m: "«" + "_" * (len(m.group(0)) - 2) + "»", text or "")


def citation_spans(text: str) -> List[Tuple[int, int]]:
    """The spans the citation check removes: every headless reference with its date (rulings.headless) and every date
    left orphaned before a closing parenthesis -- outside the quotes, merged, in order."""
    from . import rulings
    masked = _masked(text)
    merged: List[List[int]] = []
    for a, b in sorted(rulings.headless(masked) + orphan_dates(masked)):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [(a, b) for a, b in merged]


# --- (c) a procedural act narrated ------------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _acts() -> List[Dict[str, Any]]:
    out = []
    for a in data().get("procedural_acts") or []:
        rx = _join(a.get("patterns") or [])
        if rx is not None and a.get("act"):
            out.append({"act": str(a["act"]), "stems": [str(s).lower() for s in a.get("stems") or []], "re": rx})
    return out


@lru_cache(maxsize=1)
def _modals_re() -> Optional["re.Pattern"]:
    return _join(data().get("procedural_act_modals") or [])


def procedural_act(sentence: str) -> str:
    """The first act of the table the sentence narrates ("" when none) -- (review) none when the sentence holds a modal
    or a conditional («il PM potrebbe contestare», «sarebbe avvenuta»): a hypothetical act is no act done."""
    modals = _modals_re()
    if modals is not None and modals.search(sentence or ""):
        return ""
    for a in _acts():
        if a["re"].search(sentence or ""):
            return a["act"]
    return ""


def act_remainder(sentence: str, act: str) -> str:
    """The sentence without the act's own words and the hedges that describe the act's content («ipotizza»)."""
    out = sentence or ""
    for a in _acts():
        if a["act"] == act:
            out = a["re"].sub(" ", out)
    hedges = _join(data().get("procedural_act_hedges_ignored") or [])
    return " ".join((hedges.sub(" ", out) if hedges is not None else out).split())


def act_named(element: str, act: str) -> bool:
    """The element names the act (one of its stems begins one of its words): the act's sentence grounds it."""
    stems = next((a["stems"] for a in _acts() if a["act"] == act), [])
    words = re.findall(r"[a-zà-ÿ]+", str(element or "").lower())
    return any(w.startswith(s) for s in stems for w in words)


# --- (f) the cenni's rule statements ----------------------------------------------------------------------------------------

@lru_cache(maxsize=1)
def cenni_verbs_re() -> Optional["re.Pattern"]:
    return _join(data().get("cenni_rule_verbs") or [])


def cenni_rule(sentence: str) -> bool:
    from . import rule_statements
    rx = cenni_verbs_re()
    return rule_statements.is_rule(sentence) or bool(rx is not None and rx.search(sentence or ""))


def _commi(text: str) -> List[str]:
    """An article's text by comma: its lines, or its numbered paragraphs («1. …», «2. …»), else the whole text."""
    parts = [p for p in re.split(r"\n+|(?:(?<=[.;:])|^)\s*(?=\d{1,2}(?:-[a-z]+)?\.\s+[A-ZÀ-Ü])", text or "") if p and p.strip()]
    return parts or ([text] if text else [])


def _labels(sentence: str) -> List[str]:
    from . import rule_statements
    from .lawyer_method import _norm_label
    return [_norm_label(x) for x in rule_statements._labels(sentence)]


_NEG_WORDS = {"non", "mai", "né", "nessun", "nessuna", "nessuno"}


def _polarity(text: str) -> Dict[str, set]:
    """{stem: {negated?}} of the text's words of four letters or more, cut to five (the verbs included: «esclude»,
    «escludono» -> «esclu») -- negated when «non», «mai», «né», «nessun…» stands in the two words before it."""
    words = re.findall(r"[a-zà-öø-ÿ]+", str(text or "").lower())
    out: Dict[str, set] = {}
    for i, w in enumerate(words):
        if len(w) < 4 or w in _NEG_WORDS:
            continue
        out.setdefault(w[:5], set()).add(any(x in _NEG_WORDS for x in words[max(0, i - 2):i]))
    return out


@lru_cache(maxsize=1)
def cenni_absolutes_re() -> Optional["re.Pattern"]:
    return _join(data().get("cenni_absolutes") or [])


def cenni_distorts(sentence: str, comma: str) -> str:
    """(review) How a rule statement sharing its words with a comma still says another rule: «inverted» (a word the
    comma negates the sentence asserts, or the reverse: «il concorso di cause … esclude il rapporto di causalità» on
    «… non esclude …»), «absolute» (an absolute the comma does not hold: «escludono sempre» on «escludono … quando sono
    state da sole sufficienti»); "" when neither."""
    mine, theirs = _polarity(sentence), _polarity(comma)
    if any(mine[k].isdisjoint(theirs[k]) for k in mine.keys() & theirs.keys()):
        return "inverted"
    rx = cenni_absolutes_re()
    if rx is not None:
        said = {m.group(0).lower() for m in rx.finditer(sentence or "")}
        if said - {m.group(0).lower() for m in rx.finditer(comma or "")}:
            return "absolute"
    return ""


def cenni_grounded(sentence: str, labels: Sequence[str], texts: Dict[str, str], src: Dict[str, Any]) -> bool:
    """The rule statement shares enough words with one comma of a cited article's text, or traces to one of Block B's
    sources on that article: an element of its record, a store field, a registry passage (rule_statements)."""
    from . import rule_statements
    d = data()
    need, share = int(d.get("cenni_min_shared") or 2), float(d.get("cenni_min_share") or 0.5)
    words = rule_statements.stems(re.sub(r"\bart(?:icol[oi]|t?\.)\s*[\d\w-]+", " ", sentence or ""))
    if not words:
        return True
    for label in labels:
        # (review) the comma the sentence draws on most grounds it only when the sentence says its rule: not inverted,
        # not made absolute -- never another comma's verb on this one's subject
        commi = [(len(words & rule_statements.stems(c)), c) for c in _commi(texts.get(label) or "")]
        top = max((n for n, _ in commi), default=0)
        if top >= need and top >= share * len(words) and any(
                n == top and not cenni_distorts(sentence, c) for n, c in commi):
            return True
        if any(rule_statements._overlap(words, e) for e in (src.get("elements") or {}).get(label) or []):
            return True
        if any(rule_statements.STORE_FIELDS[f].search(sentence or "") for f in (src.get("store") or {}).get(label) or ()
               if f in rule_statements.STORE_FIELDS):
            return True
        if any(rule_statements._overlap(words, p) for p in (src.get("passages") or {}).get(label) or []):
            return True
    return False


def cenni_check(text: str, texts: Dict[str, str], src: Dict[str, Any], log: Optional[List[Dict[str, str]]] = None,
                rewrite: bool = True) -> str:
    """A «Cenni» section's text without its rule statements the cited article does not ground (each logged {sentence,
    reason}); rewrite False: the text as it is, the ungrounded ones logged (the count of those left)."""
    from .lawyer_method import _by_sentence, _norm_label
    if not text:
        return text or ""
    texts = {_norm_label(k): v for k, v in (texts or {}).items()}
    section = list(dict.fromkeys(_labels(text)))

    def review(sentence: str) -> str:
        s = sentence.strip()
        if s.startswith("|") or s.startswith("#") or not cenni_rule(s):
            return sentence
        labels = _labels(s) or section or sorted(texts)
        if not labels:
            from . import rule_statements
            if rule_statements.source_of(s, src):
                return sentence
        elif cenni_grounded(s, labels, texts, src):
            return sentence
        if log is not None:
            log.append({"sentence": s[:300], "reason": "regola nei cenni senza fonte nel testo o nel registro di "
                        + (", ".join(labels) if labels else "alcuna norma citata")})
        return "" if rewrite else sentence
    out = "\n".join(line if _table(line) else _by_sentence(line, review) for line in text.split("\n"))
    out = re.sub(r"\n{3,}", "\n\n", out)
    return out if rewrite and out != text else text


# --- (e) the main line once per section -------------------------------------------------------------------------------------

def _line_key(text: str) -> str:
    t = _CODE_PAREN_RE.sub("", str(text or ""))
    t = re.sub(r"[«»*_]", "", t).lower()
    return " ".join(t.split()).strip(" .")


def main_line_count(section: str, tesi: str) -> int:
    """How many times the section prints the main line's text (case, codes, quotes and emphasis aside; tables in)."""
    key = _line_key(tesi)
    return _line_key(section).count(key) if len(key) >= 12 else 0


def main_line_repeated(sections: Sequence[Tuple[str, str]], tesi: str) -> List[str]:
    """The titles of the sections printing the main line more than once."""
    return [t for t, x in sections if main_line_count(x, tesi) > 1]


# --- the last pass over the delivered answer --------------------------------------------------------------------------------

def citations_out(text: str, log: Optional[List[str]] = None) -> str:
    """lawyer_method.whole_citations (the quotes aside: a ruling's passage may cite another headless)."""
    from .lawyer_method import whole_citations
    return whole_citations(text, log)


def residue_out(text: str, known: Dict[str, Dict[str, Any]], log: Optional[Dict[str, List[str]]] = None) -> str:
    """The delivered answer without the residue: element codes, stances, the internal notes, the headless citations
    and dates -- each step's removals in log[step]."""
    log = log if log is not None else {}
    out = codes_out(text, known, log.setdefault("codes", []))
    out = stance_out(out, log.setdefault("stance", []))
    out = internal_notes_out(out, log.setdefault("internal_notes", []))
    return citations_out(out, log.setdefault("headless", []))


def residue_left(text: str, known: Dict[str, Dict[str, Any]]) -> Dict[str, int]:
    """What a text still shows of each kind (the counters «…_left»: 0)."""
    head = len(citation_spans(text))
    return {"codes": len(codes_hits(text, known)), "stance": len(stance_hits(text)),
            "internal_notes": len(internal_note_hits(text)), "headless": head}
