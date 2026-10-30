"""The text defects the edits after writing leave behind, repaired on every strategy answer (no flag: it only repairs).

Seen in real answers after anchored writing and the review edits took anchors or citations out:
  - empty brackets where an anchor or a citation was ("...all'uso a cui è destinata ()", "( )", "[]");
  - a sentence starting with a fragment of a citation (", comma 1, la cosa ..." once "L'art. 1490 c.c." went);
  - an unverified note in the middle of a sentence ("Il venditore [DA VERIFICARE: ...] risponde ..."); a bare
    "[DA VERIFICARE]" right after a citation marks that citation and stays where it is.
tidy() takes the empty brackets out, cuts a leading fragment (punctuation, "comma N", "del Codice Civile", a bare
code) and capitalises what remains -- the sentence dropped when nothing readable is left (fewer than three content
words) -- and moves a mid-sentence unverified note to the end of its sentence as "(da verificare)": the warning is
never deleted. Logged by the caller (defensive_generation: answer_tidy).
"""

import re
from typing import Any, Dict, List, Tuple

EMPTY_BRACKETS_RE = re.compile(r"[ \t]*(?:\(\s*[,;:.]?\s*\)|\[\s*[,;:.]?\s*\])")
_NOTE_RE = re.compile(r"[ \t]*(?:\[(?:TESTO\s+)?DA\s+VERIFICARE(?::[^\]\n]*)?\]|\(da\s+verificare(?::[^)\n]*)?\))",
                      re.IGNORECASE)
_SHORT_NOTE = "(da verificare)"
_CITATION_END_RE = re.compile(r"(?:\bc\.\s?[a-z](?:\.\s?[a-z])*\.|\bcost\.|\b\d{1,6}/\d{4}(?:\s+R\.G\.)?|"
                              r"\b(?:Codice|codice)\s+[A-Za-zà-ù ]{3,40}|\bartt?\.\s*\d+(?:-\w+)?)\s*$")
_LIST_RE = re.compile(r"^(?:[-*•]\s+|\d+[.)]\s+)?")
_CODE = (r"(?:c\.\s?p\.\s?c\.|c\.\s?p\.\s?p\.|c\.\s?p\.\s?a\.|c\.\s?d\.\s?s\.|c\.\s?c\.|c\.\s?p\.|cost\.|"
         r"(?:codice|cod\.)\s+(?:civile|penale|di\s+procedura\s+(?:civile|penale)|del\s+consumo|della\s+strada)\b)")
_FRAGMENT_RE = re.compile(
    r"^(?:\s*(?:[,;:.)\]]+|(?:comma|co\.)\s*\d+(?:-\w+)?(?:\s*(?:e|,)\s*\d+)*|(?:del|della|dello|dei|ex)\s+" + _CODE
    + r"|" + _CODE + r"))+\s*", re.IGNORECASE)
_FRAGMENT_START_RE = re.compile(r"^\s*(?:[,;:.)\]]|(?:comma|co\.)\s*\d)")     # "comma" lower-case: a cut citation


# a fragment right after a sentence's end: the sentence splitter reads "valido. comma 2 ..." as one sentence
_GLUED_RE = re.compile(r"(?<=[.!?])([ \t]+)((?:[,;:)\]]|(?:comma|co\.)\s*\d)[^\n]*?(?:[.!?](?=\s+[A-ZÀ-Ü*\-]|\s*$)|$))",
                       re.MULTILINE)


def _stems(text: str) -> set:
    return {w[:6] for w in re.findall(r"[a-zà-ù]{4,}", (text or "").lower())}


def capitalise(text: str) -> str:
    """The first letter of text in capitals (after a list marker or opening quotes)."""
    m = re.search(r"[a-zà-ùA-ZÀ-Ü]", text or "")
    return text if not m else text[:m.start()] + text[m.start()].upper() + text[m.start() + 1:]


def empty_brackets(text: str) -> str:
    """text without "()", "( )", "[]" (and a punctuation-only pair, "(, )"), the space before them too."""
    return EMPTY_BRACKETS_RE.sub("", text or "")


def lead_fragment(sentence: str) -> str:
    """The sentence without a leading fragment of a cut citation, capitalised; "" when fewer than three content
    words remain."""
    prefix = _LIST_RE.match(sentence).group(0)
    body = sentence[len(prefix):]
    if not _FRAGMENT_START_RE.match(body):
        return sentence
    rest = _FRAGMENT_RE.sub("", body, count=1)
    if len(_stems(rest)) < 3:
        return ""
    return prefix + capitalise(rest)


def move_notes(sentence: str) -> str:
    """An unverified note inside the sentence moved to its end as "(da verificare)" (once, before the final
    punctuation); a note already at the end stays as it is."""
    tail = sentence.rstrip()
    end_mark = re.search(r"[.!?;:]*$", tail)
    core, punct = tail[:end_mark.start()], end_mark.group(0)

    def at_end(m):    # the sentence's end, or the previous one's the splitter did not see ("c.c. [..] Né ...")
        return m.end() >= len(core.rstrip()) or not sentence[:m.start()].strip() or (
            sentence[:m.start()].rstrip().endswith(".")
                                                 and re.match(r"\s+[A-ZÀ-Ü]", sentence[m.end():]))
    def tags_citation(m):    # a bare "[DA VERIFICARE]" right after a citation marks that citation: it stays
        return ":" not in m.group(0) and _CITATION_END_RE.search(sentence[:m.start()])
    mid = [m for m in _NOTE_RE.finditer(core) if not at_end(m) and not tags_citation(m)]
    if not mid:
        return sentence
    for m in reversed(mid):
        core = core[:m.start()] + core[m.end():]
    core = re.sub(r"[ \t]{2,}", " ", core).rstrip()
    core = re.sub(r"\s+([,;:.])", r"\1", core)
    return f"{core} {_SHORT_NOTE}{punct}" + sentence[len(tail):]


def tidy(text: str) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): every sentence's mid-sentence notes moved, empty brackets taken out, a leading fragment cut or the
    sentence dropped (see the module docstring)."""
    from . import plan_then_write as ptw
    log: List[Dict[str, Any]] = []

    def glued(m):
        fixed = lead_fragment(m.group(2))
        log.append({"kind": "fragment dropped" if not fixed else "fragment repaired", "sentence": m.group(2)[:300],
                    **({"now": fixed[:300]} if fixed else {})})
        return m.group(1) + fixed if fixed else ""
    text = _GLUED_RE.sub(glued, text or "")
    changes = []
    for s, e in ptw._sentence_spans(text or ""):
        old = text[s:e]
        new = move_notes(old)
        if new != old:
            log.append({"kind": "note moved", "sentence": old[:300], "now": new[:300]})
        cleaned = empty_brackets(new)
        if cleaned != new:
            log.append({"kind": "empty brackets", "sentence": new[:300], "now": cleaned[:300]})
        fixed = lead_fragment(cleaned)
        if fixed != cleaned:
            log.append({"kind": "fragment dropped" if not fixed else "fragment repaired", "sentence": cleaned[:300],
                        **({"now": fixed[:300]} if fixed else {})})
        if fixed != old:
            changes.append((s, e, fixed))
    if changes:
        from . import grounded_review
        text = grounded_review.apply(text, {(s, e): new or None for s, e, new in changes})
    rest = empty_brackets(text)          # brackets left outside any sentence (a heading, a list line)
    if rest != text:
        log.append({"kind": "empty brackets", "sentence": "(outside a sentence)"})
    return rest, log
