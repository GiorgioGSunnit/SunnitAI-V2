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

EMPTY_BRACKETS_RE = re.compile(r"[ \t]*(?:\([\s,;:.]*\)|\[[\s,;:.]*\])")
# A sentence ending where its clause was cut: "... ha luogo quando consta mediante."
_CUT_END_RE = re.compile(r"\b(?:mediante|quando|qualora|che|di|del|della|dello|dei|delle|degli|con|per|tra|fra|da|dal|dalla|in|"
                         r"su|a|al|alla|e|o|ma|se|il|lo|la|i|gli|le|un|uno|una)\s*[.;][»”\"]?\s*$", re.IGNORECASE)
# The generic text of an issue a step added: an instruction to the writer, never a sentence of the answer.
_TEMPLATE_RE = re.compile(r"che cosa prevedono\b.*\bcome incidono sulla posizione|^\W*Il fatto «[^»]*» pone\b|"
                          r"da esaminare una per una", re.IGNORECASE | re.DOTALL)
# "...dopo la notificazione., Leonardo può": a stop left before the comma when what stood between them went.
_STOP_COMMA_RE = re.compile(r"\b([a-zà-ù]{4,})\.,(?=\s)")
# "...art. 2043 c.c., co. 1., è giustificato": a comma's number closed by a stop in the middle of a sentence.
_COMMA_NUMBER_STOP_RE = re.compile(r"(\b(?:co\.|comma|commi)\s*\d+(?:-[a-z]+)?)\.(?=,|\s+(?-i:[a-zà-ù]))", re.IGNORECASE)
_ABBREVIATIONS = {"lett", "cost", "sent", "proc", "cass", "ordin", "sez", "pagg"}
_DOUBLE_STOP_RE = re.compile(r"(?<![.…])\.\.(?![.…])")
_TRAILING_CITATION_RE = re.compile(r"\s*\([^()]*\)(?=[.;]*\s*$)")
_SELF_CONDITION_RE = re.compile(r",\s*(?:se|qualora|ove|nel caso in cui|sempre che)\s+([^,;:.]+?)(?=\s*(?:[,;:.]|$))", re.IGNORECASE)
_CONDITION_FILLERS = {"effettivamente", "davvero", "realmente", "effettivo", "effettiva", "già", "ormai", "ancora"}
_SAME_STATE_RE = re.compile(r"^(?:scadut|spirat|trascors|decors)", re.IGNORECASE)
# ". " then a lower-case word, after a whole word: a cut clause pasted into a sentence ("... quando consta mediante. o
# all'art. 1064 ...", "... innovazioni che rendano. vieta al ...").
_BROKEN_JOIN_RE = re.compile(r"\b([a-zà-ù]{4,})\.\s+(?=[a-zà-ù])")
_SHORT_FORMS = _ABBREVIATIONS | {"artt", "comm", "ibid", "disp", "prel", "attuaz", "cons", "trib", "gazz", "ecc"}
_MIN_REPEAT_WORDS = 8
_REPEAT_SHARE = 0.9
_COMMA_CITED_RE = re.compile(r"(\b(?:art\.|articolo)\s*(\d+(?:-[a-z]+)?)\s+(c\.\s?p\.\s?c\.|c\.\s?p\.\s?p\.|c\.\s?c\.|c\.\s?p\.))"
                             r"\s*,?\s*(?:comma|co\.)\s*(\d+)(?:-[a-z]+)?", re.IGNORECASE)
_COMMA_OF_RE = re.compile(r"(?:(?:\bdel|\bal)\s+)?\bcomma\s+(\d+)(?:-[a-z]+)?\s+(dell['’]art\.|all['’]art\.)\s*(\d+(?:-[a-z]+)?)\s+"
                          r"(c\.\s?p\.\s?c\.|c\.\s?p\.\s?p\.|c\.\s?c\.|c\.\s?p\.)", re.IGNORECASE)
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


def stop_comma(text: str) -> str:
    """"notificazione., Leonardo" -> "notificazione, Leonardo"; an abbreviation's ("lett., ") stays."""
    text = _COMMA_NUMBER_STOP_RE.sub(r"\1", text or "")
    return _STOP_COMMA_RE.sub(lambda m: m.group(0) if m.group(1).lower() in _ABBREVIATIONS else f"{m.group(1)},", text)


def unreadable(sentence: str) -> str:
    """Why the sentence must go, "" when it may stay: the template of an added issue, a clause cut at its end, no subject."""
    from . import plan_then_write as ptw
    if _TEMPLATE_RE.search(sentence):
        return "issue template"
    if any(m.group(1) not in _SHORT_FORMS for m in _BROKEN_JOIN_RE.finditer(sentence)):
        return "cut clause inside a sentence"
    if _CUT_END_RE.search(_TRAILING_CITATION_RE.sub("", sentence.rstrip())):
        return "cut clause"
    if ptw._dangling(sentence.strip(), True):
        return "no subject"
    return ""


def _condition_words(text: str) -> List[str]:
    """A clause's words to compare a condition with: no adverb, "scaduto", "spirato", "trascorso" read as "decorso"."""
    return ["decorso" if _SAME_STATE_RE.match(w) else w for w in _words(text) if w not in _CONDITION_FILLERS]


def self_condition(sentence: str) -> str:
    """"il termine è decorso, se il termine è effettivamente scaduto." -> "il termine è decorso.": a condition repeating
    the clause before it goes, what follows it stays."""
    for m in _SELF_CONDITION_RE.finditer(sentence):
        before, tail = _condition_words(sentence[:m.start()]), _condition_words(m.group(1))
        if len(tail) >= 3 and before[-len(tail):] == tail:
            return sentence[:m.start()] + sentence[m.end():]
    return sentence


def _words(text: str) -> List[str]:
    return re.findall(r"\w+", (text or "").lower())


def _repeats(text: str) -> List[Tuple[int, int]]:
    """The spans of sentences saying again, word for word or nearly, what an earlier one said, with the same numbers."""
    from . import plan_then_write as ptw
    seen: List[Tuple[set, set]] = []
    out = []
    for s, e in ptw._sentence_spans(text):
        words = _words(text[s:e])
        if len(words) < _MIN_REPEAT_WORDS:
            continue
        mine, numbers = set(words), set(re.findall(r"\d+", text[s:e]))
        if any(numbers == theirs and len(mine & other) >= _REPEAT_SHARE * len(mine | other) for other, theirs in seen):
            out.append((s, e))
        else:
            seen.append((mine, numbers))
    return out


# The written shape of a comma copied as a sentence: its opening words, then ", ai sensi dell'art. 2043 c.c., co. 1".
# Nothing in the code builds it: the model copies a passage labelled "art. 2043 c.c., co. 1" (anchored_writing).
_CITED_AFTER_RE = re.compile(r",?\s*\b(?:ai sensi|a norma|in base|in forza|per effetto)\s+(?:dell['’]\s*|al\s+|all['’]\s*)art\.\s*\d+(?:-[a-z]+)?"
                             r"(?:\s+" + _CODE + r")?(?:\s*,?\s*(?:comma|co\.)\s*\d+(?:-[a-z]+)?)?\.?(?=[,\s]|$),?", re.IGNORECASE)
_MIN_COMMA_WORDS = 5
_MIN_SHARED_WORDS = 7       # "...ha luogo quando consta mediante opera visibile": the model ends it in words of its own


def _opening_words(part: str) -> List[str]:
    words = _words(part)
    return words[1:] if words[:1] and words[0].isdigit() else words


def comma_words_cited(sentence: str, parts: List[str]) -> bool:
    """Is this sentence a comma's opening words with its citation put inside or after them? Any of the commi: the number
    written may be another's. Read by its words, as the model's punctuation differs from the comma's."""
    m = _CITED_AFTER_RE.search(sentence)
    if not m:
        return False
    lead = _words(sentence[:m.start()])
    whole = lead + _words(sentence[m.end():].rstrip(" .;"))
    for opening in map(_opening_words, parts):
        for words in (lead, whole):
            shared = next((k for k, (a, b) in enumerate(zip(words, opening)) if a != b), min(len(words), len(opening)))
            if shared >= _MIN_COMMA_WORDS and (shared == len(words) or shared >= _MIN_SHARED_WORDS):
                return True
    return False


def _parts_by_article(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, str], Tuple[List[str], str]]:
    from . import comma_material
    out: Dict[Tuple[str, str], Tuple[List[str], str]] = {}
    for row in rows or []:
        ref, text = row.get("_article"), (row.get("s") or {}).get("plain_text")
        if ref and text and tuple(ref) not in out:
            out[tuple(ref)] = comma_material.article_parts(text)
    return out


def _last_comma(parts: List[str], how: str) -> int:
    from . import comma_material
    numbers = [n for n in comma_material.comma_numbers(parts, how) if n]
    return max(numbers) if numbers else len(parts)


def against_articles(text: str, rows: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log), against the articles' own text. A comma number past the article's last comma goes ("art. 1051 c.c.,
    comma 10" -> "art. 1051 c.c."). A sentence that is only the start of a comma, a citation after it, goes: a clause
    with no main verb ("Qualunque fatto doloso o colposo che cagiona ad altri un danno ingiusto (art. 2043 c.c., co. 1)."),
    written by the model from a passage, its punctuation gone, so read by its words."""
    from . import grounded_review, plan_then_write as ptw
    parts = _parts_by_article(rows)
    log: List[Dict[str, Any]] = []

    def over(number: str, code: str, k: str) -> bool:
        found = parts.get((number.lower(), re.sub(r"\s", "", code).lower()))
        return bool(found) and int(k) > _last_comma(*found)

    def cited(m):
        if not over(m.group(2), m.group(3), m.group(4)):
            return m.group(0)
        log.append({"kind": "comma past the article's last", "sentence": m.group(0)})
        return m.group(1)

    def named(m):
        if not over(m.group(3), m.group(4), m.group(1)):
            return m.group(0)
        log.append({"kind": "comma past the article's last", "sentence": m.group(0)})
        return f"{m.group(2)} {m.group(3)} {m.group(4)}"
    text = _COMMA_OF_RE.sub(named, _COMMA_CITED_RE.sub(cited, text or ""))
    every_part = [p for found, _ in parts.values() for p in found]
    openings = [w for w in (_words(p) for p in every_part) if len(w) >= 8]
    drops = {}
    for s, e in ptw._sentence_spans(text):
        if comma_words_cited(text[s:e].strip(), every_part):
            drops[(s, e)] = None
            log.append({"kind": "comma words with a citation", "sentence": text[s:e].strip()[:300]})
            continue
        words = _words(_TRAILING_CITATION_RE.sub("", text[s:e]))
        if len(words) >= 5 and any(w[:len(words)] == words and len(words) < 0.8 * len(w) for w in openings):
            drops[(s, e)] = None
            log.append({"kind": "comma opening as a sentence", "sentence": text[s:e].strip()[:300]})
    return (grounded_review.apply(text, drops) if drops else text), log


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
    joined = _DOUBLE_STOP_RE.sub(".", stop_comma(text))
    if joined != text:
        log.append({"kind": "stop before a comma, or doubled", "sentence": "(a join)"})
        text = joined
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
        shorter = self_condition(fixed) if fixed else fixed
        if shorter != fixed:
            log.append({"kind": "condition repeating its clause", "sentence": fixed[:300], "now": shorter[:300]})
            fixed = shorter
        why = unreadable(fixed) if fixed else ""
        if why:
            log.append({"kind": why, "sentence": fixed[:300]})
            fixed = ""
        if fixed != old:
            changes.append((s, e, fixed))
    if changes:
        from . import grounded_review
        text = grounded_review.apply(text, {(s, e): new or None for s, e, new in changes})
    repeated = _repeats(text)
    if repeated:
        from . import grounded_review
        log += [{"kind": "repeated sentence", "sentence": text[s:e].strip()[:300]} for s, e in repeated]
        text = grounded_review.apply(text, {span: None for span in repeated})
    rest = empty_brackets(text)          # brackets left outside any sentence (a heading, a list line)
    if rest != text:
        log.append({"kind": "empty brackets", "sentence": "(outside a sentence)"})
    return rest, log
