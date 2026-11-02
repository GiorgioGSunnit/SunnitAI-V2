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
                         r"su|a|al|alla|e|o|ma|se|il|lo|la|i|gli|le|un|uno|una|sono)\s*[.;][»”\"]?\s*$", re.IGNORECASE)
# The generic text of an issue a step added: an instruction to the writer, never a sentence of the answer.
_TEMPLATE_RE = re.compile(r"che cosa prevedono\b.*\bcome incidono sulla posizione|^\W*Il fatto «[^»]*» pone\b|"
                          r"da esaminare una per una", re.IGNORECASE | re.DOTALL)
# "...dopo la notificazione., Leonardo può": a stop left before the comma when what stood between them went.
_STOP_COMMA_RE = re.compile(r"\b([a-zà-ù]{4,}|dal|del|dei|che|per|con|tra|fra)\.,(?=\s)")
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
    text = re.sub(r"([:;]),(?=\s)", r"\1", _COMMA_NUMBER_STOP_RE.sub(r"\1", text or ""))      # "Mediazione obbligatoria:,"
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


_MIN_LEFT_WORDS = 7
_ARTICLE_CITED_RE = re.compile(r"\bartt?\.\s*\d|\barticol[oi]\s+\d", re.IGNORECASE)
# Where a citation stood and went: ".,", or a subject lost ("Gli artt. 700 e 614-bis c.p.c. permettono ..." -> "Permettono ...").
_CITATION_SUBJECT_RE = re.compile(r"^\W*(?:(?:l['’]\s*|gli\s+|il\s+)?(?:artt?\.|articol[oi]\b))", re.IGNORECASE)


def comma_words_left(sentence: str, parts: List[str]) -> bool:
    """Is this sentence carrying an article's words in place of its citation, no citation left? Seven words or more of a
    comma (or of a section title cut off), up to a comma of the sentence and a clause that goes on: "Qualunque fatto doloso
    o colposo che cagiona ad altri un danno ingiusto, è fondata". A quotation is a quotation, and stays."""
    if _ARTICLE_CITED_RE.search(sentence) or re.search("[«“\"]", sentence):
        return False
    bodies = [f" {' '.join(_opening_words(p))} " for p in parts]
    for m in re.finditer(r"\.?,\s+\S", sentence):
        head, after = _words(sentence[:m.start()]), _words(sentence[m.end() - 1:])[:1]
        run = " ".join(head[-_MIN_LEFT_WORDS:])
        if len(head) >= _MIN_LEFT_WORDS and any(f" {run} " in body and f" {run} {' '.join(after)} " not in body for body in bodies):
            return True                       # the words stop where the comma goes on: a clause of the writer's follows
    return False


def breaks_without_citation(old: str, new: str) -> bool:
    """Taking the citation out left no sentence: a stop before a comma, or the citation was its subject."""
    return bool(re.search(r"[a-zà-ù]{2,}\.,\s", new)) or bool(re.search(r"\b(?:co\.|comma)\s*\d+\.,\s", old)) or (bool(_CITATION_SUBJECT_RE.match(old)) and not _ARTICLE_CITED_RE.search(new))


# v42: given whole commi, the writer pastes them. Twelve words in a row from an article, outside «...», are a copy.
COPY_WORDS = 12
_MAX_COPIES = 40
_COPY_LEAD_WORDS = 2       # what a whole sentence may hold besides the copy ("Infatti, ...")


def _copy_index(rows: List[Dict[str, Any]]) -> Dict[Tuple[str, ...], Tuple[str, str]]:
    """{twelve words of an article: the article}."""
    out: Dict[Tuple[str, ...], Tuple[str, str]] = {}
    for row in rows or []:
        words = _words((row.get("s") or {}).get("plain_text"))
        if row.get("_article"):
            for i in range(len(words) - COPY_WORDS + 1):
                out.setdefault(tuple(words[i:i + COPY_WORDS]), tuple(row["_article"]))
    return out


def _in_quotes(text: str, at: int) -> bool:
    return text.count("«", 0, at) > text.count("»", 0, at) or text.count("“", 0, at) > text.count("”", 0, at)


def _first_copy(text: str, index: Dict[Tuple[str, ...], Tuple[str, str]]) -> Any:
    """(start, end, article) of the first run of copied words outside quotation marks, or None."""
    found = list(re.finditer(r"\w+", text))
    words = [m.group(0).lower() for m in found]
    i = 0
    while i + COPY_WORDS <= len(words):
        ref = index.get(tuple(words[i:i + COPY_WORDS]))
        if not ref or _in_quotes(text, found[i].start()):
            i += 1
            continue
        j = i + COPY_WORDS
        while j < len(words) and tuple(words[j - COPY_WORDS + 1:j + 1]) in index:
            j += 1
        return found[i].start(), found[j - 1].end(), ref
    return None


def _without_copy(text: str, start: int, end: int, ref: Tuple[str, str]) -> Tuple[str, str]:
    """(text, what was done) with one copy dealt with: taken out of its brackets; a whole sentence, or one ending in it after
    the citation of that same article, put in «...»; else its sentence dropped."""
    from . import plan_then_write as ptw
    opening, closing = text.rfind("(", 0, start), text.find(")", end)
    if opening > text.rfind(")", 0, start) and closing >= 0 and "\n" not in text[opening:closing]:
        cut = end + len(re.match(r"[\s.;,:]*", text[end:]).group(0))
        return (text[:start].rstrip() if cut == closing else text[:start]) + text[cut:], "copy out of its brackets"
    a, b = next(((x, y) for x, y in ptw._sentence_spans(text) if x <= start < y), (start, end))
    b = max(b, end + len(re.match(r"[\s.;,:)]*?(?=\s|$)", text[end:]).group(0)))
    lead, tail = text[a:start], text[end:b]
    quote = f"«{text[start:end]}»"
    cites = any(ptw._same_ref(r, ref) for _, _, refs in ptw._citations(lead, [], {ref}) for r in refs)
    if len(_words(lead)) + len(_words(tail)) <= _COPY_LEAD_WORDS:
        cited = re.match(r"\s*\(art", text[b:]) is not None
        new = quote if cited else f"{quote} (art. {ref[0]} {ref[1]})."
        return text[:start] + new + text[b:], "a whole comma quoted"
    if not _words(tail) and cites:
        return text[:start] + quote + "." + text[b:], "a copy after its citation quoted"
    return ptw._drop_spans(text, [(a, b)]), "a copy inside a sentence: sentence dropped"


def without_copies(text: str, rows: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): no run of COPY_WORDS words of an article left outside «...» (_without_copy)."""
    index = _copy_index(rows)
    log: List[Dict[str, Any]] = []
    for _ in range(_MAX_COPIES):
        copy = _first_copy(text, index) if index else None
        if not copy:
            break
        log.append({"kind": "", "article": f"{copy[2][0]} {copy[2][1]}", "sentence": text[copy[0]:copy[1]][:300]})
        text, log[-1]["kind"] = _without_copy(text, *copy)
    return empty_brackets(text) if log else text, log


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
    every_part += [t for t in (str((r.get("s") or {}).get("title") or "") for r in rows or []) if t]
    openings = [w for w in (_words(p) for p in every_part) if len(w) >= 8]
    drops = {}
    for s, e in ptw._sentence_spans(text):
        if comma_words_cited(text[s:e].strip(), every_part) or comma_words_left(text[s:e].strip(), every_part):
            drops[(s, e)] = None
            log.append({"kind": "comma words with a citation", "sentence": text[s:e].strip()[:300]})
            continue
        words = _words(_TRAILING_CITATION_RE.sub("", text[s:e]))
        if len(words) >= 5 and any(w[:len(words)] == words and len(words) < 0.8 * len(w) for w in openings):
            drops[(s, e)] = None
            log.append({"kind": "comma opening as a sentence", "sentence": text[s:e].strip()[:300]})
    text = grounded_review.apply(text, drops) if drops else text
    text, copies = without_copies(text, rows)
    return text, log + copies


# The norms section (§3): each paragraph opens with its article. PO feedback on Leonardo: when a check drops a paragraph's
# first sentence the next is left headless ("esclude che le servitù non apparenti...", "Ai sensi del comma 1, ...").
_NORMS_SECTION_RE = re.compile(r"(?ms)^(\*\*3\.\s+Norme e giurisprudenza\*\*[ \t]*\n)(.*?)(?=^\*\*\d+\.\s|\Z)")
_ARTICLE_NAMED_RE = re.compile(r"\bartt?\.\s*\d|\barticol[oi]\s+\d", re.IGNORECASE)
_RULING_NAMED_RE = re.compile(r"\bcass\b|cassazione|\bsez\.|sentenz|ordinanz|giurisprudenz|pronunc|\b\d{1,6}/\d{4}\b", re.IGNORECASE)
_COMMA_LEAD_RE = re.compile(r"^((?:ai sensi del|al|il|nel)\s+(?:comma\s+\S+?|(?:primo|secondo|terzo|quarto)\s+comma))(,?)\s+", re.IGNORECASE)
_BARE_VERB_RE = re.compile(r"^(?:esclude|prevede|stabilisce|dispone|richiede|disciplina|consente|riconosce|attribuisce|impone|"
                           r"regola|sancisce|precisa|aggiunge|vieta|presume)\b", re.IGNORECASE)
_MIN_SHARED_STEMS = 3
_MIN_SHARE = 0.3


def _paragraph_article(paragraph: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The row of the article a paragraph is about: the first it cites, else the plan's article sharing most words with
    it, the best alone; {} when none does."""
    from . import plan_then_write as ptw
    with_ref = [r for r in rows if r.get("_article") and (r.get("s") or {}).get("plain_text")]
    known = {r["_article"] for r in with_ref}
    for _, _, refs in ptw._citations(paragraph, [], known):
        found = next((r for r in with_ref if ptw._same_ref(r["_article"], refs[0])), None)
        if found:
            return found
    own = _stems(paragraph)
    scored = sorted(((len(own & _stems(r["s"]["plain_text"])), id(r)) for r in [r for r in with_ref if r.get("_in_plan")] or with_ref),
                    reverse=True)
    top = scored[0][0] if scored else 0
    if top < _MIN_SHARED_STEMS or top < _MIN_SHARE * len(own) or sum(1 for n, _ in scored if n == top) > 1:
        return {}
    return next(r for r in with_ref if id(r) == scored[0][1])


def _with_article(paragraph: str, row: Dict[str, Any]) -> str:
    from . import article_text, plan_then_write as ptw
    number, code = row["_article"]
    rubrica = article_text.heading(row.get("s"))
    label = f"art. {number} {ptw._code_key(code)}" + (f" ({rubrica})" if rubrica else "")
    lead = _COMMA_LEAD_RE.match(paragraph)
    if lead:
        return f"{lead.group(1)} dell'{label}{lead.group(2)} {paragraph[lead.end():]}"
    if _BARE_VERB_RE.match(paragraph) or paragraph[:1].islower():
        return f"L'{label} {paragraph[0].lower()}{paragraph[1:]}"
    return f"L'{label}: {paragraph}"


def headed_norms(text: str, rows: List[Dict[str, Any]]) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): every paragraph of the norms section opening with its article's reference (_with_article); a paragraph
    of a ruling, a heading or a list item is left alone; one that matches no article joins the paragraph before it."""
    from . import plan_then_write as ptw
    match = _NORMS_SECTION_RE.search(text or "")
    if not match:
        return text, []
    log: List[Dict[str, Any]] = []
    out: List[str] = []
    for paragraph in re.split(r"\n\s*\n", match.group(2).strip()):
        spans = ptw._sentence_spans(paragraph)
        first = paragraph[spans[0][0]:spans[0][1]] if spans else paragraph
        heading = ptw._HEADING_LINE_RE.match(paragraph)
        if not spans or (heading and not heading.group(2).strip()) or _LIST_RE.match(paragraph).group(0) \
                or _ARTICLE_NAMED_RE.search(first) or _RULING_NAMED_RE.search(first) or (out and _is_article_heading(out[-1])):
            out.append(paragraph)
            continue
        row = _paragraph_article(paragraph, rows)
        if row:
            out.append(_with_article(paragraph, row))
            log.append({"kind": "paragraph given its article", "sentence": first[:300], "now": out[-1][:300]})
        elif out:
            out[-1] = f"{out[-1]} {paragraph}"
            log.append({"kind": "headless paragraph joined to the one before", "sentence": first[:300]})
        else:
            out.append(paragraph)
    if not log:
        return text, []
    body = "\n\n".join(out)
    return text[:match.start(2)] + "\n" + body + "\n\n" + text[match.end(2):].lstrip("\n"), log


def _is_article_heading(paragraph: str) -> bool:
    from . import plan_then_write as ptw
    heading = ptw._HEADING_LINE_RE.match(paragraph)
    return bool(heading and not heading.group(2).strip() and _ARTICLE_NAMED_RE.search(paragraph))


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
