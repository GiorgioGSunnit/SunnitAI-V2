"""Sentence-level repetition guard (v51, no flag): within a section, a sentence whose content words are already said by
an earlier sentence of the same section -- at least OVERLAP of its content stems in that one -- goes; the first stays.

Leonardo v50 (75 sentences, 3 repeats): "Il foro è corretto in quanto si trova nel luogo dove si trovano i beni immobili
interessati." after "La competenza territoriale spetta al foro del luogo dove si trovano i beni immobili interessati."
(0.86); the "Verifiche da effettuare" lines saying again the litisconsorzio's and the mediation's sentences (0.93, 0.82).
text_tidy._repeats (the whole answer, 0.9 of the union of words, the same numbers) caught none of them. A sentence
sharing its words with two earlier ones, none alone ("La chiusura del cancello impedisce l'accesso necessario e quindi
è vietata.", 0.67 at most), stays.

Content words: the stems (STEM letters) of the words of four letters or more, function words left out (STOP). A
sentence with fewer than MIN_STEMS of them is never judged. A sentence is a repeat only when it cites no article the
earlier one does not (a citation is never lost) and holds no number the earlier one does not (another date, another
term). Applied to each section after writing and after the rewrites (plan_then_write.write_sections, write_parere),
and to each section of the final answer (defensive_generation's tidy); logged as case_analysis_repetition_guard and in
the citation ledger.
"""

import re
from typing import Any, Dict, List, Tuple

OVERLAP = 0.8
MIN_STEMS = 4
STEM = 5
EVENT = "case_analysis_repetition_guard"
STOP = frozenset(
    "della delle dello degli dalla dalle dallo dagli nella nelle nello negli sulla sulle sullo sugli alla alle allo "
    "agli quale quali quando anche perché perche essere stato stata stati state sono questo questa questi queste quello "
    "quella quelli quelle ovvero oppure infatti dunque quindi pertanto tuttavia inoltre ancora sempre molto come dove "
    "mentre però poiché perciò cioè loro essa esso esse essi tale tali ogni ciascun ciascuna tutto tutti tutte altro "
    "altra altri altre solo già può possono deve devono viene vengono avere aveva hanno abbia sarà sarebbe essendo "
    "quanto modo caso parte ai sensi dell nell sull all dall".split())
_SECTION_RE = re.compile(r"(?m)^\*\*\d+\.\s+[^*\n]+\*\*[ \t]*$")


def stems(sentence: str) -> set:
    words = re.findall(r"[^\W\d_]+", (sentence or "").lower())
    return {w[:STEM] for w in words if len(w) >= 4 and w not in STOP}


def _numbers(sentence: str) -> set:
    from . import defensive_generation as dg
    bare = dg._CITED_ARTICLE_RE.sub(" ", sentence or "")       # an article's number is its citation's, not a number
    return set(re.findall(r"\d+", bare))


def _refs(sentence: str) -> set:
    from . import defensive_generation as dg
    return set(dg._refs_in(sentence or ""))


def drop_repeats(text: str, section: str = "") -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): the section without its repeated sentences, by the module's rule."""
    from . import plan_then_write as ptw
    seen: List[Tuple[set, set, set, str]] = []
    drops, log = [], []
    for s, e in ptw._sentence_spans(text or ""):
        sentence = text[s:e].strip()
        mine = stems(sentence)
        if len(mine) < MIN_STEMS:
            continue
        refs, numbers = _refs(sentence), _numbers(sentence)
        earlier = next(((other, said) for other, other_refs, other_numbers, said in seen
                        if len(mine & other) >= OVERLAP * len(mine) and refs <= other_refs and numbers <= other_numbers),
                       None)
        if earlier is None:
            seen.append((mine, refs, numbers, sentence))
            continue
        drops.append((s, e))
        log.append({"section": section, "how": "sentence dropped", "why": "repeated sentence",
                    "overlap": round(len(mine & earlier[0]) / len(mine), 2), "sentence": sentence[:300],
                    "earlier": earlier[1][:300]})
    return (ptw._drop_spans(text, drops) if drops else text), log


def guard_sections(sections: List[Tuple[str, str]], stage: str) -> Tuple[List[Tuple[str, str]], List[Dict[str, Any]]]:
    """drop_repeats on each (title, text); the log of all of them, logged once (EVENT) with the stage."""
    out, log = [], []
    for title, text in sections:
        new, entries = drop_repeats(text, title)
        out.append((title, new or text))
        log += entries
    _log(stage, log)
    return out, log


def guard_answer(answer: str, stage: str) -> Tuple[str, List[Dict[str, Any]]]:
    """drop_repeats on each "**N. Title**" section of a whole answer (on all of it when it has no such heading)."""
    heads = list(_SECTION_RE.finditer(answer or ""))
    if not heads:
        text, log = drop_repeats(answer or "")
        _log(stage, log)
        return text, log
    parts, log = [answer[:heads[0].start()]], []
    for k, head in enumerate(heads):
        end = heads[k + 1].start() if k + 1 < len(heads) else len(answer)
        body = answer[head.end():end]
        new, entries = drop_repeats(body, head.group(0).strip("* \t"))
        if entries:           # the body's own spacing around it kept
            new = body[:len(body) - len(body.lstrip())] + new.strip() + body[len(body.rstrip()):]
        parts.append(head.group(0) + (new if entries else body))
        log += entries
    _log(stage, log)
    return "".join(parts), log


def _log(stage: str, log: List[Dict[str, Any]]) -> None:
    if not log:
        return
    from . import citation_ledger
    from . import defensive_generation as dg
    dg.vlog(EVENT, {"stage": stage, "dropped": log, "count": len(log)})
    citation_ledger.record_entries(f"repetition_guard:{stage}", log)
