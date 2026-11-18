"""v60f: the answer's ledger -- every sentence the writer wrote that a later step removes, with the step that removed
it (the reason), so no deletion goes unrecorded.

While a ledger is active (active(), a context variable: the interactive answer step), each review step run through
plan_then_write._step, and each the flow runs through step() / draft_step(), is diffed sentence by sentence: a sentence
of the step's input missing from its output is "removed" by that step; one replaced by a sentence sharing most of its
words is "rewritten" (a marker added, a quotation inserted, a word changed); one now in another section is "moved".
The first step's input is the baseline: the writer's sections as written. unaccounted(final) lists the baseline's
sentences missing from the final text that no entry explains -- the invariant is that it is empty.

Each entry names the articles its sentence belongs to: those it cites (citations), else its §2 paragraph's «[N]» (the
writer's numbering) or the §2 heading above it.

v60g (e): a deferring ledger (defer=True: the answer with records) lets no step delete. A step's removals are put back
where they stood (restore) and recorded as "proposed" by that step; its rewrites and moves stand. The one check run
last (record_check) decides each sentence against the record and records its removals itself (remove: "removed", step
"record_check", with the reason). The writer's own rewrite of a cut section (WRITING_STEPS) is writing, not a check:
its removals stand, recorded with that reason.
"""

import contextlib
import contextvars
import re
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

_ACTIVE: "contextvars.ContextVar[Optional[Ledger]]" = contextvars.ContextVar("answer_ledger", default=None)
REWRITE_SHARE = 0.6
SENTENCE_CHARS = 300
CHECK_STEP = "record_check"
WRITING_STEPS = {"complete_sections": "il redattore ha riscritto la sezione troncata",
                 "cover_blocks": "il redattore ha riscritto la sezione per coprire un blocco",
                 # v60q: a rewrite, never a removal -- a short sentence whose status word changed shares too few stems
                 "status_words": "parole di stato del redattore sostituite: gli stati li scrive il codice"}
_MARKERS_RE = re.compile(r"\s*\((?:da verificare|non risulta dal caso|calcolato)[^)]*\)|\s*\[DA VERIFICARE:[^\]]*\]",
                         re.IGNORECASE)
_MARK_RE = re.compile(r"^\s*\[(\d{1,2})\]\s*")
_HEADING_RE = re.compile(r"^\*\*Art\.\s+(\S+)\s+(.+?)(?:\s+—\s+.*?)?\*\*\s*$")
_SECTION_RE = re.compile(r"(?m)^\*\*\d+\. ([^*\n]+)\*\*\s*$")


def _norm(sentence: str) -> str:
    text = _MARK_RE.sub("", sentence or "")
    text = _MARKERS_RE.sub("", text)
    return " ".join(re.sub(r"[*_«»\"“”]", "", text).lower().split()).strip(" .;:")


def _stems(text: str) -> set:
    return {w[:5] for w in re.findall(r"[a-zà-ü]{4,}", (text or "").lower())}


def tracked(text: str, numbering: Sequence[Tuple[str, str]] = (), documents: Sequence[Dict[str, Any]] = ()
            ) -> List[Dict[str, Any]]:
    """The section's sentences the ledger tracks -- [{start, end, text, norm, articles}] --, headings, list lines and
    sentences of fewer than three words aside; a «[N]» mark holds for its paragraph and the unmarked ones after it, a §2
    heading for its block."""
    from .plan_then_write import _sentence_spans
    from . import citations
    lines = text.split("\n")
    starts, context, pos = [], [], 0
    mark, heading = None, None
    for line in lines:
        starts.append(pos)
        pos += len(line) + 1
        h = _HEADING_RE.match(line.strip())
        if h:
            heading, mark = (h.group(1), citations.code_key(h.group(2))), None
        m = _MARK_RE.match(line)
        if m:
            mark = int(m.group(1))
        context.append((mark, heading))
    out = []
    for s, e in _sentence_spans(text):
        sentence = text[s:e]
        line_no = max(i for i, st in enumerate(starts) if st <= s)
        line = lines[line_no]
        if _HEADING_RE.match(line.strip()) or line.lstrip().startswith("- ") or len(_norm(sentence).split()) < 3:
            continue
        mark_n, head = context[line_no]
        arts = [(str(r[0]), citations.code_key(r[1])) for r in citations.refs_in(sentence, documents)]
        if not arts and mark_n and 0 < mark_n <= len(numbering):
            arts = [numbering[mark_n - 1]]
        if not arts and head:
            arts = [head]
        out.append({"start": s, "end": e, "text": " ".join(sentence.split())[:SENTENCE_CHARS], "norm": _norm(sentence),
                    "articles": [f"{a} {b}" for a, b in dict.fromkeys(arts)], "heading": f"{head[0]} {head[1]}"
                    if head else ""})
    return out


class Ledger:
    def __init__(self, numbering: Sequence[Tuple[str, str]] = (), documents: Sequence[Dict[str, Any]] = (),
                 defer: bool = False):
        from . import citations
        self.numbering = [(str(r[0]), citations.code_key(r[1])) for r in numbering]   # §2's «[N]» -> the N-th of these
        self._memo: Dict[str, List[Dict[str, Any]]] = {}
        self.documents = list(documents)
        self.defer = defer
        self.baseline: Optional[List[Tuple[str, str]]] = None
        self.entries: List[Dict[str, Any]] = []
        self.steps: List[str] = []

    # -- sentences ---------------------------------------------------------------------------------------------
    def _sentences(self, sections: Sequence[Tuple[str, str]]) -> List[Dict[str, Any]]:
        """[{section, text, norm, articles}] of the sections, headings and status lines aside."""
        out = []
        for title, text in sections or []:
            key = f"{title}\x00{text or ''}"
            if key not in self._memo:
                self._memo[key] = self._section_sentences(title, text or "")
            out += self._memo[key]
        return out

    def _section_sentences(self, title: str, text: str) -> List[Dict[str, Any]]:
        return [{"section": title, "text": x["text"], "norm": x["norm"], "articles": x["articles"]}
                for x in tracked(text, self.numbering, self.documents)]

    # -- one step ------------------------------------------------------------------------------------------------
    def record(self, step: str, before: Sequence[Tuple[str, str]], after: Sequence[Tuple[str, str]]
               ) -> List[Tuple[str, str]]:
        """The step's entries; its output -- deferring (and not the writer's own rewrite), with its removals put back
        (restore) and recorded as "proposed"."""
        if self.baseline is None:
            self.baseline = [(t, x) for t, x in before]
        self.steps.append(step)
        old, new = self._sentences(before), self._sentences(after)
        new_norms = {s["norm"]: s for s in new}
        old_norms = {s["norm"] for s in old}
        fresh = [s for s in new if s["norm"] not in old_norms]
        used: Set[int] = set()          # v60g: a fresh sentence is the rewrite of one old sentence at most
        defer = self.defer and step not in WRITING_STEPS
        gone: List[Dict[str, Any]] = []
        rewrites: Dict[str, str] = {}
        for s in old:
            if s["norm"] in new_norms:
                if new_norms[s["norm"]]["section"] != s["section"]:
                    self.entries.append({"step": step, "kind": "moved", "section": s["section"],
                                         "to": new_norms[s["norm"]]["section"], "sentence": s["text"],
                                         "articles": s["articles"]})
                continue
            mine = _stems(s["norm"])
            best, share = None, 0.0
            for k, f in enumerate(fresh):
                if k in used:
                    continue
                theirs = _stems(f["norm"])
                if mine and theirs:
                    x = len(mine & theirs) / max(1, min(len(mine), len(theirs)))
                    if x > share:
                        best, share = k, x
            if best is not None and share >= REWRITE_SHARE:
                used.add(best)
                rewrites[s["norm"]] = fresh[best]["norm"]
                self.entries.append({"step": step, "kind": "rewritten", "section": s["section"], "sentence": s["text"],
                                     "now": fresh[best]["text"], "articles": s["articles"]})
            elif defer:
                gone.append(s)
                self.entries.append({"step": step, "kind": "proposed", "section": s["section"], "sentence": s["text"],
                                     "articles": s["articles"]})
            else:
                self.entries.append({"step": step, "kind": "removed", "section": s["section"], "sentence": s["text"],
                                     "articles": s["articles"], "reason": WRITING_STEPS.get(step, step)})
        if not gone:
            return list(after)
        return restore(before, after, gone, rewrites)

    def remove(self, reason: str, section: str, sentence: str, articles: Sequence[str], rule: str = "",
               proposed_by: Sequence[str] = ()) -> None:
        """A removal of the one check (record_check), with its reason."""
        self.entries.append({"step": CHECK_STEP, "kind": "removed", "section": section,
                             "sentence": " ".join(sentence.split())[:SENTENCE_CHARS], "articles": list(articles),
                             "reason": reason, "rule": rule, "proposed_by": list(proposed_by)})

    def proposals(self) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e["kind"] == "proposed"]

    def proposed_by(self, norm: str) -> List[str]:
        return list(dict.fromkeys(e["step"] for e in self.entries if e["kind"] == "proposed"
                                  and _norm(e["sentence"]) == norm))

    # -- the invariant --------------------------------------------------------------------------------------------
    def unaccounted(self, final_sections: Sequence[Tuple[str, str]]) -> List[str]:
        """The baseline's sentences missing from the final text with no removal recorded along their rewrites."""
        if self.baseline is None:
            return []
        final = {s["norm"] for s in self._sentences(final_sections)}
        removed = {_norm(e["sentence"]) for e in self.entries if e["kind"] == "removed"}
        rewritten = {_norm(e["sentence"]): _norm(e["now"]) for e in self.entries if e["kind"] == "rewritten"}
        out = []
        for s in self._sentences(self.baseline):
            norm, seen = s["norm"], set()
            while norm not in final and norm not in removed and norm in rewritten and norm not in seen:
                seen.add(norm)
                norm = rewritten[norm]
            if norm not in final and norm not in removed:
                out.append(s["text"])
        return out

    def unreasoned(self) -> List[Dict[str, Any]]:
        """v60g: removals without a reason, or by a step that is neither the check nor the writer's own rewrite (with
        a deferring ledger: none may be)."""
        allowed = {CHECK_STEP} | set(WRITING_STEPS)
        return [e for e in self.entries if e["kind"] == "removed"
                and (not str(e.get("reason") or "").strip() or (self.defer and e["step"] not in allowed))]

    def removals(self, label: str) -> List[Dict[str, Any]]:
        return [e for e in self.entries if e["kind"] == "removed" and label in e["articles"]]

    def written(self, label: str) -> bool:
        """The baseline (the writer's text) holds a sentence of this article."""
        return any(label in s["articles"] for s in self._sentences(self.baseline or []))

    def as_log(self) -> Dict[str, Any]:
        counts: Dict[str, int] = {}
        proposed: Dict[str, int] = {}
        reasons: Dict[str, int] = {}
        for e in self.entries:
            if e["kind"] == "removed":
                counts[e["step"]] = counts.get(e["step"], 0) + 1
                key = e.get("rule") or e.get("reason") or e["step"]
                reasons[key] = reasons.get(key, 0) + 1
            elif e["kind"] == "proposed":
                proposed[e["step"]] = proposed.get(e["step"], 0) + 1
        return {"steps": self.steps, "defer": self.defer, "removed_by_step": counts, "proposed_by_step": proposed,
                "removed_by_reason": reasons, "entries": self.entries}


# --- v60g (e): a step's removals put back ------------------------------------------------------------------------------

_SEP_MARK_RE = re.compile(r"\[\d{1,2}\][ \t]*$")


def _separator(text: str, prev_end: Optional[int], start: int) -> str:
    """What stood between a sentence and the one before it, as a separator to put back: a paragraph break (with the
    paragraph's «[N]» mark), a line break, or a space -- never a heading or a status line between them."""
    if prev_end is None:
        return ""
    between = text[prev_end:start]
    mark = _SEP_MARK_RE.search(between)
    tail = mark.group(0) if mark else ""
    if "\n\n" in between or (mark and "\n" in between):
        return "\n\n" + tail
    if "\n" in between:
        return "\n" + tail
    return " "


def _put_back(before: str, after: str, norms: Set[str], rewrites: Dict[str, str]) -> str:
    """after with the sentences of before whose norm is in norms back where they stood, in before's order: before the
    next sentence of their paragraph still in after (or its rewrite); else after the nearest earlier sentence still in
    after, with what separated them (a paragraph break and its «[N]»); else at the start."""
    from .plan_then_write import _sentence_spans
    b_spans = _sentence_spans(before)
    a_spans = [(s, e, _norm(after[s:e])) for s, e in _sentence_spans(after)]
    b_norms = [_norm(before[s:e]) for s, e in b_spans]

    def in_after(k: int, start: int) -> Optional[int]:
        target = rewrites.get(b_norms[k], b_norms[k])
        return next((j for j in range(start, len(a_spans)) if a_spans[j][2] in (b_norms[k], target)), None)
    inserts: List[Tuple[int, int, str]] = []
    cursor, a_at, order = 0, 0, 0
    for i, (s, e) in enumerate(b_spans):
        sentence = before[s:e]
        if b_norms[i] not in norms:
            j = in_after(i, a_at)
            if j is not None:
                cursor, a_at = a_spans[j][1], j + 1
            continue
        nxt = None              # the next sentence of the same paragraph (no line break between) still in after
        for k in range(i + 1, len(b_spans)):
            if "\n" in before[b_spans[k - 1][1]:b_spans[k][0]]:
                break
            if b_norms[k] not in norms:
                nxt = in_after(k, a_at)
                break
        if nxt is not None:
            pos = a_spans[nxt][0]
            piece = sentence + " "
        elif cursor == 0:
            pos = 0
            following = b_spans[i + 1][0] if i + 1 < len(b_spans) else None
            piece = sentence + ((_separator(before, e, following) if following is not None else "") or " ")
        else:
            pos = cursor
            piece = (_separator(before, b_spans[i - 1][1] if i else None, s) or " ") + sentence
        inserts.append((pos, order, piece))
        order += 1
    for pos, _, piece in sorted(inserts, key=lambda x: (x[0], x[1]), reverse=True):
        after = after[:pos] + piece + after[pos:]
    return after


def restore(before: Sequence[Tuple[str, str]], after: Sequence[Tuple[str, str]], gone: Sequence[Dict[str, Any]],
            rewrites: Optional[Dict[str, str]] = None) -> List[Tuple[str, str]]:
    """after with the gone sentences (the step's removals) put back in their sections; a section the step dropped comes
    back with them, at its place."""
    out = [(t, x) for t, x in after]
    by_section: Dict[str, Set[str]] = {}
    for g in gone:
        by_section.setdefault(g["section"], set()).add(g["norm"] if "norm" in g else _norm(g["text"]))
    for k, (title, text) in enumerate(before):
        norms = by_section.get(title)
        if not norms:
            continue
        at = next((i for i, (t, _) in enumerate(out) if t == title), None)
        if at is not None:
            out[at] = (title, _put_back(text or "", out[at][1] or "", norms, rewrites or {}))
            continue
        kept = "\n\n".join(" ".join(text[s["start"]:s["end"]].split()) for s in tracked(text or "")
                           if s["norm"] in norms)
        following = {t for t, _ in before[k + 1:]}
        out.insert(next((i for i, (t, _) in enumerate(out) if t in following), len(out)), (title, kept))
    return out


def current() -> Optional[Ledger]:
    return _ACTIVE.get()


@contextlib.contextmanager
def active(ledger: Ledger):
    token = _ACTIVE.set(ledger)
    try:
        yield ledger
    finally:
        _ACTIVE.reset(token)


def note(step: str, before: Sequence[Tuple[str, str]], after: Sequence[Tuple[str, str]]):
    """The active ledger's record of one step; the step's output -- with a deferring ledger, its removals put back.
    Nothing with none (never fails the step: on an error, the output as the step gave it)."""
    ledger = _ACTIVE.get()
    if ledger is None or after is None:
        return after
    try:
        return ledger.record(step, before, after)
    except Exception:           # noqa: BLE001 -- the ledger never fails an answer
        return after


def sections_of(draft: str) -> List[Tuple[str, str]]:
    """A joined draft («**N. Title**» sections) as [(title, text)]; the text before the first title as ("", head)."""
    parts = _SECTION_RE.split(draft or "")
    out = [("", parts[0])] if parts and parts[0].strip() else []
    out += [(parts[i].strip(), parts[i + 1]) for i in range(1, len(parts) - 1, 2)]
    return out


def rejoin(draft: str, sections: Sequence[Tuple[str, str]]) -> str:
    """draft with each section's text replaced by the one given (sections_of's order and titles: the head "", then
    each «**N. Title**» section); a section given that draft lacks is appended under its title."""
    marks = list(_SECTION_RE.finditer(draft or ""))
    texts = {}
    for title, text in sections:
        texts.setdefault(title, []).append(text)
    out, pos = [], 0
    head = draft[:marks[0].start()] if marks else draft
    if texts.get(""):
        head = texts[""].pop(0)
    out.append(head)
    seen = set()
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(draft)
        title = m.group(1).strip()
        seen.add(title)
        body = texts[title].pop(0) if texts.get(title) else draft[m.end():end]
        out.append(draft[m.start():m.end()] + body)
    for title, bodies in texts.items():
        if title and title not in seen:
            for body in bodies:
                out.append(f"\n\n**{len(marks) + 1}. {title}**\n\n{body.strip()}")
    return "".join(out)


def draft_step(step: str, draft: str, run: Callable[[str], str]) -> str:
    """A step on the whole draft, recorded section by section; the draft as it was when it fails. With a deferring
    ledger, the step's removals put back."""
    try:
        out = run(draft)
    except Exception:           # noqa: BLE001 -- as plan_then_write._step: the step's input stands
        return draft
    out = out if isinstance(out, str) and out else draft
    after = sections_of(out)
    kept = note(step, sections_of(draft), after)
    if kept is not None and kept != after:
        out = rejoin(out, kept)
    return out


# --- v79.10.5 (E3): every sentence logged as removed is absent from the final text --------------------------------------

MIN_CHECKED = 25                # the nightly's ledger_removed_absent: a shorter sentence is too short to tell
_OPEN_MARKUP = "*_«\"“"
_CLOSE_MARKUP = "*_»\"”"
_BOUNDARY_LEFT_RE = re.compile(r"(?:^|[.!?|]|^(?:[-•*]|\d{1,2}[.)])|^\[\d{1,2}\])$")
_TABLE_ROW_RE = re.compile(r"^\s*\|")
_BARE_LINE_RE = re.compile(r"^\s*(?:[-•*]|\d{1,2}[.)]|\[\d{1,2}\])?\s*[.;:]?\s*$")


def _norm_map(text: str) -> Tuple[str, List[int]]:
    """text as _norm reads it (the «[N]» mark at its head and the status markers out, the markup out, lower case, the
    spaces collapsed), with each kept character's position in text."""
    skip: Set[int] = set()
    head = _MARK_RE.match(text or "")
    if head:
        skip.update(range(head.start(), head.end()))
    for m in _MARKERS_RE.finditer(text or ""):
        skip.update(range(m.start(), m.end()))
    chars: List[str] = []
    where: List[int] = []
    for i, c in enumerate(text or ""):
        if i in skip or c in "*_«»\"“”":
            continue
        if c.isspace():
            if chars and chars[-1] != " ":
                chars.append(" ")
                where.append(i)
            continue
        low = c.lower()
        chars.append(low[0] if low else c)
        where.append(i)
    return "".join(chars), where


def removed_present(answer: str, entries: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """v79.10.5 (E3): the ledger's «removed» entries whose sentence (MIN_CHECKED characters or more, as _norm reads it)
    is still in the answer -- the nightly's ledger_removed_absent and settle read it alike; must be none."""
    final = _norm(answer or "")
    return [e for e in entries or [] if e.get("kind") == "removed"
            and len(_norm(e.get("sentence") or "")) >= MIN_CHECKED and _norm(e.get("sentence") or "") in final]


def _whole(text: str, s: int, e: int) -> Optional[Tuple[int, int]]:
    """The span to strike for an occurrence text[s:e] of a logged sentence -- widened over its markup, its closing
    punctuation, a status marker after it and the spaces that follow -- when it is a whole sentence (a line's, a list
    item's, a table cell's, or one between two sentences); None when it is part of a longer one or of a heading."""
    line_start = text.rfind("\n", 0, s) + 1
    line_end = text.find("\n", e)
    line_end = len(text) if line_end < 0 else line_end
    line = text[line_start:line_end]
    from .plan_then_write import _HEADING_LINE_RE
    h = _HEADING_LINE_RE.match(line)
    if h and not h.group(2).strip(" .:;–-"):
        return None
    start = s
    while start > line_start and text[start - 1] in _OPEN_MARKUP:
        start -= 1
    left = text[line_start:start].rstrip()
    while left and left[-1] in _OPEN_MARKUP:
        left = left[:-1].rstrip()
    if not _BOUNDARY_LEFT_RE.search(left.strip()):
        return None
    end, closed = e, False
    while True:
        m = _MARKERS_RE.match(text, end, line_end)
        if m and m.end() > end:
            end = m.end()
            continue
        if end < line_end and text[end] in _CLOSE_MARKUP + ".;:!?":
            closed = closed or text[end] in ".;:!?"
            end += 1
            continue
        break
    rest = text[end:line_end].lstrip()
    if rest and not rest.startswith("|") and not (closed and (rest[0].isupper() or rest[0] in "«\"“[*")):
        return None
    while end < line_end and text[end] in " \t":
        end += 1
    return start, end


def _line(text: str, s: int, e: int) -> str:
    end = text.find("\n", e)
    return " ".join(text[text.rfind("\n", 0, s) + 1:len(text) if end < 0 else end].split())


def strike(text: str, sentence: str, allow: Optional[Callable[[str, int], bool]] = None,
           keep_first: bool = False) -> Tuple[str, int, List[str]]:
    """v79.10.5 (E3): the removal and its account in one place -- (text, how many occurrences struck, the longer
    sentences or headings that still hold it). Every whole-sentence occurrence of the sentence (as _norm reads it) goes;
    a line, list item or «[N]» left bare goes with it; a table cell it emptied reads «—» (the row stays).
    allow(text, start): only the occurrences it admits (the removal's own section, the client's own text) are struck;
    keep_first: the first of them stays (a repetition's removal keeps the sentence it repeated)."""
    needle = _norm(sentence or "")
    if not needle:
        return text, 0, []
    struck, held = 0, []
    while True:
        hay, where = _norm_map(text)
        spans, k = [], hay.find(needle)
        while k >= 0:
            spans.append((where[k], where[k + len(needle) - 1] + 1))
            k = hay.find(needle, k + 1)
        whole = [w for w in (_whole(text, s, e) for s, e in spans if allow is None or allow(text, s)) if w]
        held = [_line(text, s, e)[:200] for (s, e) in spans if not _whole(text, s, e)]
        if keep_first:
            whole = whole[1:]
        if not whole:
            break
        s, e = whole[-1]
        line_start = text.rfind("\n", 0, s) + 1
        line_end = text.find("\n", e)
        line_end = len(text) if line_end < 0 else line_end
        line = text[line_start:s] + text[e:line_end]
        if _TABLE_ROW_RE.match(line):   # a table cell it emptied keeps its row: «—»
            line = "|".join(c if c.strip() or i in (0, len(line.split("|")) - 1) else " — "
                            for i, c in enumerate(line.split("|")))
        if _BARE_LINE_RE.match(line) or not line.strip():
            text = (text[:line_start] + text[line_end + 1:]).lstrip("\n") if line_start == 0 else \
                text[:line_start] + text[line_end + 1:] if line_end < len(text) else text[:max(0, line_start - 1)]
        else:
            text = text[:line_start] + line.rstrip() + text[line_end:]
        struck += 1
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text, struck, held


# v79.10.5 (E3): the rules of the one check whose removal holds in its section only -- the lead's alternative, the
# asserting sections' requirement, a repetition of its section's earlier sentence: a copy elsewhere is no removal of theirs
SECTION_RULES = ("alternativa_senza_fatti", "requisito_non_registrato", "ripetizione")
REPEAT_RULE = "ripetizione"
_CLIENT_HEAD_RE = re.compile(r"(?m)^\*\*Nell'interesse di ([^*\n]+)\*\*\s*$")


def _section_at(text: str, pos: int) -> str:
    """The title of the «**N. Title**» section holding pos ("" before the first)."""
    title = ""
    for m in _SECTION_RE.finditer(text):
        if m.start() > pos:
            break
        title = m.group(1).strip()
    return title


def _clients_at(text: str, pos: int) -> Optional[List[str]]:
    """The clients the «**Nell'interesse di …**» block holding pos is for (its section's), None outside any."""
    floor = max((m.start() for m in _SECTION_RE.finditer(text) if m.start() <= pos), default=0)
    head = None
    for m in _CLIENT_HEAD_RE.finditer(text, floor):
        if m.start() > pos:
            break
        head = m
    if head is None:
        return None
    return [x.strip() for x in re.split(r",\s*|\s+e\s+", head.group(1)) if x.strip()]


def _scope(entry: Dict[str, Any], text: str) -> Tuple[Optional[Callable[[str, int], bool]], str]:
    """(allow, why) for strike: the one check's removal is its section's (a copy in another section is that section's,
    «section»; with its section gone, a SECTION_RULES copy is kept, another struck anywhere); a client's compose removal
    is that client's text's (another client's own block keeps its copy, «client»)."""
    if entry.get("client"):
        name = str(entry["client"]).strip()
        return (lambda t, p: name in (_clients_at(t, p) or [name])), "client"
    if entry.get("step") == CHECK_STEP and entry.get("section"):
        title = str(entry["section"]).strip()
        if any(m.group(1).strip() == title for m in _SECTION_RE.finditer(text)):
            return (lambda t, p: _section_at(t, p) == title), "section"
        if entry.get("rule") in SECTION_RULES:
            return (lambda t, p: False), "section"
    return None, ""


def settle(text: str, entries: List[Dict[str, Any]], code_texts: Sequence[str] = ()) -> Tuple[str, Dict[str, Any]]:
    """v79.10.5 (E3): the final text and the ledger made to agree -- for each «removed» entry still in the text
    (removed_present): struck again where the removal held (strike: a later step reprinted it) and marked settled
    «struck»; what it cannot strike is «kept» with why, never «removed»: part of a longer sentence or a heading
    («contained»), a sentence the code itself writes (code_texts: the method's sources, «code»), a copy outside the
    removal's scope (_scope: another section for the one check's per-section rules, «section»; another client's own
    block, «client»), the sentence a repetition repeated («ripetizione»). After it, removed_present(text, entries) is
    empty. The report: {"struck", "kept", "present"} (the sentences)."""
    report: Dict[str, List[str]] = {"struck": [], "kept": [], "present": []}
    code = [_norm(x) for x in code_texts or [] if x]
    for e in removed_present(text, entries):
        needle = _norm(e.get("sentence") or "")
        if needle not in _norm(text):       # an earlier entry's strike took it
            continue
        report["present"].append(str(e.get("sentence"))[:200])
        why = "code" if any(needle in c for c in code) else ""
        held: List[str] = []
        if not why:
            allow, outside = _scope(e, text)
            repeat = e.get("rule") == REPEAT_RULE
            text, n, held = strike(text, e.get("sentence") or "", allow, repeat)
            if needle not in _norm(text):
                e["settled"] = "struck"
                report["struck"].append(str(e.get("sentence"))[:200])
                continue
            if n:
                e["struck"] = n
            hay, where = _norm_map(text)
            starts = [where[k] for k in _finds(hay, needle)]
            if repeat and any(allow is None or allow(text, p) for p in starts):
                why = REPEAT_RULE           # the sentence it repeated, kept by the rule itself
            elif allow is not None and any(not allow(text, p) for p in starts):
                why = outside
            else:
                why = "contained"
        e.update({"kind": "kept", "was": "removed", "kept_why": why})
        if held:
            e["now"] = held[0]
        report["kept"].append(str(e.get("sentence"))[:200])
    return text, report


def _finds(hay: str, needle: str) -> List[int]:
    out, k = [], hay.find(needle)
    while k >= 0:
        out.append(k)
        k = hay.find(needle, k + 1)
    return out
