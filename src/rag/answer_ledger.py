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
