"""The fact record of the main user query (flag FACT_RECORD, default off): a temporary JSON, built once per case.

Why. The steps that decide (the fit, the element links, the analysis step) and §2 read the case as numbered sentences
(issue_records.sentences): the text cut at full stops by a rule, each piece cut at 260 characters. On the review's
twelve cases that is 3 to 6 pieces a case, about 150 characters each: one piece holds several facts and gets one
certainty label for all of them, and whatever the word lists do not know as a request (a pasted earlier answer, an
instruction on how to reply) is a fact of the case.

What. The model reads the whole text and returns its elements in order, each with the EXACT words it is made of
(«citazione»), its kind (fatto / richiesta / istruzione / estraneo) and, for a fact, how the text states it (accertato /
allegato / ipotizzato) and who alleges it. The code verifies, the model never writes a fact:
  - a citazione not found in the text (spaces and typographic quotes aside) is dropped -- the fact's text is always the
    span of the user's own words, never the model's;
  - a kind outside the four is a fatto; a status outside the three is the rule's (issue_records.certainty);
  - a fact's certainty is the MORE CAUTIOUS of the model's and the rule's on the same words: the model can make a fact
    less certain (an allegation the span alone does not show), never more;
  - a speaker is kept only for an allegation and only when the name is in the text;
  - what the model leaves out comes back by the rule: a whole sentence always, a stretch of one from «gap_words»
    words on -- no text of the case is lost to a careless reply.
A failed or late call for a piece of the text: that piece by the rule. No piece answered: no record (None), and the
numbered sentences are the rule's, as with the flag off.

Use. interactive_flow.analyse_case starts it beside round 1 (start) and keeps it in the state («fact_record», so it is
saved with the session); interactive_flow.case_sentences hands its facts (facts) to issue_records.sentences, which
numbers them S1.. in place of the rule's split -- uncut, with no ceiling on their number (their words are the case's,
already bounded). Requests, instructions and extraneous text stay in the record and are never a sentence.

The record: {"version", "hash", "chars", "status" ok | partial, "facts": [{"id" F1.., "text", "start", "end" (offsets in
the text read), "tipo", "stato", "chi", "source" modello | regola}], "counters", "chunks", "answered", "ms"}.
Logged «fact_record» (counts only). Prompt and numbers: data/fact_record.json.
"""
import hashlib
import json
import logging
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .feature_flags import enabled as _flag_enabled

logger = logging.getLogger(__name__)

FLAG = "FACT_RECORD"
DATA_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data",
                         "fact_record.json")
VERSION = 1
FACT, REQUEST, INSTRUCTION, OTHER = "fatto", "richiesta", "istruzione", "estraneo"
KINDS = (FACT, REQUEST, INSTRUCTION, OTHER)
MODEL, RULE = "modello", "regola"
MIN_CHARS = 12                      # issue_records._split's own floor for a sentence
_WORD_RE = re.compile(r"[a-zà-ü]{3}", re.IGNORECASE)
_QUOTES = str.maketrans({"’": "'", "‘": "'", "`": "'", "“": '"', "”": '"', "„": '"', "«": '"', "»": '"'})
_EDGE = " ,;:.-–—()\"'"             # what a left-out stretch is trimmed of before it is weighed
_JOINERS = {"e", "ed", "ma", "che", "o", "oppure", "mentre", "poi", "quindi"}       # and the words that only join
_FIRST_WORD_RE, _LAST_WORD_RE = re.compile(r"[^\W\d_]+"), re.compile(r"[^\W\d_]+$")


def enabled() -> bool:
    return _flag_enabled(FLAG)


@lru_cache(maxsize=1)
def config() -> Dict[str, Any]:
    with open(DATA_FILE, encoding="utf-8") as f:
        return json.load(f)


def _log(event: str, data: Dict[str, Any]) -> None:
    try:
        from .verbose_logger import vlog
        vlog(event, data)
    except Exception:           # noqa: BLE001 -- the log never fails the step
        pass


def _norm(text: str) -> Tuple[str, List[int]]:
    """The text with every run of spaces as one and typographic quotes as plain ones, and for each of its characters
    the index it has in the original."""
    out: List[str] = []
    pos: List[int] = []
    space = True
    for i, ch in enumerate(text or ""):
        if ch.isspace():
            if not space:
                out.append(" ")
                pos.append(i)
                space = True
            continue
        out.append(ch.translate(_QUOTES))
        pos.append(i)
        space = False
    if out and out[-1] == " ":
        out.pop()
        pos.pop()
    return "".join(out), pos


def _span_text(text: str, pos: Sequence[int], a: int, b: int) -> str:
    """The user's own words of the normalised span [a, b), spaces collapsed."""
    return " ".join(text[pos[a]:pos[b - 1] + 1].split())


def _find(quote: str, norm: str, cursor: int, start: int) -> Optional[Tuple[int, int]]:
    """Where the quote is in the normalised text: from the cursor (the previous element), else from the piece's start,
    else anywhere; as written, else without closing punctuation or enclosing quotes."""
    q = _norm(quote)[0]
    for cand in dict.fromkeys((q, q.rstrip(" .;,:…"), q.strip("\"' "), q.strip("\"' ").rstrip(" .;,:…"))):
        if len(cand) < 2:
            continue
        for begin in dict.fromkeys((cursor, start, 0)):
            i = norm.find(cand, begin)
            if i >= 0:
                return i, i + len(cand)
    return None


def rule_spans(text: str, norm: str) -> List[Tuple[int, int]]:
    """The rule's sentences of the text (issue_records' own split, uncut), as spans of the normalised text."""
    from . import issue_records
    from .sentence_split import split
    spans: List[Tuple[int, int]] = []
    cursor = 0
    for line in re.split(r"\n+", text or ""):
        for piece in split(line, issue_records._SENTENCE_RE):
            p = _norm(piece)[0]
            i = norm.find(p, cursor) if p else -1
            if i < 0:
                continue
            cursor = i + len(p)
            spans.append((i, cursor))
    return spans


def chunks(spans: Sequence[Tuple[int, int]], size: int) -> List[Tuple[int, int]]:
    """The text in pieces of about `size` characters, each a run of whole sentences (one longer sentence: its own)."""
    out: List[Tuple[int, int]] = []
    for a, b in spans:
        if out and b - out[-1][0] <= size:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
    return out


def ask(chunk: str) -> List[Dict[str, Any]]:
    """One model call: the elements of a piece of the text, as the reply gives them (verify checks them)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    from . import defensive_generation as dg
    c = config()
    tokens = min(int(c["tokens_max"]), int(c["tokens_base"]) + int(len(chunk) * float(c["tokens_per_char"])))
    raw = dg._chat_json([SystemMessage(content=c["prompt_system"]), HumanMessage(content=f"TESTO:\n{chunk}")],
                        max_tokens=tokens, temperature=0)
    items = raw.get("elementi") if isinstance(raw, dict) else raw
    return [x for x in items or [] if isinstance(x, dict)]


def _status(said: Any, words: str, counters: Counter) -> str:
    """A fact's certainty: the more cautious of the model's and the rule's on the same words."""
    from . import issue_records as ir
    rank = {ir.ACCERTATO: 0, ir.ALLEGATO: 1, ir.IPOTIZZATO: 2}
    rule = ir.certainty(words)
    value = str(said or "").strip().lower()
    if value not in rank:
        counters["status_unknown"] += 1
        return rule
    if rank[value] < rank[rule]:
        counters["status_kept_cautious"] += 1
        return rule
    return value


def verify(items: Sequence[Dict[str, Any]], text: str, norm: str, pos: Sequence[int], start: int,
           counters: Counter) -> List[Dict[str, Any]]:
    """The reply's elements that are the text's own words, each with its span; the others counted and dropped."""
    from . import issue_records as ir
    out: List[Dict[str, Any]] = []
    cursor, low = start, norm.lower()
    for item in items or []:
        hit = _find(str(item.get("citazione") or ""), norm, cursor, start)
        if hit is None:
            counters["quote_not_found"] += 1
            continue
        a, b = hit
        words = _span_text(text, pos, a, b)
        if len(words) < MIN_CHARS or not _WORD_RE.search(words):
            counters["quote_too_short"] += 1
            continue
        cursor = a
        kind = str(item.get("tipo") or "").strip().lower()
        if kind not in KINDS:
            counters["kind_unknown"] += 1
            kind = FACT
        fact = {"text": words, "start": pos[a], "end": pos[b - 1] + 1, "tipo": kind, "source": MODEL, "_n": (a, b)}
        if kind == FACT:
            fact["stato"] = _status(item.get("stato"), words, counters)
            who = " ".join(str(item.get("chi") or "").split())
            if who and fact["stato"] == ir.ALLEGATO and _norm(who)[0].lower() in low:
                fact["chi"] = who
            elif who:
                counters["speaker_dropped"] += 1
        out.append(fact)
    return out


def _uncovered(span: Tuple[int, int], covered: Sequence[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """The parts of the span no covered interval holds (covered: sorted, merged)."""
    a, b = span
    out: List[Tuple[int, int]] = []
    for x, y in covered:
        if y <= a or x >= b:
            continue
        if x > a:
            out.append((a, x))
        a = max(a, y)
    if a < b:
        out.append((a, b))
    return out


def _trim(norm: str, a: int, b: int) -> Tuple[int, int]:
    """A left-out stretch without the punctuation and the joining words («e», «ma», «che») at its two ends."""
    while a < b:
        if norm[a] in _EDGE:
            a += 1
        elif norm[b - 1] in _EDGE:
            b -= 1
        else:
            first, last = _FIRST_WORD_RE.match(norm[a:b]), _LAST_WORD_RE.search(norm[a:b])
            if first and first.group(0).lower() in _JOINERS and first.end() < b - a:
                a += first.end()
            elif last and last.group(0).lower() in _JOINERS and last.start() > 0:
                b -= len(last.group(0))
            else:
                break
    return a, b


def fill(facts: Sequence[Dict[str, Any]], text: str, norm: str, pos: Sequence[int], spans: Sequence[Tuple[int, int]],
         gap_words: int, counters: Counter) -> List[Dict[str, Any]]:
    """What the verified elements leave out, by the rule: a whole sentence always (the rule's own floor), a stretch of
    one from gap_words words on; shorter stretches (connectives, punctuation) are counted, not kept."""
    from . import fact_signals, issue_records as ir
    covered: List[Tuple[int, int]] = []
    for a, b in sorted(f["_n"] for f in facts):
        if covered and a <= covered[-1][1]:
            covered[-1] = (covered[-1][0], max(covered[-1][1], b))
        else:
            covered.append((a, b))
    out: List[Dict[str, Any]] = []
    for span in spans:
        for a, b in _uncovered(span, covered):
            whole = (a, b) == span
            if not whole:
                a, b = _trim(norm, a, b)
            words = _span_text(text, pos, a, b) if b > a else ""
            if len(words) < MIN_CHARS or not _WORD_RE.search(words) or (not whole and len(words.split()) < gap_words):
                counters["left_out_chars"] += len(words)
                continue
            kind = REQUEST if fact_signals.user_message(words) else FACT
            fact = {"text": words, "start": pos[a], "end": pos[b - 1] + 1, "tipo": kind, "source": RULE, "_n": (a, b)}
            if kind == FACT:
                fact["stato"] = ir.certainty(words)
            counters["rule_whole" if whole else "rule_stretch"] += 1
            out.append(fact)
    return out


def build(case: str, asker: Optional[Callable[[str], List[Dict[str, Any]]]] = None,
          deadline: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """The record of the case text; None when there is no text or no piece of it was answered in time (the caller
    then keeps the rule's split). asker: the model call (ask), replaceable in a test."""
    from . import fact_signals
    t0 = time.time()
    c = config()
    # an upload's request header and the mark of a cut text are the app's own words, never the user's
    text = fact_signals._REQUEST_RE.sub("", case or "", count=1).replace("[... documento troncato ...]", "")
    norm, pos = _norm(text)
    spans = rule_spans(text, norm)
    if not spans:
        return None
    pieces = chunks(spans, int(c["chunk_chars"]))
    counters: Counter = Counter()
    limit = t0 + float(c["deadline_s"] if deadline is None else deadline)
    pool = ThreadPoolExecutor(max_workers=max(1, min(int(c["workers"]), len(pieces))), thread_name_prefix="fact_record")
    futures = [pool.submit(asker or ask, text[pos[a]:pos[b - 1] + 1]) for a, b in pieces]
    facts: List[Dict[str, Any]] = []
    answered = 0
    for (a, _), future in zip(pieces, futures):
        try:
            items = future.result(timeout=max(0.0, limit - time.time()))
        except FutureTimeout:
            counters["chunks_late"] += 1
            continue
        except Exception as exc:        # noqa: BLE001 -- a failed piece is read by the rule
            counters["chunks_failed"] += 1
            logger.warning("fact_record: a piece was not read: %s: %s", type(exc).__name__, str(exc)[:200])
            continue
        answered += 1
        facts += verify(items, text, norm, pos, a, counters)
    pool.shutdown(wait=False, cancel_futures=True)
    if not answered:
        _log("fact_record", {"status": "failed", "chunks": len(pieces), "counters": dict(counters),
                             "ms": round((time.time() - t0) * 1000)})
        return None
    facts += fill(facts, text, norm, pos, spans, int(c["gap_words"]), counters)
    facts.sort(key=lambda f: f["_n"])
    kept: List[Dict[str, Any]] = []
    for f in facts:                     # the same words twice: once
        if kept and kept[-1]["_n"] == f["_n"]:
            counters["repeated"] += 1
            continue
        kept.append(f)
    for n, f in enumerate(kept, 1):
        f["id"] = f"F{n}"
        del f["_n"]
    record = {"version": VERSION, "hash": hashlib.sha1(norm.encode("utf-8")).hexdigest()[:12], "chars": len(text),
              "status": "ok" if answered == len(pieces) else "partial", "facts": kept, "counters": dict(counters),
              "chunks": len(pieces), "answered": answered, "ms": round((time.time() - t0) * 1000)}
    _log("fact_record", {**{k: record[k] for k in ("status", "hash", "chars", "chunks", "answered", "counters", "ms")},
                         "kinds": dict(Counter(f["tipo"] for f in kept)),
                         "sources": dict(Counter(f["source"] for f in kept)),
                         "states": dict(Counter(f.get("stato") for f in kept if f["tipo"] == FACT))})
    return record


class Pending:
    """The record being built beside round 1. result() waits at most what is left of the step's deadline."""

    def __init__(self, case: str):
        self.t0 = time.time()
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fact_record_step")
        self._future = self._pool.submit(build, case)

    def result(self) -> Optional[Dict[str, Any]]:
        try:
            left = float(config()["deadline_s"]) + 2.0 - (time.time() - self.t0)
            return self._future.result(timeout=max(0.0, left))
        except Exception as exc:        # noqa: BLE001 -- no record: the rule's sentences, as with the flag off
            logger.warning("fact_record: no record: %s: %s", type(exc).__name__, str(exc)[:200])
            _log("fact_record", {"status": "late" if isinstance(exc, FutureTimeout) else "failed"})
            return None
        finally:
            self._pool.shutdown(wait=False)


def start(case: str) -> Optional[Pending]:
    """The record's build started in its own thread; None with the flag off or no text."""
    return Pending(case) if enabled() and (case or "").strip() else None


def facts(state: Optional[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """The state's record's facts of the case (kind fatto), in the text's order; [] with the flag off or no record."""
    record = (state or {}).get("fact_record")
    if not enabled() or not isinstance(record, dict) or record.get("version") != VERSION:
        return []
    if record.get("status") != "ok":     # v79.16.0 (A1): a partial record never drives the sentences -- the rule's
        return []                        # split instead (interactive_flow.case_sentences: fact_record_fallback = 1)
    return [f for f in record.get("facts") or [] if isinstance(f, dict) and f.get("tipo") == FACT and f.get("text")]


def fallback(state: Optional[Dict[str, Any]]) -> bool:
    """v79.16.0 (A1): with the flag on, the rule's split stands in for the record -- no record, or one not «ok»."""
    if not enabled():
        return False
    record = (state or {}).get("fact_record")
    return not isinstance(record, dict) or record.get("version") != VERSION or record.get("status") != "ok"


def uncertain_items(state: Optional[Dict[str, Any]]) -> List[Dict[str, str]]:
    """v79.16.0 (A3): §10's items from the record -- a fact the text states as a hypothesis («ipotizzato») and a
    stretch the model left out, read back by the rule («regola»): [{"kind": ipotesi | non_letto, "text"}], in the
    text's order; [] with no record in use."""
    out: List[Dict[str, str]] = []
    for f in facts(state):
        if f.get("stato") == "ipotizzato":
            out.append({"kind": "ipotesi", "text": str(f["text"])})
        elif f.get("source") == RULE:
            out.append({"kind": "non_letto", "text": str(f["text"])})
    return out
