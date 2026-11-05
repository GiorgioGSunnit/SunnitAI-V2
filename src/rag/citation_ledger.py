"""The citation ledger (v46, always on): one citation_removed event per article for every sentence citing it that a pass
after the draft drops or rewrites -- writer review, citation discipline, unsupported claims, anchored writing, final
checks, plan cover's rewrites, citation closure, the qualification check, answer tidy. Each event: the pass, the article,
the reason, the sentence's first 80 characters, and "dropped": whether the article is gone from what the pass left of
the sentence (a rewrite keeping the citation is false). scripts/diagnose_long.py puts, per expected article, the last
pass that dropped it in its slot (removed_by).

The passes call record_entries on the entries they already log: {"sentence"} with {"how"} or {"kind"}, a reason in
{"why"} or {"reason"}, what the pass left in {"now"} (absent or empty: the sentence went), the articles taken out by name
in {"removed"} or {"refs"}, and the whole sentence in {"in"} when "sentence" holds only a part of it (a copied run).
Articles are read from the sentence by their code (defensive_generation._refs_in), so a law named by its title counts
only when the entry names it.

v49, protected citations (no flag): once the plan is final (plan_then_write.write_parere: protect), a sentence citing one
of its articles is never dropped by answer_tidy, final_checks or anchored_writing -- they trim the clause or tie it to
the article's own text (a copied run quoted) instead, or keep it as written; each such sentence logs a citation_removed
event with "dropped": false and "kept": "plan article" (kept). The qualification check keeps its drop, its reason in
full (REASON_CHARS)."""

import threading
from typing import Any, Dict, Iterable, List, Optional

EVENT = "citation_removed"
SENTENCE_CHARS = 80
REASON_CHARS = 1000         # v49: the qualification check's reason in full (was 200)

# Entries that change nothing in the sentence (a log line, a kept or verified one), or no sentence at all.
_UNCHANGED = {"kept", "logged", "verified", "marked", "fact_flag_filtered", "stop before a comma, or doubled",
              "comma past the article's last"}


_local = threading.local()
KEPT = "plan article"


def protect(refs: Iterable[Any]) -> None:
    """The plan's articles, for this case (thread): their sentences are never dropped by the protected passes."""
    _local.plan = [tuple(r) for r in refs or []]


def plan_cited(sentence: str) -> List[str]:
    """The plan's articles a sentence cites, by their code; [] with no plan."""
    plan = getattr(_local, "plan", None) or []
    if not plan or not sentence:
        return []
    from . import defensive_generation as dg
    from . import plan_then_write as ptw
    return list(dict.fromkeys(_label(p) for ref in dg._refs_in(sentence) for p in plan if ptw._same_ref(ref, p)))


def kept(pass_name: str, sentence: str, reason: str, now: Optional[str] = None) -> List[Dict[str, Any]]:
    """The events for a sentence a protected pass would have dropped and kept, trimmed or quoted instead (now: what it
    left, when it changed it): one per plan article it cites, "dropped" false."""
    out = []
    try:
        from . import defensive_generation as dg
        for label in plan_cited(sentence):
            event = {"pass": pass_name, "article": label, "reason": f"kept: {KEPT} ({str(reason or '')[:REASON_CHARS]})",
                     "sentence": " ".join(str(sentence or "").split())[:SENTENCE_CHARS], "dropped": False, "kept": KEPT,
                     **({"now": " ".join(str(now).split())[:SENTENCE_CHARS]} if now else {})}
            dg.vlog(EVENT, event)
            out.append(event)
    except Exception as exc:
        from .verbose_logger import exception_data
        from . import defensive_generation as dg
        dg.vlog("pipeline_exception", exception_data(f"citation_ledger.kept.{pass_name}", exc))
    return out


def _label(ref) -> str:
    return f"{ref[0]} {ref[1]}"


def _key(label: str) -> str:
    return " ".join(str(label).lower().replace(".", " ").split())


def cited(sentence: str) -> List[str]:
    """The articles a sentence cites by their code, in order, once each."""
    from . import defensive_generation as dg
    return list(dict.fromkeys(_label(r) for r in dg._refs_in(sentence or "")))


def record(pass_name: str, sentence: str, reason: str, now: Optional[str] = None,
           articles: Iterable[str] = (), only: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
    """The events for one sentence a pass dropped (now None or empty) or rewrote (now its new text): one per article it
    cites or the pass names (only: just these). Nothing for a sentence citing none."""
    from . import defensive_generation as dg
    named = [str(a) for a in articles or [] if a]
    labels = list(dict.fromkeys(cited(sentence) + named))
    if only is not None:
        keep = {_key(a) for a in only}
        labels = [a for a in labels if _key(a) in keep]
    if not labels:
        return []
    left = {_key(a) for a in cited(now)} if now else set()
    taken = {_key(a) for a in named}
    out = []
    for label in labels:
        event = {"pass": pass_name, "article": label, "reason": str(reason or "")[:REASON_CHARS],
                 "sentence": " ".join(str(sentence or "").split())[:SENTENCE_CHARS],
                 "dropped": _key(label) in taken or _key(label) not in left}
        dg.vlog(EVENT, event)
        out.append(event)
    return out


def record_entries(pass_name: str, entries: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """record() for each entry of a pass's own log that dropped or rewrote a sentence; never raises (a ledger failure
    must not cost the answer)."""
    out: List[Dict[str, Any]] = []
    try:
        for e in entries or []:
            if not isinstance(e, dict):
                continue
            how = str(e.get("how") or e.get("kind") or "")
            if how in _UNCHANGED or e.get("kind") in _UNCHANGED:
                continue
            sentence = str(e.get("in") or e.get("sentence") or "")
            if not sentence or sentence.startswith("("):        # "(a join)", "(outside a sentence)"
                continue
            reason = " ".join(x for x in (str(e.get("kind") or ""), str(e.get("why") or e.get("reason") or ""))
                              if x and x != how) or how
            reason = f"{how}: {reason}" if reason != how else how
            named = e.get("removed") if isinstance(e.get("removed"), list) else e.get("refs")
            named = [x if isinstance(x, str) else _label(x) for x in named or [] if x]
            out += record(pass_name, sentence, reason, e.get("now"), named)
    except Exception as exc:
        from .verbose_logger import exception_data
        from . import defensive_generation as dg
        dg.vlog("pipeline_exception", exception_data(f"citation_ledger.{pass_name}", exc))
    return out


def last_dropped(events: Iterable[Dict[str, Any]], key) -> Dict[str, Dict[str, Any]]:
    """{key(article): the last event that dropped it}, in the events' order."""
    out: Dict[Any, Dict[str, Any]] = {}
    for e in events or []:
        if isinstance(e, dict) and e.get("dropped") and e.get("article"):
            out[key(e["article"])] = e
    return out
