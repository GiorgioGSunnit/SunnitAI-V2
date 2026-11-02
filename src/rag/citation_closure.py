"""CITATION_CLOSURE_ENABLED (default false): after writing, no citation stays unverified.

The v37 32-case report: 44 citations in 32 answers named an article the case's material did not hold (2059, 1917, 124 c.p.,
696, 700, 76 c.p.p., 1158, 1667/1669, 2947...; the writer read them in a ruling's passage), 20 of them unmarked. On the
written answer, before _mark_unverified:

  1. RESOLVE  a bare number no reference and no material article resolves (plan_then_write.read_citations) is looked up
              in the case's area codes: found in exactly one, it is that article; in several or none it stays unresolved.
  2. SCOPE    a citation of an act outside the case's material and areas (scope_codes) is not read.
  3. FETCH    every other article not in the material is read by number, whole (article_text, through dg._fetch_any).
  4. CHECK    each sentence citing it goes through the qualification check (answer_invariants.judge), on the text read.

Compatible, the article becomes verified material: its row joins article_rows (_closure, CLOSURE_SOURCE), so it reaches the
sources and no later step marks or removes it. Not found, incompatible, without a verdict, outside the scope or unresolved:
the citation goes, its sentence kept when it stands without it, else dropped. An article verified in one sentence and
incompatible in another stays only in the first. Logged as case_analysis_citation_closure; `remaining` is 0 by design.
"""

import logging
import re
import time
from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from . import article_text, feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

FLAG = "CITATION_CLOSURE_ENABLED"
CLOSURE_SOURCE = "citation_closure"
ROW_KEY = "_closure"
TEXT_CHARS = 1500            # of the article read, for the check
MIN_WORDS = 6                # a sentence left shorter does not stand alone
_AREA_CODES = {"penale": ("c.p.", "c.p.p.", "c.c."), "amministrativo": ("c.p.a.", "c.c."), "civile": ("c.c.", "c.p.c.")}
_TAIL_RE = re.compile(r"(?:\b(?:e|o|ed|od|di|del|della|dei|delle|che|il|la|le|un|una|per|con|a|ai|al|alla|da|in|su|ma|ex))\s*[.;:,]?\s*$",
                      re.IGNORECASE)


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


def _areas(analysis: Dict[str, Any]) -> List[str]:
    from . import method_candidates
    return method_candidates.areas_for(((analysis or {}).get("frame") or {}).get("areas") or [], (analysis or {}).get("area") or "")


def area_codes(analysis: Dict[str, Any]) -> List[str]:
    """The codes of the case's areas; with no area, the six."""
    from . import defensive_generation as dg
    from . import method_candidates
    areas = _areas(analysis)
    if not areas:
        return list(dg._CODE_PREFIXES)
    return list(dict.fromkeys(c for a in areas for c in _AREA_CODES[method_candidates.area_family(a)]))


def material_refs(analysis: Dict[str, Any], article_rows: List[Dict[str, Any]], named: List[tuple]) -> List[tuple]:
    """The case's material: every article row, what the case names, the procedural tools and lists, the computed terms."""
    from . import answer_invariants, method_candidates
    procedural = method_candidates.candidate_refs([], method_candidates.case_lists(analysis)[1])
    rows = [r["_article"] for r in article_rows if r.get("_article")]
    extra = answer_invariants.allowed_refs([], named, (analysis or {}).get("fact_signals") or {}, procedural)
    return list(dict.fromkeys([tuple(r) for r in rows] + extra))


def scope_codes(analysis: Dict[str, Any], material: List[tuple]) -> Set[str]:
    """The acts a citation may be attached to, by sigla or name: the material's, the areas' codes, their method lists'."""
    from . import method_candidates
    from . import plan_then_write as ptw
    listed = method_candidates.candidate_refs(_areas(analysis), "")
    return {ptw._code_key(code) for _, code in list(material) + listed} | set(area_codes(analysis))


def _in_scope(ref: tuple, scope: Set[str]) -> bool:
    from . import defensive_generation as dg
    from . import plan_then_write as ptw
    code = ptw._code_key(ref[1])
    return code in scope or (code not in dg._CODE_PREFIXES and any(code.lower() in s.lower() or s.lower() in code.lower()
                                                                   for s in scope if s not in dg._CODE_PREFIXES))


def _text(row: Dict[str, Any]) -> str:
    return " ".join(str((row.get("s") or {}).get("plain_text") or "").split())


def _usable(row: Dict[str, Any]) -> bool:
    text = _text(row)
    return bool(text) and text != "(abrogato)"


def _read(session, documents: List[Dict[str, Any]], refs: List[tuple]) -> Dict[tuple, Dict[str, Any]]:
    """{ref: its first row with a text in force}, one read."""
    from . import defensive_generation as dg
    rows: Dict[tuple, Dict[str, Any]] = {}
    for row in dg._fetch_any(session, documents, refs):
        if row.get("_article") and _usable(row):
            rows.setdefault(row["_article"], row)
    return rows


def _unresolved_numbers(reading: List[tuple]) -> List[str]:
    return list(dict.fromkeys(n for _, _, numbers, refs, _ in reading for n, r in zip(numbers, refs) if r is None))


def _in_areas(numbers: List[str], analysis: Dict[str, Any], known: set, session,
              documents: List[Dict[str, Any]]) -> Dict[str, tuple]:
    """{number: the one article of the areas' codes holding it}: a number several known articles share stays ambiguous."""
    open_numbers = [n for n in numbers if not any(k[0] == n for k in known)]
    if not open_numbers:
        return {}
    found = _read(session, documents, [(n, c) for n in open_numbers for c in area_codes(analysis)])
    return {n: same[0] for n in open_numbers for same in [[r for r in found if r[0] == n]] if len(same) == 1}


def stands(sentence: str) -> bool:
    """Does a sentence left without a citation still say something?"""
    from . import plan_then_write as ptw
    words = re.findall(r"\w+", sentence)
    return len(words) >= MIN_WORDS and not _TAIL_RE.search(sentence) and not ptw._dangling(sentence, False)


def _in_the_case(number: str, case_text: str) -> bool:
    return bool(re.search(rf"\b(?:artt?\.?|articol[oi])\s*{re.escape(number)}\b", case_text or "", re.IGNORECASE))


def _span_of(spans: List[Tuple[int, int]], at: int) -> Optional[Tuple[int, int]]:
    return next(((s, e) for s, e in spans if s <= at < e), None)


def _apply(text: str, drops: Dict[Tuple[int, int], List[tuple]], documents: List[Dict[str, Any]]
           ) -> Tuple[str, List[Dict[str, Any]]]:
    """(text, log): each span without its citations (anchored_writing.drop_citations), kept when it stands and cites none
    of them any more (a run naming a kept article too stays whole), else dropped."""
    from . import anchored_writing
    from . import plan_then_write as ptw
    log, gone, offset = [], [], 0
    for (s, e), refs in sorted(drops.items()):
        s, e = s + offset, e + offset
        old = text[s:e]
        new = anchored_writing.drop_citations(old, refs, documents, set(refs))
        kept = new != old.strip() and stands(new) and not ptw._citing(new, refs, documents, set(refs))
        log.append({"refs": [f"{n} {c}".strip() for n, c in refs], "how": "citation removed" if kept else "sentence dropped",
                    "sentence": " ".join(old.split())[:300], "now": new[:300] if kept else ""})
        if kept:
            lead = len(old) - len(old.lstrip())
            new = old[:lead] + new + old[len(old.rstrip()):]
            text = text[:s] + new + text[e:]
            offset += len(new) - len(old)
        else:
            gone.append((s, e))
    return (ptw._drop_spans(text, gone) if gone else text), log


def _clear_marks(text: str, verified: List[tuple]) -> str:
    """A "[DA VERIFICARE]" the writer put after a citation now verified goes."""
    from . import defensive_generation as dg
    out, pos = [], 0
    for m in dg._CITED_ARTICLE_RE.finditer(text):
        ref = (dg._article_num(m.group(1)), dg._code_of(m.group(2)))
        mark = re.match(r"\s*\[DA VERIFICARE[^\]]*\]", text[m.end():])
        if mark and ref in verified:
            out.append(text[pos:m.end()])
            pos = m.end() + mark.end()
    return "".join(out) + text[pos:]


def close(draft: str, *, analysis: Dict[str, Any], article_rows: List[Dict[str, Any]], documents: List[Dict[str, Any]],
          session, case_text: str, chat_json: Optional[Callable] = None) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
    """(draft, the verified rows to add to article_rows, log), by the module's rule. An error reading the articles
    propagates: the caller keeps the draft, and _mark_unverified marks what it can."""
    from . import answer_invariants
    from . import defensive_generation as dg
    from . import plan_then_write as ptw
    t0 = time.time()
    chat_json = chat_json or dg._chat_json
    named = dg._named_refs(case_text, documents)
    material = material_refs(analysis, article_rows, named)
    known = set(material)
    scope = scope_codes(analysis, material)
    by_area = _in_areas(_unresolved_numbers(ptw.read_citations(draft, documents, known)), analysis, known, session, documents)
    known |= set(by_area.values())
    reading = ptw.read_citations(draft, documents, known)
    spans = ptw._sentence_spans(draft)
    drops: Dict[Tuple[int, int], List[tuple]] = {}
    citing: Dict[tuple, List[Tuple[int, int]]] = {}
    outcomes: Counter = Counter()
    entries: List[Dict[str, Any]] = []

    def drop(span, ref, why, label):
        if span:
            drops.setdefault(span, [])
            if ref not in drops[span]:
                drops[span].append(ref)
        outcomes[why] += 1
        entries.append({"ref": label, "how": "removed", "why": why, "sentence": " ".join(draft[span[0]:span[1]].split())[:300] if span else ""})

    for start, _, numbers, refs, _ in reading:
        span = _span_of(spans, start)
        for number, ref in zip(numbers, refs):
            if ref is None and _in_the_case(number, case_text):
                outcomes["the case's own"] += 1      # the user's "art. 5": no act to read, and nothing the answer made up
            elif ref is None:
                drop(span, (number, ""), "unresolved", number)
            elif not any(ptw._same_ref(ref, m) for m in material):
                if _in_scope(ref, scope):
                    if span and span not in citing.setdefault(ref, []):
                        citing[ref].append(span)
                else:
                    drop(span, ref, "outside the case's material and areas", f"{ref[0]} {ref[1]}")
    rows = _read(session, documents, list(citing)) if citing else {}
    items: List[Dict[str, Any]] = []
    for ref, where in citing.items():
        row = next((r for k, r in rows.items() if ptw._same_ref(k, ref)), None)
        if row is None:
            for span in where:
                drop(span, ref, "not found", f"{ref[0]} {ref[1]}")
            continue
        heading = article_text.heading(row.get("s"))
        items += [{"span": span, "ref": ref, "heading": heading or "-", "sentence": " ".join(draft[span[0]:span[1]].split()),
                   "opening": article_text.clause_cut(_text(row), TEXT_CHARS)} for span in where]
    bad, answered = answer_invariants.judge(items, chat_json) if items else ({}, set())
    verified: Dict[tuple, Dict[str, Any]] = {}
    for k, item in enumerate(items):
        ref, label = item["ref"], f"{item['ref'][0]} {item['ref'][1]}"
        if k not in answered:
            drop(item["span"], ref, "no verdict", label)
        elif k in bad:
            drop(item["span"], ref, "incompatible", label)
            entries[-1]["reason"] = bad[k]
        else:
            row = next(r for key, r in rows.items() if ptw._same_ref(key, ref))
            verified[ref] = {**row, "_article": ref, "_source": CLOSURE_SOURCE, ROW_KEY: True, "_applicability": "applicabile"}
            outcomes["verified"] += 1
            entries.append({"ref": label, "how": "verified", "sentence": item["sentence"][:300]})
    text, removal = _apply(draft, drops, documents)
    text = _clear_marks(text, list(verified))
    if drops:
        text, dangling = ptw.drop_dangling(text, "answer")
        removal += dangling
    allowed = material + list(verified)
    left = [r for _, _, numbers, refs, _ in ptw.read_citations(text, documents, set(allowed))
            for n, r in zip(numbers, refs) if not (r is None and _in_the_case(n, case_text))
            and (r is None or not any(ptw._same_ref(r, a) for a in allowed))]
    resolved = Counter(how for *_, hows in reading for how in hows)
    resolved.update({"area": len(by_area)})
    log = {"resolved": dict(resolved), "verified": sorted({e["ref"] for e in entries if e["how"] == "verified"}), "counts": dict(outcomes),
           "by_area": {n: f"{r[0]} {r[1]}" for n, r in by_area.items()}, "entries": entries[:60], "removal": removal[:60],
           "sentences_dropped": sum(1 for e in removal if e.get("how") == "sentence dropped"),
           "citations_removed": sum(1 for e in removal if e.get("how") == "citation removed"),
           "remaining": len(left), "remaining_refs": [f"{r[0]} {r[1]}" if r else "unresolved" for r in left][:20]}
    vlog("case_analysis_citation_closure", log, duration_ms=(time.time() - t0) * 1000)
    return text, list(verified.values()), log
