"""COMMA_MATERIAL_ENABLED (default false): the writer gets an article's commi, not its first N characters.

scripts/fetch_audit.py on run 0812 (no judge): the largest measurable defect was granularity -- sentences relying
on a comma the writer never received, because an article reached it as a prefix cut (_source_block's
`text[:chars]`, _with_text_cut's) or not at all. With the flag:

  1. attach(): after retrieval, each article row gets its parts ("_commi"), in one read for all of them:
     article_text's whole text split (article_parts: opening, commi, list items). "(co. K)" only for a real comma
     (comma_numbers): the number the text gives or its paragraph, never a list item or a sentence. Within the per-article budget, select() keeps the commi whose words best
     match the issue the article serves ("_focus": the plan's issue that places it, before the plan the
     analysis' question it was found for), in article order and labelled "(co. K)"; the commi left out are
     said ("[commi non riportati: 2, 4]"); a comma the case, the request or the plan names ("art. 1384, comma
     2, c.c.") is always in ("_named_commi").
  2. for_plan(): once the plan is made, each article's focus becomes its plan issue (title, argument,
     objection, reply), the commi the plan or the case names are added, and any of them the row does not hold
     is read before the sections are written (ensure(): one read, no model call).
  3. log_material(): per writer call, which commi of each article were given, logged as comma_material
     ({stage, articles: [{ref, total, given, named, from}]}). fetch_audit.py's "comma not given / beyond the
     truncation" is the measure (target near 0).

Off, nothing changes: no read, no row key, the same material.
"""

import logging
import re
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from . import article_text, feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

FLAG = "COMMA_MATERIAL_ENABLED"
COMMI_KEY, FROM_KEY, FOCUS_KEY, NAMED_KEY = "_commi", "_commi_from", "_focus", "_named_commi"
# A text split part shorter than this joins the one before it (a list's "a)" line, a stray footer).
MIN_PART = 40
SHORT_ARTICLE = 1200      # an article this short reaches the writer whole, every comma


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


# --- the commi of an article -----------------------------------------------------------------------------------

_LIST_ITEM_RE = re.compile(r"(?<=[:;.])\s+(?=(?:\d{1,2}|[a-z])(?:-?(?:bis|ter|quater))?\)\s)")
_ITEM_RE = re.compile(r"^(?:\d{1,2}|[a-z])(?:-?(?:bis|ter|quater))?\)\s")
_COMMA_HEAD_RE = re.compile(r"^(\d{1,3})(?:-[a-z]+)?\.\s")
_NUMBERED_RE = re.compile(r"(?<=[.;:])\s+(?=\d{1,2}\.\s+[A-ZÀ-Ü])")
_SENTENCE_RE = re.compile(r"(?<=[.;])\s+(?=[A-ZÀ-Ü])")


def _merge_short(parts: List[str]) -> List[str]:
    out: List[str] = []
    for p in parts:
        if out and len(p) < MIN_PART:
            out[-1] = f"{out[-1]} {p}"
        else:
            out.append(p)
    return out


def _unique(parts: List[str]) -> List[str]:
    """A paragraph the corpus repeats counts once: art. 614-bis c.p.c. holds one twice, and its "comma 11" does not exist."""
    seen, out = set(), []
    for part in parts:
        key = " ".join(part.lower().split())
        if key not in seen:
            seen.add(key)
            out.append(part)
    return out


# v41 (art. 1067 c.c.: 'total': 4, 'from': 'lines'): a line of the PDF is no comma. One ends only after . ; : (or a note's
# "(n)"), and the next must open like one, not with a lower-case word or a bare number ("dell'art.\n1051 c.c.").
_LINE_END_RE = re.compile(r"(?:[.;:][»”\"'’)\]]*|\(abrogato\))$", re.IGNORECASE)
_NOTE_MARK_RE = re.compile(r"\(\d{1,3}\)$")
_NOTE_HEAD_RE = re.compile(r"^\(\d{1,3}\)\s")


def unwrap(text: Optional[str]) -> List[str]:
    """The text's real paragraphs: its lines with the PDF's wrapped ones joined back. A line opens a new paragraph after a
    line ending a clause, or when it is a numbered or lettered item or a note; never in the middle of a sentence."""
    out: List[str] = []
    for line in (p.strip() for p in re.split(r"\n+", text or "")):
        if not line:
            continue
        before = out[-1] if out else ""
        ended = bool(_LINE_END_RE.search(before)) and not (line[0].islower() or line[0].isdigit())
        if not out or _ITEM_RE.match(line) or _COMMA_HEAD_RE.match(line) or _NOTE_HEAD_RE.match(line) or ended \
                or _NOTE_MARK_RE.search(before):
            out.append(line)
        elif before.endswith("-"):
            out[-1] = before[:-1] + line
        else:
            out[-1] = f"{before} {line}"
    return out


def split_commi(text: Optional[str]) -> Tuple[List[str], str]:
    """(the parts of an article's text, how): its paragraphs when it has several (unwrap, never its PDF lines), else its
    numbered paragraphs ("1. ... 2. ..."), else its sentences ("sentences": not real commi, never labelled with a number)."""
    text = (text or "").strip()
    if not text:
        return [], "none"
    lines = _unique(_merge_short(unwrap(text)))
    if len(lines) > 1:
        return lines, "lines"
    flat = " ".join(text.split())
    numbered = _unique(_merge_short([p.strip() for p in _NUMBERED_RE.split(flat) if p.strip()]))
    if len(numbered) > 1:
        return numbered, "numbered"
    return _unique(_merge_short([p.strip() for p in _SENTENCE_RE.split(flat) if p.strip()])) or [flat], "sentences"


def article_parts(text: Optional[str]) -> Tuple[List[str], str]:
    """(the article's opening, commi and list items, how): split_commi, then "1) ..." and "a) ..." inside each."""
    parts: List[str] = []
    commi, how = split_commi(text)
    for comma in commi:
        parts += [p.strip() for p in _LIST_ITEM_RE.split(comma) if p.strip()]
    return parts, how


def comma_numbers(parts: Sequence[str], how: Optional[str]) -> List[Optional[int]]:
    """Each part's real comma: the number the text gives ("2. ..."), else its paragraph, a list item belonging to the
    paragraph above; None for all when the text has no commi to count (sentences of a flat text)."""
    if how in (None, "sections"):
        return list(range(1, len(parts) + 1))
    if how not in ("lines", "numbered"):
        return [None] * len(parts)
    explicit = any(_COMMA_HEAD_RE.match(p) for p in parts)
    out: List[Optional[int]] = []
    k = 0
    for part in parts:
        head = _COMMA_HEAD_RE.match(part)
        if head:
            k = int(head.group(1))
        elif not explicit and not _ITEM_RE.match(part):
            k += 1
        out.append(max(k, 1))
    return out


def commi_from_sections(number: str, texts: Dict[str, str], kinds: Optional[Dict[str, str]] = None,
                        following: Optional[Dict[str, str]] = None, rubrica: str = "") -> Tuple[List[str], str]:
    """(the parts, how) of article `number` from its Sections: article_text's whole text, split (article_parts)."""
    from . import article_text
    return article_parts(article_text.choose_text(number, texts, kinds, following, rubrica))


def _read(session, items: Sequence[Tuple[str, str]]) -> Dict[Tuple[str, str], Tuple[List[str], str]]:
    """{(doc id, number): (commi, how)} for every item, in one query (every spelling of a suffix tried)."""
    from . import article_text
    from .article_suffix import number_variants
    wanted = {(doc, spelling): (doc, number) for doc, number in items for spelling in number_variants(number)}
    grouped = article_text._sections(session, list(wanted))
    out: Dict[Tuple[str, str], Tuple[List[str], str]] = {}
    for (doc, spelling), rows in grouped.items():
        key = wanted[(doc, spelling)]
        if key in out or not rows:
            continue
        commi, how = commi_from_sections(spelling, *article_text._fields(article_text._best_by_name(rows)))
        if commi:
            out[key] = (commi, how)
    return out


def _item(row: Dict[str, Any]) -> Optional[Tuple[str, str]]:
    d = row.get("d") if isinstance(row.get("d"), dict) else {}
    if not row.get("_article") or not d.get("id"):
        return None
    return d["id"], str(row["_article"][0])


def attach(session, rows: Iterable[Dict[str, Any]], stage: str = "") -> int:
    """Each article row's commi read and attached (row["_commi"], row["_commi_from"]); the rows holding them
    already are skipped. How many rows got them. Nothing with the flag off or no session."""
    if not enabled() or session is None:
        return 0
    rows = [r for r in rows if isinstance(r, dict) and _item(r) and not r.get(COMMI_KEY)]
    if not rows:
        return 0
    try:
        found = _read(session, sorted({_item(r) for r in rows}))
    except Exception as exc:
        logger.warning("comma material: commi not read (%s), prefix cut kept: %s", stage, exc)
        return 0
    n = 0
    for row in rows:
        got = found.get(_item(row))
        if got:
            row[COMMI_KEY], row[FROM_KEY] = got
            n += 1
    vlog("comma_material_attached", {"stage": stage, "rows": len(rows), "with_commi": n})
    return n


# --- which commi a text names -----------------------------------------------------------------------------------

_ORDINALS = {"primo": 1, "secondo": 2, "terzo": 3, "quarto": 4, "quinto": 5, "sesto": 6, "settimo": 7,
             "ottavo": 8, "nono": 9, "decimo": 10}
_K = r"(\d{1,2}|" + "|".join(_ORDINALS) + r")"


def _code_alternatives() -> str:
    from .defensive_generation import _CODE_SPELLINGS
    return "|".join(p for p, _ in _CODE_SPELLINGS)


def _k(raw: str) -> int:
    return _ORDINALS.get(raw.lower()) or int(raw)


def named_commi(text: Optional[str]) -> Dict[Tuple[str, str], Set[int]]:
    """{(article, code): {k}} for every comma the text names: "art. 1384, comma 2, c.c.", "art. 1384 c.c.,
    comma 2", "co. 2 dell'art. 1384 c.c.", "il secondo comma dell'art. 1384 c.c."."""
    from .defensive_generation import _article_num, _code_of
    codes = _code_alternatives()
    num = r"(\d+(?:\s*-?\s*[a-z]+)?)"
    patterns = [
        (re.compile(rf"\bart(?:t)?\.?\s*{num}\s*,?\s*(?:co\.|comma)\s*{_K}\s*,?\s*(?:del(?:la|lo)?\s+)?({codes})", re.I),
         (1, 2, 3)),
        (re.compile(rf"\bart(?:t)?\.?\s*{num}\s*,?\s*(?:del(?:la|lo)?\s+)?({codes})\s*,?\s*(?:co\.|comma)\s*{_K}\b", re.I),
         (1, 3, 2)),
        (re.compile(rf"\b(?:(?:co\.|comma)\s*{_K}|{_K}\s+comma)\s+(?:dell['’]\s*|del(?:l['’])?\s*)?art(?:icolo|\.)?\s*{num}"
                    rf"\s*,?\s*(?:del(?:la|lo)?\s+)?({codes})", re.I), None),
    ]
    out: Dict[Tuple[str, str], Set[int]] = {}
    for pattern, order in patterns:
        for m in pattern.finditer(text or ""):
            if order:
                number, k, code = (m.group(i) for i in order)
            else:
                k = m.group(1) or m.group(2)
                number, code = m.group(3), m.group(4)
            abbr = _code_of(code)
            if abbr and k:
                out.setdefault((_article_num(number), abbr), set()).add(_k(k))
    return out


# --- the commi given -----------------------------------------------------------------------------------------------

def _stems(text: str) -> set:
    from .defensive_generation import _STOPWORDS
    return {w[:6] for w in re.findall(r"[a-zà-ü]{4,}", (text or "").lower()) if w not in _STOPWORDS}


def select(commi: Sequence[str], chars: int, focus: str = "", named: Iterable[int] = ()) -> List[Tuple[int, str]]:
    """[(k, text)] in article order, within `chars`: the named commi first (cut to the budget if they alone
    exceed it), then by how many words they share with `focus` (ties: the earlier comma), each whole while it
    fits; a comma sharing nothing comes after every one that shares something. Never empty for an article with
    text: the best comma, cut, when none fits whole."""
    if not commi:
        return []
    chars = max(1, chars)
    words = _stems(focus)
    named_ks = [k for k in sorted(set(named)) if 1 <= k <= len(commi)]
    ranked = sorted(range(1, len(commi) + 1),
                    key=lambda k: (k not in named_ks, -len(words & _stems(commi[k - 1])), k))
    chosen: Dict[int, str] = {}
    used = 0
    for k in ranked:
        text = commi[k - 1]
        cost = len(text) + 9                     # its "(co. K) " label
        if used + cost <= chars:
            chosen[k] = text
            used += cost
        elif k in named_ks or not chosen:
            room = max(80, chars - used - 9)
            chosen[k] = article_text.clause_cut(text, room)
            used += room + 9
        if used >= chars:
            break
    return sorted(chosen.items())


def _keep(parts: Sequence[str], how: Optional[str], named: Iterable[int], chars: int) -> Set[int]:
    """The parts always given: the commi named (every part of each) and, for a split text, its opening when it leaves
    room for something else."""
    numbers, named = comma_numbers(parts, how), set(named)
    opening = how not in (None, "sections") and len(parts[0]) <= chars // 2
    return {i for i, n in enumerate(numbers, 1) if n in named} | ({1} if opening else set())


def trim(text: str, chars: int, focus: str = "", named: Iterable[int] = ()) -> str:
    """An article over its budget, never by prefix, flag or no flag: the opening (what a list hangs on), then the commi and
    list items sharing most words with the focus (select), in article order, a gap marked "[…]". No comma label: these
    parts may be list items, and a label would be a citation nobody wrote."""
    text = (text or "").strip()
    if len(text) <= chars:
        return text
    parts, how = article_parts(text)
    if len(parts) < 2:
        return article_text.clause_cut(text, chars)    # one unbroken sentence: nothing to choose among
    chosen = select(parts, chars, focus, _keep(parts, how, named, chars))
    out: List[str] = []
    last = 0
    for k, part in chosen:
        if k != last + 1 and not (out and out[-1].endswith("[…]")):
            out.append("[…]")
        out.append(part)
        last = k
    if last != len(parts) and not out[-1].endswith("[…]"):
        out.append("[…]")
    return " ".join(out)


def trim_row(text: str, chars: int, row: Dict[str, Any]) -> str:
    if len(text or "") <= SHORT_ARTICLE and not row.get("_comma_chars"):
        return (text or "").strip()
    return trim(text, chars, row.get(FOCUS_KEY) or "", row.get(NAMED_KEY) or ())


def passage(row: Dict[str, Any], chars: int) -> Optional[str]:
    """The article's material from its commi (select), or None when the row holds none (the caller keeps its
    prefix cut). Commi from Sections, lines or numbered paragraphs are labelled "(co. K)"; sentence parts are
    not numbered (they are not commi), the gaps marked "[…]"."""
    commi = row.get(COMMI_KEY)
    if not enabled() or not commi:
        return None
    how = row.get(FROM_KEY)
    numbers = comma_numbers(commi, how)
    whole = sum(len(c) + 9 for c in commi)
    if whole <= SHORT_ARTICLE and not row.get("_comma_chars"):      # not when the prompt itself had to shrink the row
        chars = max(chars, whole)
    chosen = select(commi, chars, row.get(FOCUS_KEY) or "", _keep(commi, how, row.get(NAMED_KEY) or (), chars))
    out, last = [], 0
    for k, t in chosen:
        n = numbers[k - 1]
        head = n is not None and (k == 1 or numbers[k - 2] != n)
        if head:
            out.append(f"(co. {n}) {_COMMA_HEAD_RE.sub('', t, count=1)}")
        else:
            if k != last + 1:
                out.append("[…]")
            out.append(t)
        last = k
    given = {numbers[k - 1] for k, _ in chosen}
    left = sorted({n for n in numbers if n is not None and n not in given})
    if left:
        out.append(f"[commi non riportati: {', '.join(map(str, left))}]")
    elif last != len(commi) and not any(n is not None for n in numbers):
        out.append("[…]")
    return " ".join(out)


def set_focus(rows: Iterable[Dict[str, Any]], focus_of: Dict[tuple, str], named: Dict[Tuple[str, str], Set[int]],
              keep: bool = False) -> None:
    """Each article row's focus and named commi (keep=True: only where none is set yet)."""
    for row in rows:
        ref = row.get("_article") if isinstance(row, dict) else None
        if not ref:
            continue
        ref = (str(ref[0]), ref[1])
        if focus_of.get(ref) and not (keep and row.get(FOCUS_KEY)):
            row[FOCUS_KEY] = focus_of[ref]
        if named.get(ref):
            row[NAMED_KEY] = sorted(set(row.get(NAMED_KEY) or ()) | named[ref])


def analysis_focus(rows: Iterable[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[tuple, str]:
    """Before the plan: each article's focus, the analysis question it was found for (its title and key words),
    else all the questions' titles and the facts searched."""
    questioni = [q for q in analysis.get("questioni") or [] if isinstance(q, dict)]
    def text_of(q):
        return " ".join([str(q.get("titolo") or "")] + [str(w) for w in q.get("parole_chiave") or []])
    every = " ".join([text_of(q) for q in questioni] + [str(x) for x in analysis.get("ricerca_fatti") or []])
    out: Dict[tuple, str] = {}
    for row in rows:
        ref = row.get("_article") if isinstance(row, dict) else None
        if not ref:
            continue
        i = row.get("_question")
        own = text_of(questioni[i]) if isinstance(i, int) and 0 <= i < len(questioni) else ""
        out.setdefault((str(ref[0]), ref[1]), own or every)
    return out


def plan_focus(plan: Dict[str, Any]) -> Dict[tuple, str]:
    """After the plan: each article's focus, the issue(s) that place it (title, argument, objection, reply)."""
    out: Dict[tuple, List[str]] = {}
    for q in plan.get("questioni") or []:
        text = " ".join(str(q.get(k) or "") for k in ("titolo", "argomento", "obiezione", "replica"))
        for ref in q.get("norme") or []:
            out.setdefault((str(ref[0]), ref[1]), []).append(text)
    return {ref: " ".join(texts) for ref, texts in out.items()}


def plan_text(plan: Dict[str, Any]) -> str:
    parts = [str(plan.get(k) or "") for k in ("obiettivo", "questione_centrale")]
    for q in plan.get("questioni") or []:
        parts += [str(q.get(k) or "") for k in ("titolo", "argomento", "obiezione", "replica")]
    parts += [str(a.get("risposta") or "") for a in plan.get("risposte") or []]
    parts += [str(x) for x in (plan.get("subordinate") or []) + (plan.get("punti_deboli") or [])]
    return "\n".join(parts)


def ensure(rows: List[Dict[str, Any]], named: Dict[Tuple[str, str], Set[int]], session=None,
           stage: str = "") -> List[str]:
    """Every named comma available to its article's row: a row without its commi, or holding fewer than the
    comma named, is read again (one query for all; the app's driver when no session is given). The refs read
    ("1384 c.c. co. 2"), logged as comma_material_fetched."""
    if not enabled() or not named:
        return []
    missing = []
    for row in rows:
        ref = row.get("_article") if isinstance(row, dict) else None
        if not ref or not _item(row):
            continue
        ks = named.get((str(ref[0]), ref[1])) or set()
        have = len(row.get(COMMI_KEY) or [])
        if ks and (not have or max(ks) > have):
            missing.append(row)
    if not missing:
        return []
    fetched: List[str] = []
    try:
        if session is None:
            import neo4j
            from .main import NEO4J_DATABASE, driver
            with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as own:
                found = _read(own, sorted({_item(r) for r in missing}))
        else:
            found = _read(session, sorted({_item(r) for r in missing}))
    except Exception as exc:
        logger.warning("comma material: named commi not read (%s): %s", stage, exc)
        return []
    for row in missing:
        got = found.get(_item(row))
        if got:
            row[COMMI_KEY], row[FROM_KEY] = got
            ref = row["_article"]
            ks = named.get((str(ref[0]), ref[1])) or set()
            fetched += [f"{ref[0]} {ref[1]} co. {k}" + ("" if k <= len(got[0]) else " (non esiste)") for k in sorted(ks)]
    vlog("comma_material_fetched", {"stage": stage, "fetched": fetched})
    return fetched


def for_plan(plan: Dict[str, Any], rows: List[Dict[str, Any]], case_text: str, session=None) -> Dict[str, Any]:
    """After the plan: the focus of each article its plan issue, the commi the plan or the case names marked
    and read when missing (ensure)."""
    if not enabled() or not plan:
        return {}
    named = named_commi(f"{case_text}\n{plan_text(plan)}")
    set_focus(rows, plan_focus(plan), named)
    fetched = ensure(rows, named, session, "plan")
    return {"named": {f"{n} {c}": sorted(ks) for (n, c), ks in named.items()}, "fetched": fetched}


# --- what was given -------------------------------------------------------------------------------------------------

_LABEL_RE = re.compile(r"^\[art\. ([^\]\s]+) ([^\]]+?)\](?: \[[^\]]*\])?\s*$", re.MULTILINE)
_CO_RE = re.compile(r"\(co\. (\d+)\)")


def given(material: str, rows: Iterable[Dict[str, Any]] = ()) -> List[Dict[str, Any]]:
    """Per "[art. N code]" block of a writer call's material: which commi it holds ("(co. K)" labels; "all" for
    a block built without commi, "parts" for a sentence split), of how many, and which were named."""
    by_ref = {}
    for row in rows:
        ref = row.get("_article") if isinstance(row, dict) else None
        if ref and row.get(COMMI_KEY):
            by_ref.setdefault(f"{ref[0]} {ref[1]}", row)
    marks = list(_LABEL_RE.finditer(material or ""))
    out = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(material)
        body = material[m.end():end]
        ref = f"{m.group(1)} {m.group(2).strip()}"
        row = by_ref.get(ref) or {}
        ks = sorted({int(k) for k in _CO_RE.findall(body)})
        out.append({"ref": ref, "total": len(row.get(COMMI_KEY) or []) or None, "from": row.get(FROM_KEY),
                    "given": ks or ("parts" if row.get(FROM_KEY) == "sentences" else "prefix"),
                    "named": row.get(NAMED_KEY) or [], "chars": len(body.strip())})
    return out


def log_material(stage: str, material: str, rows: Iterable[Dict[str, Any]] = ()) -> None:
    """vlog comma_material: per article of one writer call's material, the commi given (with the flag)."""
    if not enabled() or not material:
        return
    articles = given(material, rows)
    if articles:
        vlog("comma_material", {"stage": stage, "articles": articles})
