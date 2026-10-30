"""COMMA_MATERIAL_ENABLED (default false): the writer gets an article's commi, not its first N characters.

scripts/fetch_audit.py on run 0812 (no judge): the largest measurable defect was granularity -- sentences relying
on a comma the writer never received, because an article reached it as a prefix cut (_source_block's
`text[:chars]`, _with_text_cut's) or not at all. With the flag:

  1. attach(): after retrieval, each article row gets its commi ("_commi"), in one read for all of them: its
     "N.k.0" Sections in order (article_text's convention), else its text split (split_commi: lines, numbered
     paragraphs, else sentences). Within the per-article budget, select() keeps the commi whose words best
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

from . import feature_flags
from .verbose_logger import vlog

logger = logging.getLogger(__name__)

FLAG = "COMMA_MATERIAL_ENABLED"
COMMI_KEY, FROM_KEY, FOCUS_KEY, NAMED_KEY = "_commi", "_commi_from", "_focus", "_named_commi"
# A text split part shorter than this joins the one before it (a list's "a)" line, a stray footer).
MIN_PART = 40


def enabled() -> bool:
    return feature_flags.enabled(FLAG)


# --- the commi of an article -----------------------------------------------------------------------------------

_COMMA_NAME_RE = re.compile(r"^\.(\d+)\.0$")
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


def split_commi(text: Optional[str]) -> Tuple[List[str], str]:
    """(the parts of an article's text, how): its lines when it has several, else its numbered paragraphs
    ("1. ... 2. ..."), else its sentences ("sentences": not real commi, never labelled with a number)."""
    text = (text or "").strip()
    if not text:
        return [], "none"
    lines = _merge_short([p.strip() for p in re.split(r"\n+", text) if p.strip()])
    if len(lines) > 1:
        return lines, "lines"
    flat = " ".join(text.split())
    numbered = _merge_short([p.strip() for p in _NUMBERED_RE.split(flat) if p.strip()])
    if len(numbered) > 1:
        return numbered, "numbered"
    return _merge_short([p.strip() for p in _SENTENCE_RE.split(flat) if p.strip()]) or [flat], "sentences"


def commi_from_sections(number: str, texts: Dict[str, str]) -> Tuple[List[str], str]:
    """(the commi, how) of article `number` from its Sections {name: text}: the "N.k.0" ones (k > 0) in order,
    else article_text's chosen text split (split_commi)."""
    from . import article_text
    found = {}
    for name, text in texts.items():
        m = _COMMA_NAME_RE.match(name[len(number):]) if name.startswith(number + ".") else None
        if m and int(m.group(1)) > 0 and (text or "").strip() and not article_text.in_force_only(text):
            found[int(m.group(1))] = " ".join(text.split())
    if found:
        return [found[k] for k in sorted(found)], "sections"
    return split_commi(article_text.choose_text(number, texts))


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
        commi, how = commi_from_sections(spelling, {(r.get("s") or {}).get("name") or "": (r.get("s") or {})
                                                    .get("plain_text") or "" for r in rows})
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
            chosen[k] = text[:room].rstrip() + " […]"
            used += room + 9
        if used >= chars:
            break
    return sorted(chosen.items())


def passage(row: Dict[str, Any], chars: int) -> Optional[str]:
    """The article's material from its commi (select), or None when the row holds none (the caller keeps its
    prefix cut). Commi from Sections, lines or numbered paragraphs are labelled "(co. K)"; sentence parts are
    not numbered (they are not commi), the gaps marked "[…]"."""
    commi = row.get(COMMI_KEY)
    if not enabled() or not commi:
        return None
    chosen = select(commi, chars, row.get(FOCUS_KEY) or "", row.get(NAMED_KEY) or ())
    numbered = row.get(FROM_KEY) != "sentences"
    if numbered:
        text = " ".join(f"(co. {k}) {t}" for k, t in chosen)
        left = [k for k in range(1, len(commi) + 1) if k not in dict(chosen)]
        return text + (f" [commi non riportati: {', '.join(map(str, left))}]" if left else "")
    out, last = [], 0
    for k, t in chosen:
        if k != last + 1:
            out.append("[…]")
        out.append(t)
        last = k
    if last != len(commi):
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
