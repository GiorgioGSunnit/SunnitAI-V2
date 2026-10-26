"""One article's text, read the same way by every path: the chat's cross-references
(nodes/retrieval.py), issue spotting, the strategy's code lookups (defensive_generation.
_fetch_articles) and the strategy's rows from searches (fill_texts).

Many codes store an amended article three ways (26 Oct): "N.0.0" holds only the date its
wording took effect ("In vigore dal 11 dicembre 2021"), the bare "N" holds the whole article
opened by that date, and "N.1.0", "N.2.0"... hold its commi. 205 articles of the Codice del
Consumo, 224 of the Codice degli Appalti, 138 of the c.p.p. Any lookup preferring "N.0.0"
read the date alone. The rule here, for article N:
  1. "N.0.0", if it holds real text (its date line, if any, stripped);
  2. else the bare "N", its date line stripped;
  3. else the commi "N.k.0", in order, joined.
The row returned keeps the canonical Section ("N.0.0", else "N") with that text in plain_text.
An article with neither canonical Section keeps its dot fragments ("N.1", "N.2"), the first 3.
"""

import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

IN_FORCE_RE = re.compile(r"\s*in vigore dal\s+\d{1,2}\s+\w+\s+\d{4}\.?\s*", re.IGNORECASE)
_COMMA_RE = re.compile(r"^\.(\d+)\.0$")
_NUMBER_RE = re.compile(r"^(\d+(?:-[a-z]+)?)(?:\.|$)", re.IGNORECASE)


def in_force_only(text: Optional[str]) -> bool:
    match = IN_FORCE_RE.match(text or "")
    return bool(match) and match.end() == len(text or "")


def strip_in_force(text: Optional[str]) -> str:
    return IN_FORCE_RE.sub("", text or "", count=1).strip() if IN_FORCE_RE.match(text or "") else (text or "").strip()


def section_quality(section: Optional[Dict[str, Any]]) -> tuple:
    """Sort key among exact duplicates sharing a name: titolo_articolo set (only on a document's current,
    re-ingested copy) beats plain_text length alone (round 3, 18 Oct: length alone won art. 628 for a stale
    duplicate)."""
    return (1 if ((section or {}).get("titolo_articolo") or "").strip() else 0,
            len((section or {}).get("plain_text") or ""))


def article_number(section_name: str) -> Optional[str]:
    """"135-octiesdecies.0.0" -> "135-octiesdecies"; None for names that are not an article's."""
    m = _NUMBER_RE.match((section_name or "").strip())
    return m.group(1) if m else None


def choose_text(number: str, texts: Dict[str, str]) -> str:
    """The article's text from its Sections' {name: plain_text}, by the rule in the module's docstring;
    "" when none holds more than a date."""
    zero = (texts.get(f"{number}.0.0") or "").strip()
    if zero and not in_force_only(zero):
        return strip_in_force(zero)
    bare = strip_in_force(texts.get(number))
    if bare:
        return bare
    commi = {}
    for name, text in texts.items():
        m = _COMMA_RE.match(name[len(number):]) if name.startswith(number + ".") else None
        if m and int(m.group(1)) > 0 and (text or "").strip() and not in_force_only(text):
            commi[int(m.group(1))] = text.strip()
    return "\n".join(commi[k] for k in sorted(commi))


def _best_by_name(rows: Iterable[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    best: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        name = (row.get("s") or {}).get("name") or ""
        if name and (name not in best or section_quality(row.get("s")) > section_quality(best[name].get("s"))):
            best[name] = row
    return best


def article_rows(number: str, rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """From one document's {"d", "s"} rows for article N (its canonical Sections and its "N." fragments): the
    canonical row carrying the chosen text, else its first 3 dot fragments in name order, else []."""
    best = _best_by_name(rows)
    canonical = best.get(f"{number}.0.0") or best.get(number)
    if canonical:
        text = choose_text(number, {name: (row.get("s") or {}).get("plain_text") or "" for name, row in best.items()})
        section = dict(canonical.get("s") or {})
        if text:
            section["plain_text"] = text
        return [{**canonical, "s": section}]
    fragments = sorted((name for name in best if name.startswith(number + ".")))
    return [best[name] for name in fragments[:3]]


_SECTIONS_QUERY = (
    "UNWIND $items AS it "
    "MATCH (d:Document {id: it.doc})-[:CONTAINS]->(s:Section) "
    "WHERE (s.name = it.num OR s.name STARTS WITH it.num + '.') {visibility}"
    "RETURN it.doc AS doc, it.num AS num, d, s")


def _sections(session, items: List[Tuple[str, str]], visibility: str = "", **params) -> Dict[tuple, List[Dict]]:
    """{(doc_id, number): [{"d", "s"}...]} for every item, in one query."""
    if not items:
        return {}
    query = _SECTIONS_QUERY.replace("{visibility}", f"AND {visibility} " if visibility else "")
    found = session.run(query, items=[{"doc": doc, "num": num} for doc, num in items], **params).data()
    grouped: Dict[tuple, List[Dict]] = {}
    for row in found:
        grouped.setdefault((row["doc"], row["num"]), []).append({"d": row["d"], "s": row["s"]})
    return grouped


def fetch_article(session, doc_id: str, number: str, visibility: str = "", **params) -> List[Dict[str, Any]]:
    """Article N of one document: [the canonical row with its text] (see article_rows), never another article
    that merely starts with the same digits. Any spelling of the suffix: the canonical one first ("21-novies"
    finds "21-nonies.0.0"), then the legacy spellings of Sections ingested before the BE normalised them."""
    from .article_suffix import number_variants

    for spelling in number_variants(number):
        rows = article_rows(spelling, _sections(session, [(doc_id, spelling)], visibility, **params)
                            .get((doc_id, spelling), []))
        if rows:
            return rows
    return []


def fill_texts(session, rows: List[Dict[str, Any]]) -> List[str]:
    """In place, for rows other paths found (searches, votes, tables): a row of an article's canonical Section
    whose text is only its date in force, or opens with it, gets the article's text by the same rule. Returns
    the sections changed, "doc_name name". One query for all of them."""
    stubs = []
    for row in rows:
        s, d = row.get("s") or {}, row.get("d") or {}
        number = article_number(s.get("name") or "")
        if number and d.get("id") and s.get("name") in (number, f"{number}.0.0") \
                and IN_FORCE_RE.match(s.get("plain_text") or ""):
            stubs.append((row, d["id"], number))
    if not stubs:
        return []
    need = sorted({(doc, num) for row, doc, num in stubs if in_force_only(row["s"].get("plain_text"))})
    grouped = _sections(session, need) if need else {}
    filled = []
    for row, doc, num in stubs:
        if in_force_only(row["s"].get("plain_text")):
            best = _best_by_name(grouped.get((doc, num), []))
            text = choose_text(num, {name: (r.get("s") or {}).get("plain_text") or "" for name, r in best.items()})
        else:
            text = strip_in_force(row["s"].get("plain_text"))
        if text and text != row["s"].get("plain_text"):
            row["s"] = {**row["s"], "plain_text": text}
            filled.append(f"{row['d'].get('name', '')} {row['s']['name']}")
    return filled


def fill_from_rows(number: str, rows: List[Dict[str, Any]]) -> List[str]:
    """In place, without a query, for a path that already read article N's Sections (the chat's article_router):
    a canonical row whose text is only its date in force, or opens with it, gets the text the same rule
    chooses from the rows of its own document. Returns the sections changed."""
    by_doc: Dict[Any, List[Dict[str, Any]]] = {}
    for row in rows:
        by_doc.setdefault((row.get("d") or {}).get("id"), []).append(row)
    changed = []
    for doc_rows in by_doc.values():
        best = _best_by_name(doc_rows)
        texts = {name: (r.get("s") or {}).get("plain_text") or "" for name, r in best.items()}
        for row in doc_rows:
            s = row.get("s") or {}
            if s.get("name") not in (number, f"{number}.0.0") or not IN_FORCE_RE.match(s.get("plain_text") or ""):
                continue
            text = (choose_text(number, texts) if in_force_only(s.get("plain_text"))
                    else strip_in_force(s.get("plain_text")))
            if text and text != s.get("plain_text"):
                row["s"] = {**s, "plain_text": text}
                changed.append(f"{(row.get('d') or {}).get('name', '')} {s.get('name')}")
    return changed
