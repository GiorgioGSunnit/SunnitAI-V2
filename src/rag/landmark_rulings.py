"""v78: the lawyer's «pronunce di riferimento» per article -- the landmark rulings he names (SU Franzese 2002 on art. 40
c.p., Corte cost. 364/1988 on art. 5 c.p. …), which the corpus (Cassazione 2025-2026) cannot hold. Kept on the article's
Section node (PROPERTY, JSON [{"estremi", "principio"}]), imported from the file he edits (FILE: docs/lawyer/
pronunce_di_riferimento.md) by scripts/fill_procedural_store.py --pronunce, read with the article notes (attach),
printed in §5 under «Pronunce di riferimento indicate dal consulente» and given to the thesis call (rulings.py). Only
what he has written: nothing is ever added by the code.

The file: one heading per article, «## art. N codice»; under it one bullet per ruling, «- <estremi> — <principio>» (the
estremi as he writes them, an em dash, the principle in his words); HTML comments ignored. The runtime reads the store
first; an article the store has none for takes the file's (source «file», logged), as article_notes does.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROPERTY = "pronunce_di_riferimento"
FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "lawyer",
                    "pronunce_di_riferimento.md")
_HEADING_RE = re.compile(r"^##\s+art(?:icolo|\.)?\s*(\d+(?:-[a-z]+)?)\s+(.+?)\s*$", re.IGNORECASE)
_ITEM_RE = re.compile(r"^\s*[-•]\s*(.+?)\s+[—–]\s+(.+?)\s*$")
LOOKUP_QUERY = ("UNWIND $items AS it MATCH (d:Document {id: it.doc})-[:CONTAINS]->(s:Section) "
                "WHERE s.name = it.num OR s.name = it.num + '.0.0' "
                f"RETURN it.doc AS doc, it.num AS num, s.{PROPERTY} AS pronunce")
WRITE_QUERY = ("UNWIND $items AS it MATCH (d:Document {id: it.doc})-[:CONTAINS]->(s:Section) "
               "WHERE s.name = it.num OR s.name = it.num + '.0.0' "
               f"SET s.{PROPERTY} = it.note RETURN count(s) AS n")
_FILE_CACHE: Dict[str, Any] = {}


def _code(code: str) -> str:
    from .plan_then_write import _code_key
    return _code_key(" ".join(str(code or "").split()))


def parse(text: str) -> Dict[str, List[Dict[str, str]]]:
    """{label «N code»: [{"estremi", "principio"}]} of the file; a bullet without its em dash is skipped (logged)."""
    out: Dict[str, List[Dict[str, str]]] = {}
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.DOTALL)
    current: Optional[str] = None
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            current = f"{m.group(1).lower()} {_code(m.group(2))}"
            continue
        if line.startswith("#"):
            current = None
            continue
        if current is None or not line.strip():
            continue
        item = _ITEM_RE.match(line)
        if item:
            out.setdefault(current, []).append({"estremi": " ".join(item.group(1).split()),
                                                "principio": " ".join(item.group(2).split())})
        else:
            logger.warning("pronunce di riferimento: %s, line without «estremi — principio»: %r", current, line[:80])
    return out


def file_entries(path: str = FILE) -> Dict[str, List[Dict[str, str]]]:
    if path not in _FILE_CACHE:
        try:
            with open(path, encoding="utf-8") as f:
                _FILE_CACHE[path] = parse(f.read())
        except OSError:
            _FILE_CACHE[path] = {}
    return _FILE_CACHE[path]


def attach(session, rows: List[Dict[str, Any]]) -> int:
    """In place: each row the store holds pronunce for, its "_pronunce". The number attached."""
    from .topic_groups import unit_key
    items = sorted({unit_key(r) for r in rows if r.get("_article") and all(unit_key(r))})
    if not items or session is None:
        return 0
    stored = {}
    for rec in session.run(LOOKUP_QUERY, items=[{"doc": d, "num": n} for d, n in items]).data():
        try:
            value = json.loads(rec["pronunce"]) if rec.get("pronunce") else None
        except (TypeError, ValueError):
            value = None
        if isinstance(value, list) and value:
            stored[(rec["doc"], rec["num"])] = value
    n = 0
    for row in rows:
        value = stored.get(unit_key(row)) if row.get("_article") else None
        if value:
            row["_pronunce"] = value
            n += 1
    return n


def for_label(label: str, rows: Iterable[Dict[str, Any]] = (), path: str = FILE) -> Tuple[List[Dict[str, str]], str]:
    """(pronunce, source) of a record's label: the row's store entries, else the file's; ([], "") when neither."""
    m = re.match(r"^(\d+(?:-[a-z]+)?)\s+(.+)$", str(label or "").strip())
    if not m:
        return [], ""
    number, code = m.group(1).lower(), _code(m.group(2))
    for row in rows or []:
        ref = row.get("_article")
        if ref and str(ref[0]).lower() == number and _code(ref[1]) == code and row.get("_pronunce"):
            return [x for x in row["_pronunce"] if isinstance(x, dict) and x.get("estremi")], "store"
    entries = file_entries(path).get(f"{number} {code}") or []
    return (list(entries), "file") if entries else ([], "")


def for_records(labels: Iterable[str], rows: Iterable[Dict[str, Any]] = (), path: str = FILE
                ) -> Tuple[Dict[str, List[Dict[str, str]]], Dict[str, str]]:
    rows = list(rows or [])
    out, sources = {}, {}
    for label in labels:
        items, source = for_label(label, rows, path)
        if items:
            out[label], sources[label] = items, source
    return out, sources


def store_items(entries: Dict[str, List[Dict[str, str]]]) -> Dict[str, List[Tuple[str, str]]]:
    """{code: [(number, JSON)]} for the import."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    for label, items in entries.items():
        number, code = label.split(" ", 1)
        out.setdefault(code, []).append((number, json.dumps(items, ensure_ascii=False)))
    return out
