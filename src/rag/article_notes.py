"""v72: the lawyer's «note di inquadramento» per article -- what to keep in mind when an article is applied (the dolo of
the preterintenzione is on the base acts, the concause of art. 41 …), kept on the article's Section node
(PROPERTY, JSON {"nota", "numeri": {numero: nota}}), imported from the file he edits (FILE: docs/lawyer/
note_inquadramento.md) by scripts/fill_procedural_store.py --notes, read with the procedural facts (attach), carried
into the records (for_label) and shown to the writer and to the thesis call (lawyer_method).

The file: one heading per note, «## art. N codice» (the article) or «## art. N n. M codice» (one numero of it, as
art. 62 n. 2 c.p.); the lines under it, up to the next heading, are the note (a «- » bullet's mark dropped; HTML
comments ignored). The runtime reads the store first; an article the store has no note for takes the file's (source
«file», logged) -- so a note works before its import, and for an article the answer did not retrieve.
"""

import json
import logging
import os
import re
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

PROPERTY = "note_di_inquadramento"
FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "docs", "lawyer",
                    "note_inquadramento.md")
_HEADING_RE = re.compile(r"^##\s+art(?:icolo|\.)?\s*(\d+(?:-[a-z]+)?)(?:\s*,?\s*(?:n\.|numero)\s*(\d+))?\s+(.+?)\s*$",
                         re.IGNORECASE)
LOOKUP_QUERY = ("UNWIND $items AS it MATCH (d:Document {id: it.doc})-[:CONTAINS]->(s:Section) "
                "WHERE s.name = it.num OR s.name = it.num + '.0.0' "
                f"RETURN it.doc AS doc, it.num AS num, s.{PROPERTY} AS note")
WRITE_QUERY = ("UNWIND $items AS it MATCH (d:Document {id: it.doc})-[:CONTAINS]->(s:Section) "
               "WHERE s.name = it.num OR s.name = it.num + '.0.0' "
               f"SET s.{PROPERTY} = it.note RETURN count(s) AS n")
_FILE_CACHE: Dict[str, Any] = {}


def _code(code: str) -> str:
    from .plan_then_write import _code_key
    return _code_key(" ".join(str(code or "").split()))


def parse(text: str) -> Dict[str, Dict[str, Any]]:
    """{label «N code»: {"nota": text, "numeri": {M: text}}} of the file's headings."""
    out: Dict[str, Dict[str, Any]] = {}
    text = re.sub(r"<!--.*?-->", "", text or "", flags=re.DOTALL)
    current: Optional[Tuple[str, str]] = None
    lines: List[str] = []

    def close():
        if current is None:
            return
        body = " ".join(" ".join(re.sub(r"^\s*[-•]\s*", "", x).split()) for x in lines if x.strip())
        if not body:
            return
        label, numero = current
        entry = out.setdefault(label, {"nota": "", "numeri": {}})
        if numero:
            entry["numeri"][numero] = body
        else:
            entry["nota"] = body
    for line in text.splitlines():
        m = _HEADING_RE.match(line.strip())
        if m:
            close()
            current, lines = (f"{m.group(1).lower()} {_code(m.group(3))}", m.group(2) or ""), []
        elif line.startswith("#"):
            close()
            current, lines = None, []
        elif current is not None:
            lines.append(line)
    close()
    return out


def file_notes(path: str = FILE) -> Dict[str, Dict[str, Any]]:
    """The file's notes (read once per path; {} when missing)."""
    if path not in _FILE_CACHE:
        try:
            with open(path, encoding="utf-8") as f:
                _FILE_CACHE[path] = parse(f.read())
        except OSError:
            _FILE_CACHE[path] = {}
    return _FILE_CACHE[path]


def lookup(session, rows: Iterable[Dict[str, Any]]) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """{(doc id, number): note} the store holds."""
    from .topic_groups import unit_key
    items = sorted({unit_key(r) for r in rows if r.get("_article") and all(unit_key(r))})
    if not items or session is None:
        return {}
    out = {}
    for rec in session.run(LOOKUP_QUERY, items=[{"doc": d, "num": n} for d, n in items]).data():
        try:
            note = json.loads(rec["note"]) if rec.get("note") else None
        except (TypeError, ValueError):
            note = None
        if isinstance(note, dict):
            out[(rec["doc"], rec["num"])] = note
    return out


def attach(session, rows: List[Dict[str, Any]]) -> int:
    """In place: each row the store holds a note for, its "_note". The number attached."""
    from .topic_groups import unit_key
    stored = lookup(session, rows)
    n = 0
    for row in rows:
        note = stored.get(unit_key(row)) if row.get("_article") else None
        if note:
            row["_note"] = {**note, "source": "store"}
            n += 1
    return n


def for_label(label: str, rows: Iterable[Dict[str, Any]] = (), path: str = FILE) -> Tuple[str, str]:
    """(note, source) of a record's label -- «584 c.p.», or a circumstance's «62 n. 2 c.p.» (its article's numero
    note, after the article's own): the row's store note, else the file's; ("", "") when neither has one."""
    m = re.match(r"^(\d+(?:-[a-z]+)?)(?:\s+co\.\s*\d+)?(?:\s+n\.\s*(\d+))?\s+(.+)$", str(label or "").strip())
    if not m:
        return "", ""
    number, numero, code = m.group(1).lower(), m.group(2) or "", _code(m.group(3))

    def pick(entry: Dict[str, Any]) -> str:
        parts = [str(entry.get("nota") or "")]
        if numero:
            parts.append(str((entry.get("numeri") or {}).get(numero) or ""))
        return " ".join(p for p in parts if p).strip()
    for row in rows or []:
        ref = row.get("_article")
        if ref and str(ref[0]).lower() == number and _code(ref[1]) == code and row.get("_note"):
            text = pick(row["_note"])
            if text:
                return text, "store"
    entry = file_notes(path).get(f"{number} {code}")
    text = pick(entry) if entry else ""
    return (text, "file") if text else ("", "")


def store_items(notes: Dict[str, Dict[str, Any]]) -> Dict[str, List[Tuple[str, str]]]:
    """{code: [(number, note JSON)]} for the import."""
    out: Dict[str, List[Tuple[str, str]]] = {}
    for label, entry in notes.items():
        number, code = label.split(" ", 1)
        out.setdefault(code, []).append((number, json.dumps(entry, ensure_ascii=False)))
    return out
