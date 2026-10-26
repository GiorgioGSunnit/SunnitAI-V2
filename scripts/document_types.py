"""Documents with no document_type (2 on prod, 26 Oct): every search that filters by type ("primary" for the
codes, "interpretation" for rulings, "special" for books, "ccnl") skips them. Lists each with its name, its
Section count and a few Section names, and proposes a type from the name and the shape of its Sections.

Dry run by default -- nothing is written:
    cd /opt/chatbot && venv/bin/python scripts/document_types.py
Set the proposed types (only those with a proposal), or your own for a given id:
    cd /opt/chatbot && venv/bin/python scripts/document_types.py --write
    cd /opt/chatbot && venv/bin/python scripts/document_types.py --write --set "<doc id>=primary"
"""

import argparse
import logging
import os
import re
import sys
from typing import Dict, List, Optional

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"))
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

TYPES = ("primary", "interpretation", "special", "ccnl")
_NAME_RULES = [
    ("ccnl", r"\bccnl\b|contratto collettivo"),
    ("interpretation", r"\bcass(?:azione)?\b|\bsent(?:enza)?\.?\s*n|\bordinanza\b|\bcorte\b|\btar\b|consiglio di stato"
                       r"|\btribunale\b|massim"),
    ("special", r"manuale|lineamenti|commentario|trattato|compendio|formulari|\bguida\b|istituzioni di|diritto \w+ "
                r"(?:civile|penale|amministrativo)|modell"),
    ("primary", r"\bcodice\b|\bcostituzione\b|\blegge\b|\bl\.\s*\d|d\.\s*lgs|decreto|d\.p\.r|\bregolamento\b"
                r"|testo unico|\bt\.u\.|direttiva|\btuel\b|\btuf\b"),
]
_ARTICLE_NAME = re.compile(r"^\d+(?:-[a-z]+)?(?:\.\d+)*$", re.IGNORECASE)

LIST_QUERY = (
    "MATCH (d:Document) WHERE d.document_type IS NULL "
    "OPTIONAL MATCH (d)-[:CONTAINS]->(s:Section) "
    "WITH d, count(s) AS sections, collect(s.name)[..40] AS names "
    "RETURN d.id AS id, d.name AS name, sections, names ORDER BY d.name")
WRITE_QUERY = ("MATCH (d:Document {id: $id}) WHERE d.document_type IS NULL "
               "SET d.document_type = $type RETURN d.id AS id")


def propose(name: str, section_names: List[str]) -> tuple:
    """(type or None, why) from the document's name, else from its Sections looking like articles."""
    low = (name or "").lower()
    for kind, pattern in _NAME_RULES:
        if re.search(pattern, low):
            return kind, f"name matches {kind!r}"
    names = [n for n in section_names if n]
    if names and sum(1 for n in names if _ARTICLE_NAME.match(n)) >= 0.8 * len(names):
        return "primary", "most Sections are named like articles (N.0.0)"
    return None, "no rule matched: set it with --set"


def parse_overrides(values: List[str]) -> Dict[str, str]:
    out = {}
    for value in values or []:
        doc_id, _, kind = value.partition("=")
        if kind.strip() not in TYPES:
            sys.exit(f"--set {value!r}: the type must be one of {', '.join(TYPES)}")
        out[doc_id.strip()] = kind.strip()
    return out


def main(argv: Optional[List[str]] = None):
    from src.rag.main import NEO4J_DATABASE, driver

    parser = argparse.ArgumentParser(description="Documents with no document_type, and a proposed type.")
    parser.add_argument("--write", action="store_true", help="set the proposed (or --set) types")
    parser.add_argument("--set", action="append", default=[], metavar="ID=TYPE", help="your type for one id")
    args = parser.parse_args(argv)
    overrides = parse_overrides(args.set)

    with driver.session(database=NEO4J_DATABASE) as session:
        docs = session.run(LIST_QUERY).data()
        print(f"{len(docs)} documents with no document_type.\n")
        plan = []
        for d in docs:
            kind, why = propose(d["name"], d["names"])
            if d["id"] in overrides:
                kind, why = overrides[d["id"]], "set with --set"
            sample = ", ".join(n for n in d["names"][:8] if n)
            print(f"- {d['name']!r}\n    id: {d['id']}\n    sections: {d['sections']}  e.g. {sample}\n"
                  f"    proposed: {kind or '-'} ({why})")
            if kind:
                plan.append((d["id"], kind))
        if not args.write:
            print("\nDry run: nothing written. --write sets the proposed types.")
            return
        for doc_id, kind in plan:
            done = session.run(WRITE_QUERY, id=doc_id, type=kind).data()
            print(f"{'set' if done else 'NOT set (already typed?)'}: {doc_id} -> {kind}")


if __name__ == "__main__":
    main()
