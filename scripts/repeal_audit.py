#!/usr/bin/env python3
"""v79.6 (item 3): the repealed articles of c.p., c.c., c.p.c. and c.p.p. as the database holds them -- READ-ONLY, to be
run and reported before the repeal rule is judged. Per article node of the four codes:

  flagged      a property marks it repealed (any of STATUS_PROPERTIES true, or a status / stato property naming
               «abrogato» / «soppresso»)
  text_only    its stored text is a repeal reference alone -- «(abrogato)», «abrogato dall'art. 3 della L. 15/1968»,
               «soppresso», «Articolo abrogato da …» (article_text.repeal_kind == "reference")
  head_note    its text opens with a repeal note and goes on («[Abrogato dall'art. 12 …] Il testo previgente …»)
  live         none of these

The point of the measurement: the count of text_only and head_note articles that no property flags (the data lacks the
flag) and the count of flagged articles whose text still reads as live (the flag alone decides). Every article the
text marks and no property flags is listed for the ingestion side (--out: repeal_audit.json, repeal_audit.md).

    cd /opt/chatbot && venv/bin/python scripts/repeal_audit.py --env .env --out /root/repeal_audit
    python scripts/repeal_audit.py --selftest                        # no database
"""

import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

CODES = ("c.p.", "c.c.", "c.p.c.", "c.p.p.")
DOCS_QUERY = "MATCH (d:Document) WHERE d.document_type IN ['primary', 'special'] RETURN d.id AS id, d.name AS name"
SECTIONS_QUERY = ("MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) WHERE s.titolo_articolo IS NOT NULL "
                  "RETURN coalesce(s.id, elementId(s)) AS id, s.name AS name, s.titolo_articolo AS rubrica, "
                  "s.plain_text AS text, properties(s) AS props")


def classify(row: Dict[str, Any]) -> Dict[str, Any]:
    from src.rag import article_text
    props = row.get("props") or {}
    flagged = article_text.status_repealed(props)
    kind = article_text.repeal_kind(row.get("text") or "") or (
        "reference" if article_text.repeal_kind(row.get("rubrica") or "") and not (row.get("text") or "").strip() else "")
    return {"flagged": flagged, "kind": kind}


def summarise(rows_by_code: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    from src.rag import article_text
    out: Dict[str, Any] = {"codes": {}, "unflagged": {}}
    for code, rows in rows_by_code.items():
        c = Counter()
        for row in rows:
            number = article_text.article_number(row.get("name") or "")
            if not number or row.get("name") not in (number, f"{number}.0.0"):
                continue
            got = classify(row)
            c["articles"] += 1
            c["flagged"] += got["flagged"]
            c["text_only"] += got["kind"] == "reference"
            c["head_note"] += got["kind"] == "head_note"
            if got["kind"] and not got["flagged"]:
                c["text_unflagged"] += 1
                out["unflagged"].setdefault(code, []).append(
                    {"article": number, "id": str(row.get("id")), "kind": got["kind"],
                     "rubrica": " ".join(str(row.get("rubrica") or "").split())[:120],
                     "text": " ".join(str(row.get("text") or "").split())[:200]})
            if got["flagged"] and not got["kind"]:
                c["flag_only"] += 1
        out["codes"][code] = dict(c)
    return out


def report(s: Dict[str, Any]) -> str:
    lines = ["# Repealed articles (c.p., c.c., c.p.c., c.p.p.)", "",
             "| code | articles | flagged by a property | text: repeal reference only | text: repeal note at its head |"
             " text marks it, no property | property only |", "|---|---|---|---|---|---|---|"]
    for code in CODES:
        c = s["codes"].get(code) or {}
        lines.append(f"| {code} | {c.get('articles', 0)} | {c.get('flagged', 0)} | {c.get('text_only', 0)} | "
                     f"{c.get('head_note', 0)} | {c.get('text_unflagged', 0)} | {c.get('flag_only', 0)} |")
    for code, items in s["unflagged"].items():
        lines += ["", f"## {code}: the text marks it repealed, no property does ({len(items)})", ""]
        lines += [f"- art. {x['article']} ({x['kind']}): «{x['rubrica']}» — {x['text']}" for x in items[:400]]
    return "\n".join(lines) + "\n"


def selftest() -> None:
    rows = {"c.p.": [{"name": "1", "rubrica": "Reati e pene", "text": "Nessuno può essere punito per un fatto …",
                      "props": {}},
                     {"name": "2", "rubrica": "Ingiuria", "text": "Abrogato dall'art. 1 del D.Lgs. 15 gennaio 2016, n. 7.",
                      "props": {}},
                     {"name": "3", "rubrica": "Duello", "text": "(abrogato)", "props": {"abrogato": True}},
                     {"name": "4", "rubrica": "Vecchio", "text": "[Abrogato dall'art. 3 L. 1/1999] Il testo previgente "
                                                                 "puniva chi …", "props": {}},
                     {"name": "5", "rubrica": "Soppresso", "text": "Soppresso.", "props": {"status": "soppresso"}},
                     {"name": "6", "rubrica": "Vivo", "text": "Chiunque cagiona …", "props": {"stato": "abrogato"}}]}
    s = summarise(rows)
    c = s["codes"]["c.p."]
    assert c == {"articles": 6, "flagged": 3, "text_only": 3, "head_note": 1, "text_unflagged": 2, "flag_only": 1}, c
    assert [x["article"] for x in s["unflagged"]["c.p."]] == ["2", "4"]
    assert "| c.p. | 6 | 3 | 3 | 1 | 2 | 1 |" in report(s)
    print("selftest OK")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--env", default=".env")
    ap.add_argument("--out")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        selftest()
        return 0
    if os.path.isfile(args.env):
        from dotenv import load_dotenv
        load_dotenv(args.env)
    import neo4j
    from src.rag import citations
    from src.rag.main import NEO4J_DATABASE, driver
    rows_by_code: Dict[str, List[Dict[str, Any]]] = {}
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        for doc in session.run(DOCS_QUERY).data():
            code = citations.code_key(doc["name"])
            if code in CODES:
                rows_by_code.setdefault(code, []).extend(session.run(SECTIONS_QUERY, doc=doc["id"]).data())
    s = summarise(rows_by_code)
    text = report(s)
    print(text)
    if args.out:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        (out / "repeal_audit.json").write_text(json.dumps(s, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (out / "repeal_audit.md").write_text(text, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
