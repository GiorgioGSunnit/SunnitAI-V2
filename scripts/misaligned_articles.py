#!/usr/bin/env python3
"""v60e: the article nodes whose rubrica does not belong to their text (ingestion misalignment), per code, for
ingestion -- the same check the flow applies at load (src/rag/article_text.shifted: the rubrica's words, two or more
outside the generic ones, sharing no stem with the first comma nor with the whole text, confirmed by an adjacent
article: this rubrica fits its text, or its rubrica fits this text). Read only.

  python scripts/misaligned_articles.py                         # every primary and special document
  python scripts/misaligned_articles.py --codes c.c.,c.p.       # some codes (sigle) or document names
  python scripts/misaligned_articles.py --out diagnose_reports/misaligned
  python scripts/misaligned_articles.py --suspects             # also the unconfirmed stem-check failures

Writes <out>/misaligned.json ({code: [{article, id, rubrica, first_comma}]}) and <out>/misaligned.md (counts and lists).
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DOCS_QUERY = ("MATCH (d:Document) WHERE d.document_type IN ['primary', 'special'] "
              "RETURN d.id AS id, d.name AS name ORDER BY d.name")
SECTIONS_QUERY = ("MATCH (d:Document {id: $doc})-[:CONTAINS]->(s:Section) WHERE s.titolo_articolo IS NOT NULL "
                  "RETURN coalesce(s.id, elementId(s)) AS id, s.name AS name, s.titolo_articolo AS rubrica, "
                  "s.plain_text AS text")


def main(argv=None) -> None:
    import neo4j
    from src.rag import article_text, citations
    from src.rag.main import NEO4J_DATABASE, driver

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--codes", default="", help="sigle or document names, comma separated (default: all)")
    parser.add_argument("--out", default=str(ROOT / "diagnose_reports" / "misaligned"))
    parser.add_argument("--suspects", action="store_true",
                        help="also list the stem check's failures no neighbour confirms (suspects.json; never excluded)")
    args = parser.parse_args(argv)
    wanted = {citations.code_key(c) for c in args.codes.split(",") if c.strip()}
    found, totals, suspects = {}, {}, {}
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        for doc in session.run(DOCS_QUERY).data():
            code = citations.code_key(doc["name"])
            if wanted and code not in wanted:
                continue
            checked = 0
            articles = {}
            for row in session.run(SECTIONS_QUERY, doc=doc["id"]).data():
                number = article_text.article_number(row["name"] or "")
                if number and row["name"] in (number, f"{number}.0.0"):
                    articles[number] = row
            for number, row in articles.items():
                checked += 1
                neighbours = [(articles[n]["rubrica"], articles[n]["text"]) for n in article_text.neighbour_numbers(number)
                              if n in articles]
                hit = article_text.shifted(row["rubrica"], row["text"], neighbours)
                if hit:
                    bad = article_text.misaligned(row["rubrica"], row["text"])
                    found.setdefault(code, []).append({"article": number, "id": str(row["id"]), **bad, "neighbour": hit})
                elif args.suspects and article_text.misaligned(row["rubrica"], row["text"]):
                    suspects.setdefault(code, []).append({"article": number, "id": str(row["id"]),
                                                          **article_text.misaligned(row["rubrica"], row["text"])})
            totals[code] = totals.get(code, 0) + checked
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "misaligned.json").write_text(json.dumps(found, ensure_ascii=False, indent=1), encoding="utf-8")
    if args.suspects:
        (out / "suspects.json").write_text(json.dumps(suspects, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# Rubrica / text mismatches (article_text.misaligned)", "", "| code | articles checked | failing |",
             "|---|---|---|"]
    lines += [f"| {code} | {totals[code]} | {len(found.get(code, []))} |" for code in sorted(totals)]
    for code in sorted(found):
        lines += ["", f"## {code}", ""]
        lines += [f"- art. {x['article']} ({x['id']}): «{x['rubrica']}» — {x['first_comma']}" for x in found[code]]
    (out / "misaligned.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"{sum(len(v) for v in found.values())} failing of {sum(totals.values())} checked; {out}/misaligned.md")


if __name__ == "__main__":
    main()
