"""Sections whose plain_text is over --chars characters (default 20,000), by document: read-only, for ingestion (v48).

v47: art. 229 Codice degli Appalti 2026 ("Entrata in vigore") and art. 306 Testo Unico sulla Sicurezza del Lavoro 2026
("Disposizioni transitorie") hold the annexes that follow them in their text -- one comma of 108,000 characters, every
section prompt of l3, n7 and h1 over the model's context. The final articles of a code are where the loader leaves what
comes after; this lists every such Section so the ingestion can split them. Nothing is written.

    cd /root/v38 && venv/bin/python scripts/oversized_sections.py --env .env.v45 [--chars 20000] [--exclude-type ccnl]
                                                                  [--out oversized_sections.md]

--exclude-type (repeatable, case-insensitive): leave out the documents of that document_type (ccnl: collective
agreements, whose long Sections are their own).

The report: per document (most characters first) its oversized Sections -- name, titolo_articolo, characters, and the
first words after the first 3,000 characters (where the text stops being the article's own, as a rule).
"""

import argparse
import os
import sys
from collections import defaultdict

QUERY = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE size(coalesce(s.plain_text, '')) > $chars "
         "AND NOT toLower(coalesce(d.document_type, '')) IN $exclude "
         "RETURN d.name AS document, d.document_type AS type, s.name AS section, s.titolo_articolo AS rubrica, "
         "size(s.plain_text) AS chars, substring(s.plain_text, 3000, 160) AS after "
         "ORDER BY document, chars DESC")


def report(rows, chars: int, excluded=()) -> list:
    by_doc = defaultdict(list)
    for r in rows:
        by_doc[(r["document"], r.get("type") or "")].append(r)
    order = sorted(by_doc.items(), key=lambda kv: -sum(x["chars"] for x in kv[1]))
    lines = [f"# Sections with plain_text over {chars:,} characters", "",
             f"{sum(len(v) for v in by_doc.values())} Sections in {len(by_doc)} documents. Read-only: nothing changed."
             + (f" Document types left out: {', '.join(excluded)}." if excluded else ""),
             "", "| document | type | sections | characters |", "|---|---|---:|---:|"]
    lines += [f"| {doc} | {kind or '-'} | {len(v)} | {sum(x['chars'] for x in v):,} |" for (doc, kind), v in order]
    for (doc, kind), v in order:
        lines += ["", f"## {doc}", "", "| section | rubrica | characters | text from character 3,000 |", "|---|---|---:|---|"]
        for x in v:
            after = " ".join(str(x.get("after") or "").split()).replace("|", "/")
            lines.append(f"| {x['section']} | {(x.get('rubrica') or '-').replace('|', '/')} | {x['chars']:,} | {after} |")
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    parser.add_argument("--chars", type=int, default=20000)
    parser.add_argument("--exclude-type", action="append", default=[], metavar="TYPE",
                        help="a document_type to leave out (repeatable), e.g. ccnl")
    parser.add_argument("--out", default="oversized_sections.md")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(args.env)
    uri = os.getenv("NEO4J_URI")
    if not uri:
        sys.exit(f"NEO4J_URI is not set ({args.env})")
    import neo4j
    with neo4j.GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as driver:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"),
                            default_access_mode=neo4j.READ_ACCESS) as session:
            excluded = sorted({t.strip().lower() for t in args.exclude_type if t.strip()})
            rows = session.run(QUERY, chars=args.chars, exclude=excluded).data()
    lines = report(rows, args.chars, excluded)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines[:8 + len({r['document'] for r in rows})]))
    print(f"File: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
