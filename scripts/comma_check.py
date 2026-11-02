"""Read-only: the comma split on the corpus, before and after join of the PDF's wrapped lines. For the articles of c.c. and
c.p.c. most cited by the rulings (CITES): commi as PDF lines (before) and as unwrap reads them (after), and every comma under
100 characters that does not end in . ; : (likely a wrapped line still). From the app folder, with its venv and .env:

    venv/bin/python scripts/comma_check.py --top 300 --out comma_check.md
"""

import argparse
import os
import re
import sys

CODES = {"c.c.": "Codice Civile", "c.p.c.": "Codice di procedura civile"}
SHORT = 100
END_RE = re.compile(r"[.;:][»”\"'’)\]]*$")


def top_cited(session, top: int) -> list:
    """[(number, code, times)], the most cited first."""
    rows = session.run("MATCH ()-[r:CITES]->() WHERE r.code IN $codes AND r.article IS NOT NULL "
                       "RETURN r.article AS article, r.code AS code, count(*) AS n ORDER BY n DESC LIMIT $top",
                       codes=list(CODES), top=top).data()
    return [(str(r["article"]), r["code"], r["n"]) for r in rows]


def lines_of(text: str) -> list:
    """What the split did before: one comma per PDF line."""
    return [p.strip() for p in re.split(r"\n+", text or "") if p.strip()]


def suspects(commi: list) -> list:
    """The commi under SHORT characters that end no clause. The last comma of a repealed article is no suspect."""
    return [c for c in commi if len(c) < SHORT and not END_RE.search(c) and "abrogat" not in c.lower()]


def line(number: str, code: str, times: int, before: list, after: list) -> str:
    flags = suspects(after)
    return (f"| art. {number} {code} | {times} | {len(before)} | {len(after)} | "
            + ("; ".join(c.replace("|", "/") for c in flags) or "") + " |")


def report(session, top: int, comma, dg) -> list:
    out = ["# Comma split on the corpus", "", "| Article | Cited | Before (lines) | After | Under 100 chars, no end |", "|---|---|---|---|---|"]
    changed, flagged, missing = 0, 0, []
    for number, code, times in top_cited(session, top):
        rows = dg._fetch_articles(session, [(number, code)])
        text = ((rows[0].get("s") or {}).get("plain_text") or "") if rows else ""
        if not text:
            missing.append(f"{number} {code}")
            continue
        before, after = lines_of(text), comma.split_commi(text)[0] if len(comma.unwrap(text)) > 1 else comma.unwrap(text)
        changed += len(before) != len(after)
        flagged += bool(suspects(after))
        out.append(line(number, code, times, before, after))
    return out + ["", f"Changed: {changed}. With a suspect comma: {flagged}. Without text: {', '.join(missing) or 'none'}."]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--top", type=int, default=300)
    parser.add_argument("--env", default=os.path.join(os.getcwd(), ".env"))
    parser.add_argument("--out", default="comma_check.md")
    args = parser.parse_args()
    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(args.env)
    uri = os.getenv("NEO4J_URI")
    if not uri:
        sys.exit(f"NEO4J_URI is not set ({args.env})")
    import neo4j
    from src.rag import comma_material as comma, defensive_generation as dg
    with neo4j.GraphDatabase.driver(uri, auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD"))) as driver:
        with driver.session(database=os.getenv("NEO4J_DATABASE", "neo4j"), default_access_mode=neo4j.READ_ACCESS) as session:
            lines = report(session, args.top, comma, dg)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print(f"File: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
