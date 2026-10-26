"""Read-only: per code, how many articles have an "N.0.0" Section holding only their date in force ("In vigore
dal 11 dicembre 2021"), and whether the shared article fetch (src/rag/article_text.py, used by the chat's
cross-references, issue spotting and the strategy) now returns the article's text for a sample of them.

Run from the app folder so it uses the app's code, .env and venv:
    cd /opt/chatbot && venv/bin/python scripts/check_article_texts.py              # 3 samples per code
    cd /opt/chatbot && venv/bin/python scripts/check_article_texts.py --sample 10 --types primary,special

Nothing is written to the database.
"""

import argparse
import logging
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"))

import neo4j  # noqa: E402

_open_session = neo4j.Driver.session


def _read_only_session(self, **config):
    config["default_access_mode"] = neo4j.READ_ACCESS
    return _open_session(self, **config)


neo4j.Driver.session = _read_only_session   # every session in this process is read-only
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

# Java regex (Neo4j's =~ matches the whole string): the same line as article_text.IN_FORCE_RE.
DATE_ONLY = r"(?is)\s*in vigore dal\s+\d{1,2}\s+\S+\s+\d{4}\.?\s*"
ARTICLE_ZERO = r"\d+(-[a-z]+)?\.0\.0"

COUNT_QUERY = (
    "MATCH (d:Document)-[:CONTAINS]->(s:Section) "
    "WHERE d.document_type IN $types AND s.name =~ $num_re "
    "AND s.plain_text =~ $date_only "
    "WITH d, collect(DISTINCT s.name) AS names "
    "RETURN d.id AS id, d.name AS name, d.document_type AS type, size(names) AS n, names[..$k] AS sample "
    "ORDER BY n DESC")


def main():
    from src.rag import article_text
    from src.rag.main import NEO4J_DATABASE, driver

    parser = argparse.ArgumentParser(description="Articles whose N.0.0 holds only the date in force, per code.")
    parser.add_argument("--sample", type=int, default=3, help="articles fetched per code (default 3)")
    parser.add_argument("--types", default="primary,special", help="document types to scan")
    args = parser.parse_args()
    types = [t.strip() for t in args.types.split(",") if t.strip()]

    failures = []
    with driver.session(database=NEO4J_DATABASE) as session:
        codes = session.run(COUNT_QUERY, types=types, date_only=DATE_ONLY, num_re=ARTICLE_ZERO, k=args.sample).data()
        total = sum(c["n"] for c in codes)
        print(f"{len(codes)} documents, {total} articles whose N.0.0 is only its date in force.\n")
        print(f"{'articles':>8}  {'type':10}  document")
        for c in codes:
            print(f"{c['n']:>8}  {c['type'] or '':10}  {c['name']}")
        print("\nSample fetched with article_text.fetch_article:")
        for c in codes:
            print(f"\n## {c['name']}")
            for name in c["sample"]:
                number = article_text.article_number(name)
                rows = article_text.fetch_article(session, c["id"], number)
                text = ((rows[0].get("s") or {}).get("plain_text") or "") if rows else ""
                ok = bool(text.strip()) and not article_text.in_force_only(text)
                if not ok:
                    failures.append(f"{c['name']} art. {number}")
                shown = " ".join(text.split())[:90]
                print(f"  art. {number:18s} {'TEXT ' if ok else 'DATE ONLY'} {len(text):>6} chars  {shown}")
    print("\nStill date only after the fetch: " + (", ".join(failures) or "none"))
    print("Done. Nothing was written.")


if __name__ == "__main__":
    main()
