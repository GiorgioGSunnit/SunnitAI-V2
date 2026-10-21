"""Prints METHOD_CANDIDATES_ENABLED's lists (src/rag/method_candidates.py) as a
markdown table, for the lawyer to validate.

    venv/bin/python scripts/method_candidates_table.py           # the table only, no DB
    venv/bin/python scripts/method_candidates_table.py --check   # one row per article, with its
                                                                 # heading in the corpus (read-only)
"""

import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.rag import method_candidates as mc  # noqa: E402


def table() -> str:
    lines = ["| Tipo | Chiave | Ambito | Articoli candidati |", "|---|---|---|---|"]
    lines += [f"| {kind} | `{key}` | {label} | {articles} |" for kind, key, label, articles in mc.table_rows()]
    return "\n".join(lines)


def checked_table() -> str:
    """Each article with the heading of what _fetch_any finds for it, or NOT FOUND."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    import neo4j
    import src.rag.defensive_generation as dg
    from src.rag.doc_lookup import _fetch_doc_names
    from src.rag.main import NEO4J_DATABASE, driver

    documents = _fetch_doc_names(driver, NEO4J_DATABASE)
    lines = ["| Lista | Articolo | Rubrica nel corpus |", "|---|---|---|"]
    lists = [(f"area `{k}`", refs) for k, (_, refs) in mc.AREA_ARTICLES.items()] + \
            [(f"fase `{k}`", refs) for k, (_, refs) in mc.POSTURE_ARTICLES.items()]
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        for name, refs in lists:
            for number, code in refs:
                rows = dg._fetch_any(session, documents, [(number, code)])
                s = (rows[0].get("s") or {}) if rows else {}
                heading = (s.get("titolo_articolo") or s.get("title") or "").strip() if rows else "NOT FOUND"
                lines.append(f"| {name} | {number} {code} | {heading or '(senza rubrica)'} |")
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--check", action="store_true", help="look each article up in the corpus (read-only)")
    args = parser.parse_args(argv)
    print(checked_table() if args.check else table())
    return 0


if __name__ == "__main__":
    sys.exit(main())
