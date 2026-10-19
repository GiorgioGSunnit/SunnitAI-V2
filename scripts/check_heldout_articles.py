"""Read-only: for every expected article of every held-out case (tests/
eval_narrative_cases.json, "heldout": true -- h1-h11), whether a document
matching its code exists in the corpus at all, and whether that article
number has a Section in it.

This only checks the DOCUMENTS exist -- it says nothing about whether
those are the right article numbers for the facts (the held-out set's own
"expected" lists are themselves provisional, "expected_status": "draft --
pending lawyer validation"; that's a legal judgement for the lawyer who
wrote the cases, not something this script can check). h10/h11's "l241"
(Legge 7 agosto 1990, n. 241, art. 21-septies) was written without access
to the live corpus -- this is specifically how to find out whether it (or
"Legge sul Procedimento Amministrativo", in case that's the document's
real name instead) actually exists, and under which name.

Run from the app folder so it uses the app's code, .env and venv:
    cd /opt/chatbot && venv/bin/python scripts/check_heldout_articles.py

Nothing is written to the database.
"""

import json
import logging
import os
import sys
from pathlib import Path

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

DEFAULT_CASES = Path(__file__).resolve().parent.parent / "tests" / "eval_narrative_cases.json"

# The eval's own code key -> a name to search the real document names/
# aliases with (issue_spotting.resolve_code handles the six traditional
# codes and any other law by name/number+year alike). "l241" and
# "legge sul procedimento amministrativo" are both tried for h10, since
# which one the corpus actually uses under isn't known without this check.
_EVAL_CODE_TO_SEARCH_NAMES = {
    "cp": ["c.p."], "cc": ["c.c."], "cpc": ["c.p.c."], "cpp": ["c.p.p."],
    "consumo": ["Codice del Consumo"], "l392": ["L. 392/1978"],
    "cpa": ["Codice del processo amministrativo"],
    "l241": ["Legge 241/1990", "Legge sul Procedimento Amministrativo"],
}


def main():
    import src.rag.defensive_generation as dg
    from src.rag.doc_lookup import _fetch_doc_names
    from src.rag.issue_spotting import fetch_article, resolve_code
    from src.rag.main import NEO4J_DATABASE, driver

    cases = [c for c in json.loads(DEFAULT_CASES.read_text(encoding="utf-8"))["cases"] if c.get("heldout")]
    if not cases:
        sys.exit("no held-out cases found in " + str(DEFAULT_CASES))

    documents = _fetch_doc_names(driver, NEO4J_DATABASE)
    print(f"{len(documents)} documents in the corpus.\n")

    with driver.session(database=NEO4J_DATABASE) as session:
        for case in cases:
            print(f"## {case['id']}")
            for item in case.get("expected", []):
                code, number = item["code"], item["article"]
                names = _EVAL_CODE_TO_SEARCH_NAMES.get(code, [code])
                resolved = None
                tried = []
                for name in names:
                    doc = resolve_code(name, documents)
                    tried.append(f"{name!r} -> {doc['name'] if doc else 'no match'}")
                    if doc and resolved is None:
                        resolved = doc
                if resolved is None:
                    print(f"  {number:12s} {code:8s} DOCUMENT NOT FOUND ({'; '.join(tried)})")
                    continue
                rows = fetch_article(session, resolved["id"], number)
                found = "section found" if rows else "NO SECTION for this number"
                print(f"  {number:12s} {code:8s} {resolved['name']!r}: {found}")
            print()
    print("Done. Nothing was written.")


if __name__ == "__main__":
    main()
