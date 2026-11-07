"""Sections whose abstract is the ingest's error string instead of a summary: count them, and with --write replace
the abstract by what the ingest writes when the model analysis is absent (the section's first 200 words).

Background (7 Nov 2026): between 2 and 5 Nov the ingest's .env named a model the server no longer served. Every
chunk analysis failed, and the analysis module's last resort is a result whose "requirement" is the literal
"[Errore analisi comma]". flatten.py builds the section abstract from that requirement, so about 4,700 rulings of
the 2025 batch carry abstracts like "<title> - [Errore analisi comma]" (joined with "; " when a comma had several
chunks). The post-processing only replaces abstracts shorter than 20 characters, and this one is 22. The chatbot
returns s.abstract for a section view and searches abstracts for keywords, so the string is visible.

The replacement follows flatten.py's own rule for a section without analysis: the first 200 words of plain_text,
prefixed by the article title when the node has one ("titolo - text"), trimmed to 200 words. Sections whose
abstract contains "[Content filter" (the other canned string) are counted separately and left alone.

    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/fix_error_abstracts.py                 # count only
    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/fix_error_abstracts.py --write         # fix, in batches
The old abstract is kept on the node as abstract_before_fix, so --undo puts it back. Counts only, no text printed.
"""
import argparse
import sys
import time
from typing import Dict

ERROR = "[Errore analisi comma]"
FILTER = "[Content filter"


def load_env(path: str) -> Dict[str, str]:
    out = {}
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def trim_words(text: str, max_words: int = 200) -> str:
    words = (text or "").split()
    return " ".join(words[:max_words])


def fixed_abstract(plain_text: str, titolo: str) -> str:
    """flatten.py's abstract for a section without analysis: the first 200 words, with the article title in front."""
    abstract = trim_words(plain_text, 200)
    titolo = (titolo or "").strip()
    if titolo and titolo not in abstract:
        abstract = f"{titolo} - {abstract}"
    return trim_words(abstract, 200)


COUNT = (
    "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE s.abstract CONTAINS $needle "
    "RETURN count(s) AS sections, count(DISTINCT d) AS documents, "
    "count(DISTINCT CASE WHEN d.document_type = 'interpretation' THEN d END) AS rulings"
)
PICK = (
    "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE s.abstract CONTAINS $needle "
    "RETURN elementId(s) AS eid, s.plain_text AS text, s.titolo_articolo AS titolo LIMIT $n"
)
WRITE = (
    "UNWIND $rows AS row MATCH (s:Section) WHERE elementId(s) = row.eid "
    "SET s.abstract_before_fix = s.abstract, s.abstract = row.abstract"
)
UNDO = (
    "MATCH (s:Section) WHERE s.abstract_before_fix IS NOT NULL "
    "WITH s LIMIT $n SET s.abstract = s.abstract_before_fix REMOVE s.abstract_before_fix RETURN count(s) AS n"
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="/opt/v2tools/health.env", help="NEO4J_URI / USER / PASSWORD / DATABASE")
    p.add_argument("--write", action="store_true", help="replace the error abstracts (default: count only)")
    p.add_argument("--undo", action="store_true", help="put back the abstracts saved as abstract_before_fix")
    p.add_argument("--batch", type=int, default=500, help="sections per write")
    p.add_argument("--pause", type=float, default=0.2, help="seconds between writes")
    args = p.parse_args()

    env = load_env(args.env)
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"]))
    db = env.get("NEO4J_DATABASE", "neo4j")
    with driver.session(database=db) as s:
        if args.undo:
            total = 0
            while True:
                n = s.run(UNDO, n=args.batch).single()["n"]
                total += n
                if n < args.batch:
                    break
                time.sleep(args.pause)
            print(f"restored {total:,} abstracts from abstract_before_fix")
            return 0
        for label, needle in (("error string", ERROR), ("content-filter string", FILTER)):
            r = s.run(COUNT, needle=needle).single()
            print(f"abstracts with the {label}: {r['sections']:,} sections in {r['documents']:,} documents "
                  f"({r['rulings']:,} rulings)")
        if not args.write:
            print("count only; --write replaces the error-string abstracts (the old one is kept as abstract_before_fix)")
            return 0
        done = empty = 0
        seen = set()
        t0 = time.time()
        while True:
            rows = s.run(PICK, needle=ERROR, n=args.batch).data()
            if not rows or all(r["eid"] in seen for r in rows):      # nothing left, or a batch that did not change
                break
            seen.update(r["eid"] for r in rows)
            payload = []
            for r in rows:
                new = fixed_abstract(r["text"] or "", r["titolo"] or "")
                if not new.strip():
                    new = "(sezione senza testo)"          # never leave the error string, never write an empty abstract
                    empty += 1
                payload.append({"eid": r["eid"], "abstract": new})
            s.run(WRITE, rows=payload).consume()
            done += len(payload)
            if done % 5000 < args.batch:
                print(f"  {done:,} fixed ({time.time() - t0:.0f}s)")
            time.sleep(args.pause)
        print(f"fixed {done:,} abstracts ({empty:,} had no text) in {time.time() - t0:.0f}s")
        r = s.run(COUNT, needle=ERROR).single()
        print(f"left with the error string: {r['sections']:,}")
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
