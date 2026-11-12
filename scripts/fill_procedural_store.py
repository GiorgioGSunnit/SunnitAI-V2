#!/usr/bin/env python3
"""v60 (C): fill the procedural facts store -- each c.p. offence's procedibilità (v60h: with the conditions that make a
querela offence d'ufficio, procedural_facts.VERSION 2), querela term, remissione, competenza and pena, each c.p.p. / c.p.a. / c.p.c. article's stated term, computed from the article's own text
(src/rag/procedural_facts.extract) and stored on its Section node (proc_facts as JSON, proc_version). v60s: every string
cleaned of the corpus' markers («(abrogato)», footnotes) before it is written (procedural_facts.clean_facts).

Dry run by default: computes and prints the coverage, writes nothing. --write stores them.

  python scripts/fill_procedural_store.py                       # coverage only
  python scripts/fill_procedural_store.py --write               # store
  python scripts/fill_procedural_store.py --codes c.p. --write  # one code
  python scripts/fill_procedural_store.py --show 582,575,640    # the facts of some c.p. articles, nothing written
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ARTICLES_QUERY = ("MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id IN $docs AND s.titolo_articolo IS NOT NULL "
                  "RETURN DISTINCT d.id AS doc, s.name AS name")
WRITE_QUERY = ("UNWIND $items AS it MATCH (d:Document {id: it.doc})-[:CONTAINS]->(s:Section) "
               "WHERE s.name = it.num OR s.name = it.num + '.0.0' "
               "SET s.proc_facts = it.facts, s.proc_version = it.version RETURN count(s) AS n")
BATCH = 200


def main(argv=None) -> None:
    import neo4j
    from src.rag import article_text, defensive_generation as dg, procedural_facts
    from src.rag.main import NEO4J_DATABASE, driver

    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--codes", default=",".join(procedural_facts.CODES))
    parser.add_argument("--write", action="store_true", help="store the facts (default: coverage only)")
    parser.add_argument("--show", help="c.p. article numbers to print, comma separated (nothing written)")
    args = parser.parse_args(argv)
    codes = [c.strip() for c in args.codes.split(",") if c.strip()]
    mode = neo4j.WRITE_ACCESS if args.write else neo4j.READ_ACCESS
    coverage = {}
    with driver.session(database=NEO4J_DATABASE, default_access_mode=mode) as session:
        store = article_text.ArticleStore()
        for code in codes:
            prefix = dg._CODE_PREFIXES.get(code)
            if not prefix:
                print(f"{code}: no document prefix, skipped", file=sys.stderr)
                continue
            docs = store.prefix_docs(session, [prefix])[prefix]
            numbers = sorted({(r["doc"], article_text.article_number(r["name"])) for r in
                              session.run(ARTICLES_QUERY, docs=docs).data() if article_text.article_number(r["name"])})
            if args.show and code == "c.p.":
                wanted = {n.strip() for n in args.show.split(",")}
                numbers = [(d, n) for d, n in numbers if n in wanted]
            counts = Counter(articles=len(numbers))
            items = []
            for i in range(0, len(numbers), BATCH):
                found = article_text.fetch_many(session, numbers[i:i + BATCH], store)
                for (doc, number), rows in found.items():
                    if not rows:
                        continue
                    row = {**rows[0], "_article": (number, code)}
                    facts = procedural_facts.clean_facts(procedural_facts.from_row(row))   # v60s (6): no «(abrogato)»
                    if not facts:
                        continue
                    counts["with_facts"] += 1
                    counts["querela"] += bool(facts.get("querela"))
                    counts["ufficio_condizioni"] += bool(facts.get("ufficio_condizioni"))     # v60h (1)
                    counts["competenza"] += bool(facts.get("competenza"))
                    counts["pena"] += bool(facts.get("pena"))
                    counts["termine"] += bool(facts.get("termine"))
                    items.append({"doc": doc, "num": number, "facts": json.dumps(facts, ensure_ascii=False),
                                  "version": procedural_facts.VERSION})
                    if args.show:
                        print(f"art. {number} {code}: {json.dumps(facts, ensure_ascii=False)}")
            if args.write and not args.show:
                written = 0
                for i in range(0, len(items), BATCH):
                    written += session.run(WRITE_QUERY, items=items[i:i + BATCH]).single()["n"]
                counts["sections_written"] = written
            coverage[code] = dict(counts)
    print("| code | articles | with facts | a querela | d'ufficio conditions | competenza | pena | termine | sections written |")
    print("|---|---|---|---|---|---|---|---|---|")
    for code, c in coverage.items():
        print(f"| {code} | {c.get('articles', 0)} | {c.get('with_facts', 0)} | {c.get('querela', 0)} | "
              f"{c.get('ufficio_condizioni', 0)} | "
              f"{c.get('competenza', 0)} | {c.get('pena', 0)} | {c.get('termine', 0)} | "
              f"{c.get('sections_written', '—' if not args.write else 0)} |")
    from src.rag.verbose_logger import vlog
    vlog("procedural_store_coverage", {code: c for code, c in coverage.items()})


if __name__ == "__main__":
    main()
