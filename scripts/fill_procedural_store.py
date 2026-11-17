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
  python scripts/fill_procedural_store.py --codes c.p.c.,c.p.p. --show "641 c.p.c.,325 c.p.c.,585 c.p.p."
                                                                # v65: any code; each term's comma, subject, number
  python scripts/fill_procedural_store.py --codes c.p.c. --show "325 c.p.c." --text
                                                                # v67: each comma's text as the store reads it

v65 (procedural_facts.VERSION 3): a c.p.p. / c.p.a. / c.p.c. article's terms are extracted per comma, each with its
subject (the procedural noun of its clause: opposizione, appello, denuncia …), never the first number alone
(procedural_facts.termini); --show prints them one per line, «art. N code | co. K | subject | number unit», so they can
be checked. A v2 entry already stored is read as before, its terms computed from the text at read time.

v72: --notes FILE imports the lawyer's «note di inquadramento» (docs/lawyer/note_inquadramento.md: «## art. N codice»,
or «## art. N n. M codice» for one numero, the note under it) onto each article's Section node (src/rag/article_notes:
PROPERTY, JSON {"nota", "numeri"}); dry run by default, --write stores; an article the graph does not hold is listed.

  python scripts/fill_procedural_store.py --notes docs/lawyer/note_inquadramento.md --write

v78: --pronunce FILE imports the lawyer's «pronunce di riferimento» (docs/lawyer/pronunce_di_riferimento.md: «## art. N
codice», under it «- estremi — principio») onto each article's Section node (src/rag/landmark_rulings: PROPERTY, JSON
[{"estremi", "principio"}]), the same way: dry run by default, --write stores, an article the graph lacks is listed.

  python scripts/fill_procedural_store.py --pronunce docs/lawyer/pronunce_di_riferimento.md --write

v67 (procedural_facts.VERSION 4): each term's subject is the act that owns its number (procedural_facts.owner), a
«1-bis.» comma is its own; a v3 entry is read with its terms computed from the text again. --text (with --show)
prints each comma's text as the store reads it, and how the text was split.
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
    parser.add_argument("--show", help="articles to print, comma separated: «582» (every code in --codes) or «641 c.p.c.» "
                                       "(nothing written)")
    parser.add_argument("--text", action="store_true", help="with --show: each comma's text as the store reads it (v67)")
    parser.add_argument("--notes", help="v72: import the lawyer's notes from this file (article -> note)")
    parser.add_argument("--pronunce", help="v78: import the lawyer's pronunce di riferimento from this file")
    args = parser.parse_args(argv)
    if args.notes:
        import_notes(args.notes, args.write)
        return
    if args.pronunce:
        import_notes(args.pronunce, args.write, kind="pronunce")
        return
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
            if args.show:                   # v65: «N» for every code, «N code» for one
                wanted = shown(args.show, code)
                numbers = [(d, n) for d, n in numbers if n in wanted]
            counts = Counter(articles=len(numbers))
            items = []
            for i in range(0, len(numbers), BATCH):
                found = article_text.fetch_many(session, numbers[i:i + BATCH], store)
                for (doc, number), rows in found.items():
                    if not rows:
                        continue
                    row = {**rows[0], "_article": (number, code)}
                    if args.show and args.text:     # v67: the graph's wording, comma by comma, before any form reads it
                        for line in text_lines(row, f"art. {number} {code}"):
                            print(line)
                    facts = procedural_facts.clean_facts(procedural_facts.from_row(row))   # v60s (6): no «(abrogato)»
                    if not facts:
                        continue
                    counts["with_facts"] += 1
                    counts["querela"] += bool(facts.get("querela"))
                    counts["ufficio_condizioni"] += bool(facts.get("ufficio_condizioni"))     # v60h (1)
                    counts["competenza"] += bool(facts.get("competenza"))
                    counts["pena"] += bool(facts.get("pena"))
                    counts["termine"] += bool(facts.get("termine"))
                    counts["termini"] += len(facts.get("termini") or [])                       # v65
                    counts["termini_soggetto"] += sum(1 for t in facts.get("termini") or [] if t.get("subject"))
                    counts["termini_ambigui"] += sum(1 for t in facts.get("termini") or [] if t.get("ambiguous"))
                    items.append({"doc": doc, "num": number, "facts": json.dumps(facts, ensure_ascii=False),
                                  "version": procedural_facts.VERSION})
                    if args.show:
                        print(f"art. {number} {code}: {json.dumps(facts, ensure_ascii=False)}")
                        for t in facts.get("termini") or []:        # v65: one line per term, to check by eye
                            print(f"  art. {number} {code} | {t.get('comma')}"
                                  + (f", lett. {t['letter']})" if t.get("letter") else "")
                                  + f" | {t.get('subject') or '—'} | {t.get('qty')} {t.get('unit')}"
                                  + (f" {t['since']}" if t.get("since") else "")
                                  + (f" | quando: {t['when']}" if t.get("when") else "")
                                  + (" | AMBIGUO: «" + t.get("sentence", "")[:160] + "»" if t.get("ambiguous") else ""))
                        for d in facts.get("decorrenze") or []:     # v66: where they run from
                            print(f"  art. {number} {code} | {d.get('comma')}"
                                  + (f", lett. {d['letter']})" if d.get("letter") else "") + f" | decorrenza | {d['text'][:160]}")
            if args.write and not args.show:
                written = 0
                for i in range(0, len(items), BATCH):
                    written += session.run(WRITE_QUERY, items=items[i:i + BATCH]).single()["n"]
                counts["sections_written"] = written
            coverage[code] = dict(counts)
    print("| code | articles | with facts | a querela | d'ufficio conditions | competenza | pena | termine | termini "
          "| with a subject | ambiguous | sections written |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for code, c in coverage.items():
        print(f"| {code} | {c.get('articles', 0)} | {c.get('with_facts', 0)} | {c.get('querela', 0)} | "
              f"{c.get('ufficio_condizioni', 0)} | "
              f"{c.get('competenza', 0)} | {c.get('pena', 0)} | {c.get('termine', 0)} | {c.get('termini', 0)} | "
              f"{c.get('termini_soggetto', 0)} | {c.get('termini_ambigui', 0)} | "
              f"{c.get('sections_written', '—' if not args.write else 0)} |")
    from src.rag.verbose_logger import vlog
    vlog("procedural_store_coverage", {code: c for code, c in coverage.items()})


def import_notes(path: str, write: bool, kind: str = "notes") -> dict:
    """v72: the file's notes onto their Section nodes (article_notes.WRITE_QUERY); the counts, each article not found.
    v78: kind «pronunce» -- the lawyer's pronunce di riferimento (landmark_rulings), the same way."""
    import neo4j
    from src.rag import article_notes, article_text, defensive_generation as dg, landmark_rulings
    from src.rag.main import NEO4J_DATABASE, driver
    module = landmark_rulings if kind == "pronunce" else article_notes
    notes = module.parse(Path(path).read_text(encoding="utf-8"))
    report = {"notes": len(notes), "written": 0, "missing": [], "codes": {}}
    mode = neo4j.WRITE_ACCESS if write else neo4j.READ_ACCESS
    with driver.session(database=NEO4J_DATABASE, default_access_mode=mode) as session:
        store = article_text.ArticleStore()
        for code, items in module.store_items(notes).items():
            prefix = dg._CODE_PREFIXES.get(code)
            if not prefix:
                report["missing"] += [f"{n} {code} (no document prefix)" for n, _ in items]
                continue
            docs = store.prefix_docs(session, [prefix])[prefix]
            have = {(r["doc"], article_text.article_number(r["name"])) for r in session.run(ARTICLES_QUERY, docs=docs).data()}
            rows = []
            for number, note in items:
                found = [d for d, n in have if n == number]
                if not found:
                    report["missing"].append(f"{number} {code}")
                    continue
                rows += [{"doc": d, "num": number, "note": note} for d in found]
                print(f"art. {number} {code}: {note[:160]}")
            report["codes"][code] = len(rows)
            if write and rows:
                report["written"] += session.run(module.WRITE_QUERY, items=rows).single()["n"]
    print(json.dumps(report, ensure_ascii=False))
    return report


def text_lines(row: dict, where: str) -> list:
    """v67: --text -- how the article's text was split (comma_material: «lines», «numbered», «sentences») and each comma
    as procedural_facts reads it (rule_quote.row_commi, a «1-bis» comma its own), whole, so a form that reads nothing
    can be seen against the graph's own wording."""
    from src.rag import comma_material, procedural_facts, rule_quote
    s = row.get("s") or {}
    how = row.get(comma_material.FROM_KEY) or comma_material.article_parts(
        rule_quote.clean_text(s.get("plain_text") or s.get("abstract") or ""))[1]
    commi = rule_quote.row_commi(row, suffixed=True)
    out = [f"  {where} | testo: {len(commi)} commi ({how})"]
    out += [f"  {where} | {procedural_facts._label(k)} | «{text}»" for k, text in commi]
    return out


def shown(spec: str, code: str) -> set:
    """v65: the article numbers --show asks for in code: a bare «N» in every code, «N code» in its own."""
    from src.rag.plan_then_write import _code_key
    out = set()
    for item in (x.strip() for x in (spec or "").split(",") if x.strip()):
        number, _, own = item.partition(" ")
        if not own or _code_key(own.strip()) == _code_key(code):
            out.add(number.strip())
    return out


if __name__ == "__main__":
    main()
