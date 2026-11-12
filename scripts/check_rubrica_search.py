#!/usr/bin/env python3
"""v60n: the first read's rubrica search (fact_resolver.resolve, channel 1 of interactive_flow.first_read_search) checked
on the store -- every single-word rubrica of a code's range, by default c.p. Libro II (artt. 241-649), searched as an
institution name must find its own article; and named searches with the articles they must find («Corruzione» ->
318, 319, 320). v60o: identical rubriche («Confisca», seven in c.p.) are one search, each of their articles found when
the search's result set contains it. Read only on the graph; one line per article, a summary; exit 1 on a miss.

  python scripts/check_rubrica_search.py                                   # c.p. Libro II, and the default names
  python scripts/check_rubrica_search.py --code c.c. --from 1100 --to 1172
  python scripts/check_rubrica_search.py --names "Corruzione=318,319,320" "Truffa=640"
"""

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NAMES = ("Truffa=640", "Furto=624", "Corruzione=318,319,320", "Appropriazione indebita=646", "Rapina=628",
         "Estorsione=629", "Minaccia=612", "Omicidio=575")


def single_word(rubrica: str) -> str:
    """The rubrica's one word (rubrica_key), "" when it has more."""
    from src.rag import fact_resolver
    key = fact_resolver.rubrica_key(rubrica)
    return key if key and " " not in key else ""


def in_range(number: str, lo: int, hi: int) -> bool:
    m = re.match(r"\d+", number or "")
    return bool(m) and lo <= int(m.group(0)) <= hi


def check(session, documents, code: str, name: str, wanted) -> tuple:
    """(found labels, how, missing numbers) of one search."""
    from src.rag import fact_resolver
    rows, log = fact_resolver.resolve(session, documents, {"nome": name, "codice": code})
    found = list(dict.fromkeys(str(r["_article"][0]) for r in rows if r.get("_article")))
    how = (log.get("rubrica") or {}).get("how") or ("refused: " + "; ".join(x.get("why") or "" for x in log["refused"][:1])
                                                    if not found else "indici")
    return found, how, [n for n in wanted if n not in found]


def main(argv=None) -> int:
    import neo4j
    from src.rag import fact_resolver
    from src.rag.doc_lookup import _fetch_doc_names
    from src.rag.main import NEO4J_DATABASE, driver
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--code", default="c.p.")
    parser.add_argument("--from", dest="lo", type=int, default=241)
    parser.add_argument("--to", dest="hi", type=int, default=649)
    parser.add_argument("--names", nargs="*", default=list(NAMES), help="NAME=N,N... (the articles it must find)")
    args = parser.parse_args(argv)
    documents = _fetch_doc_names(driver, NEO4J_DATABASE)
    doc, abbr = fact_resolver._document(documents, args.code)
    if not doc:
        print(f"code {args.code!r}: no document")
        return 1
    misses, total, generic = [], 0, []
    with driver.session(database=NEO4J_DATABASE, default_access_mode=neo4j.READ_ACCESS) as session:
        rubriche = fact_resolver.code_rubriche(session, doc["id"])
        words = [(n, r, single_word(r)) for n, r in rubriche if in_range(n, args.lo, args.hi)]
        words = [(n, r, w) for n, r, w in words if w]
        print(f"# {args.code} artt. {args.lo}-{args.hi}: {len(words)} single-word rubriche of {len(rubriche)} "
              f"in «{doc['name']}»\n")
        print("| article | rubrica | found | how | ok |\n|---|---|---|---|---|")
        searched = {}           # v60o: one search per word -- identical rubriche share its result set
        for number, rubrica, word in words:
            if not fact_resolver.matching_words(word):
                generic.append(f"{number} «{rubrica}»")
                print(f"| {number} | {rubrica} | - | generic word (left out) | - |")
                continue
            total += 1
            if word not in searched:
                searched[word] = check(session, documents, args.code, word.capitalize(), [])[:2]
            found, how = searched[word]
            shared = sum(1 for _, _, w in words if w == word)
            ok = number in found
            if not ok:
                misses.append(f"{number} «{rubrica}» -> {', '.join(found) or 'nothing'}")
            print(f"| {number} | {rubrica}{f' ({shared} identiche)' if shared > 1 else ''} | {', '.join(found) or '-'} "
                  f"| {how} | {'yes' if ok else 'NO'} |")
        print("\n| name | must find | found | how | ok |\n|---|---|---|---|---|")
        for spec in args.names:
            name, _, numbers = spec.partition("=")
            wanted = [x.strip() for x in numbers.split(",") if x.strip()]
            total += 1
            found, how, missing = check(session, documents, args.code, name.strip(), wanted)
            if missing:
                misses.append(f"«{name}» -> {', '.join(found) or 'nothing'} (missing {', '.join(missing)})")
            print(f"| {name} | {', '.join(wanted)} | {', '.join(found) or '-'} | {how} | {'NO' if missing else 'yes'} |")
    print(f"\n{total - len(misses)}/{total} searches found their articles"
          + (f"; generic words left out: {', '.join(generic)}" if generic else ""))
    for m in misses:
        print(f"MISS {m}")
    return 1 if misses else 0


if __name__ == "__main__":
    sys.exit(main())
