"""Give Sections named "word.N..." back their article's name (29 Oct): Costituzione art. 3 is stored as
"tutti_cittadini.3.0.0", so every lookup by number finds nothing -- the strategy's (article_text.fetch_article), the
linker's, the rulings citing an article (source_tiers.citing_rulings). Arts. 1-12, 19, 20 Cost. are never read by a
strategy answer and 3,482 ruling citations of them stay unlinked; the chat finds them only through article_router's
fallback, when it already knows the document. About 40 other primary documents have such names too (cost_check.py
on prod, 29 Oct), mostly for their arts. 1-20.

The new name drops the leading word: "tutti_cittadini.3.0.0" -> "3.0.0", "tutte_confessioni.8_3" -> "8_3". A name is
left alone, and listed, when
  - the word has digits ("capitolo_1_x.1", "definizioni32.1.1_7", "articolo_2.3.1.0": the number may not be the
    article's), or the rest is not an article's name ("012.0.0", "5.2.ALTRI ELEMENTI...");
  - the new name is already in the document, or two old names give the same one.
The old name is kept in name_before_rename (the API opening a source by name also tries it); --undo puts it back.
Nothing is deleted.

  --doc NAME                  one document, by its exact name: a dry run listing every rename
  --doc NAME --write          rename them
  --doc NAME --undo --write   put the old names back
  --all                       every primary document: counts only

    cd /opt/chatbot && venv/bin/python /root/rename_odd_sections.py --doc "Costituzione della Repubblica Italiana 2025"
"""

import argparse
import datetime as dt
import os
import re
import sys
from collections import Counter
from typing import Iterable, List, Optional, Set, Tuple

_ODD = re.compile(r"^([a-z][a-z0-9_]*)\.(.+)$")
_ARTICLE_NAME = re.compile(r"^[1-9]\d*(?:-[a-z]+)?(?:\.\d+)*(?:_\d+)*$")
ODD_CYPHER = "^[a-z][a-z0-9_]*[.].+$"


def new_name(name: str) -> Tuple[Optional[str], Optional[str]]:
    """(the new name, None), or (None, why it is left alone); (None, None) for a name that is not "word.N"."""
    m = _ODD.match(name or "")
    if not m:
        return None, None
    word, rest = m.groups()
    if re.search(r"\d", word):
        return None, "digits in the word"
    if not _ARTICLE_NAME.match(rest):
        return None, "not an article's name"
    return rest, None


def _order(name: str) -> tuple:
    return tuple(int(x) for x in re.findall(r"\d+", name or "")[:4])


def plan(odd: Iterable[str], existing: Set[str]) -> Tuple[List[Tuple[str, str]], List[Tuple[str, str]]]:
    """([(old, new)] to rename, [(old, why)] left alone) for one document's odd Section names; `existing` holds its
    other names (at least those equal to a new name)."""
    odd = sorted({n for n in odd if n}, key=lambda n: (_order(n), n))
    wanted = {old: new_name(old) for old in odd}
    gives = Counter(new for new, _ in wanted.values() if new)
    renames, left = [], []
    for old, (new, why) in wanted.items():
        if new is None and why is None:
            continue
        if new and new in existing:
            new, why = None, "the new name is already in the document"
        elif new and gives[new] > 1:
            new, why = None, "two old names give it"
        if new:
            renames.append((old, new))
        else:
            left.append((old, why))
    return renames, left


def _articles(renames: List[Tuple[str, str]]) -> str:
    numbers = sorted({int(re.match(r"\d+", new).group(0)) for _, new in renames})
    return ", ".join(map(str, numbers))


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--doc", metavar="NAME")
    p.add_argument("--all", action="store_true")
    p.add_argument("--undo", action="store_true")
    p.add_argument("--write", action="store_true")
    args = p.parse_args()
    if not (args.doc or args.all) or (args.all and (args.write or args.undo)):
        p.error("--doc NAME [--write] [--undo], or --all (counts only)")

    sys.path.insert(0, os.getcwd())
    from dotenv import load_dotenv
    load_dotenv(os.path.join(os.getcwd(), ".env"))
    import neo4j
    from src.rag.main import NEO4J_DATABASE, driver
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from quarantine_docs import kind_of

    mode = neo4j.WRITE_ACCESS if args.write else neo4j.READ_ACCESS
    with driver.session(database=NEO4J_DATABASE, default_access_mode=mode) as session:
        if args.all:
            rows = session.run(
                "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.document_type = 'primary' AND s.name =~ $odd "
                "RETURN d.id AS id, d.name AS name, collect(DISTINCT s.name) AS odd", odd=ODD_CYPHER).data()
            wanted = [{"doc": r["id"], "name": new} for r in rows for new, _ in map(new_name, r["odd"]) if new]
            taken = {(x["doc"], x["name"]) for x in session.run(
                "UNWIND $items AS it MATCH (d:Document {id: it.doc})-[:CONTAINS]->(s:Section {name: it.name}) "
                "RETURN DISTINCT it.doc AS doc, it.name AS name", items=wanted).data()}
            out = []
            for r in rows:
                renames, left = plan(r["odd"], {name for doc, name in taken if doc == r["id"]})
                kind = kind_of(r["name"])[0]
                out.append((len(renames), len(left), r["name"] if kind == "other" else f"({kind})", renames))
            print(f"{'rename':>6} {'left':>5}  document: the articles renamed")
            for n, n_left, name, renames in sorted(out, key=lambda x: -x[0]):
                print(f"{n:>6} {n_left:>5}  {name}: {_articles(renames) or '-'}")
            print(f"total: {sum(x[0] for x in out)} to rename, {sum(x[1] for x in out)} left alone, "
                  f"in {len(out)} documents")
            return 0

        docs = session.run("MATCH (d:Document {name: $name}) RETURN d.id AS id, d.document_type AS type",
                           name=args.doc).data()
        if not docs:
            print(f"no document named {args.doc!r}")
            return 1
        at = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
        for doc in docs:
            print(f"{args.doc} ({doc['type']}, {doc['id']})")
            if args.undo:
                n = session.run("MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) "
                                "WHERE s.name_before_rename IS NOT NULL RETURN count(s) AS n", id=doc["id"]).single()["n"]
                print(f"  renamed Sections to put back: {n}" + ("" if args.write else " (dry run: add --write)"))
                if args.write and n:
                    done = session.run(
                        "MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) WHERE s.name_before_rename IS NOT NULL "
                        "SET s.name = s.name_before_rename REMOVE s.name_before_rename, s.renamed_at "
                        "RETURN count(s) AS n", id=doc["id"]).single()["n"]
                    print(f"  put back: {done}")
                continue
            names = [r["name"] for r in session.run(
                "MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section) RETURN s.name AS name", id=doc["id"]).data()]
            renames, left = plan([n for n in names if _ODD.match(n or "")],
                                 {n for n in names if n and not _ODD.match(n)})
            for old, new in renames:
                print(f"  {old} -> {new}")
            for old, why in left:
                print(f"  left alone: {old} ({why})")
            print(f"  {len(renames)} to rename (arts. {_articles(renames) or '-'}), {len(left)} left alone"
                  + ("" if args.write else " -- dry run: add --write"))
            if args.write and renames:
                done = session.run(
                    "UNWIND $pairs AS p MATCH (d:Document {id: $id})-[:CONTAINS]->(s:Section {name: p.old}) "
                    "WHERE s.name_before_rename IS NULL "
                    "SET s.name_before_rename = s.name, s.name = p.new, s.renamed_at = $at RETURN count(s) AS n",
                    id=doc["id"], at=at, pairs=[{"old": o, "new": n} for o, n in renames]).single()["n"]
                print(f"  renamed: {done} Sections  (undo: --undo --write)")
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
