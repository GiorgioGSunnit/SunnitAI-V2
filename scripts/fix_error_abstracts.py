"""Sections whose abstract carries the ingest's error string instead of a summary: count them, and with --write
rebuild the abstract the way the ingest (flatten.py) builds it when a chunk has no analysis.

Background (7 Nov 2026): the analysis module's last resort for a chunk whose model call fails twice (structured,
then plain text: a 404 on the model name, a chunk timeout, any error) is a result whose "requirement" is the
literal "[Errore analisi comma]". flatten.py joins the chunk requirements of a section with "; " and makes the
abstract from them ("<title> - <requirements> Testo significativo: <core text>"), so the string ends up in the
abstract: alone when every chunk failed, mixed with real summaries when one chunk of several failed. The
post-processing then strips "Testo significativo:" and "0 " from every abstract, so the stored shape differs from
flatten's: the core text follows the error string with a bare space, and a title with a zero no longer matches
the node's titolo_articolo. It replaces only abstracts under 20 characters. The document description (first 500
words of the section abstracts at flatten time) carries the string too. The chatbot returns s.abstract in the
section view, searches abstracts and descriptions for keywords, and the nightly job embeds abstract + plain_text.

Count on 8 Nov 2026: 37,510 sections in 5,993 rulings; 5,992 descriptions.

The fix removes the error fragments wherever they sit (with their "; " joiners). When nothing of the analysis is
left but the title, the abstract becomes flatten's no-analysis abstract: the first 200 words of the text with the
node's title in front. Otherwise what is left is kept, trimmed to 200 words. Then the description of every
document with a fixed section, or whose description carries the string, is rebuilt from its section abstracts
in reading order (NEXT), trimmed to 500 words. Nothing is printed but counts and masked shapes (every word
replaced by w: no text, no names).

    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/fix_error_abstracts.py            # count only
    ... --shapes             # read-only: the shapes of a sample of error abstracts, and what the fix would do
    ... --write              # fix, in batches
    ... --write --reembed    # also park each fixed section's vector (embedding_v2_before_fix) and clear
                             # embedding_v2, so the 02:00 backfill re-embeds it from the new text. Until that job
                             # runs, those sections are absent from vector search: run the fix in the evening,
                             # or start the backfill by hand right after. Not needed for retrieval (the vectors
                             # differ from the new text by a few tokens); it is for text / vector consistency.
    ... --undo               # put back abstracts, descriptions and parked vectors
    ... --purge-backups      # after checking the result: remove the *_before_fix properties
The old values stay on the nodes as abstract_before_fix / description_before_fix / embedding_v2_before_fix until
--purge-backups (a rerun keeps the first backup), so --undo restores everything.
"""
import argparse
import re
import sys
import time
from collections import Counter
from typing import Dict, Iterable, List, Optional, Set, Tuple

ERROR = "[Errore analisi comma]"
FILTER = "[Content filter"
ERR_RE = re.compile(r"\[Errore analisi comma\]|\[Content filter[^\]]*\]")
ERR_WITH_JOINER_BEFORE = re.compile(r"\s*;\s*(?:\[Errore analisi comma\]|\[Content filter[^\]]*\])")
ERR_WITH_JOINER_AFTER = re.compile(r"(?:\[Errore analisi comma\]|\[Content filter[^\]]*\])\s*;\s*")


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
    """flatten.py's _trim_words."""
    words = (text or "").split()
    return " ".join(words[:max_words])


def strip_errors(text: str) -> str:
    """The abstract without its error fragments: "a; [ERR]; b" -> "a; b", "[ERR] core" -> "core", "T - [ERR]" -> "T -"."""
    text = ERR_WITH_JOINER_BEFORE.sub("", text or "")
    text = ERR_WITH_JOINER_AFTER.sub("", text)
    text = ERR_RE.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text.strip("; ").strip()


def rebuild_abstract(abstract: str, plain_text: str, titolo: str) -> Tuple[str, bool]:
    """(new abstract, pure). pure = nothing of the analysis is left but the title: the abstract then becomes the
    first 200 words of the text with the title in front, as flatten.py does for a section without analysis."""
    titolo = (titolo or "").strip()
    body = strip_errors(abstract)
    rest = body
    if titolo and rest.startswith(titolo):
        rest = rest[len(titolo):].strip()
    if rest.startswith("-"):
        rest = rest[1:].strip()
    pure = rest == "" or body.endswith("-")       # "T -": a heading (the node's title or not) and nothing after it
    if pure:
        # the heading the abstract already carried stays in front (flatten puts the title there; the
        # post-processing adds the section name for short sections): the node's title when it has one, else
        # whatever stood before the error string
        heading = titolo or body.rstrip("- ").strip()
        new = trim_words(plain_text, 200)
        if heading and heading not in new:
            new = f"{heading} - {new}"
        new = trim_words(new, 200)
    else:
        new = trim_words(body, 200)
    if not new.strip():
        new = "(sezione senza testo)"               # never leave the error string, never write an empty abstract
    return new, pure


def shape(text: str, width: int = 90) -> str:
    """A masked outline: error strings as §, every run of words as W<n>. No word of the text survives."""
    t = ERR_RE.sub("§", text or "")
    t = re.sub(r"\w+", "w", t)
    t = re.sub(r"w(?: w)+", lambda m: f"W{m.group(0).count('w')}", t)
    return t[:width]


SPLIT_PIECE = re.compile(r"^(.*)_\d+$")           # flatten's name_i pieces of a section over MAX_SECTION_CHARS


def name_key(name: str):
    """Natural order of section names: 2 before 10, 1.2 before 1.10 (the app's own rule for a broken chain)."""
    return [int(p) if p.isdigit() else p.lower() for p in re.split(r"(\d+)", name or "")]


def main_sections(rows: List[dict]) -> List[dict]:
    """Without the split pieces name_i whose base section is present (flatten keeps them out of the description)."""
    names = {r.get("name") for r in rows}
    out = []
    for r in rows:
        m = SPLIT_PIECE.match(r.get("name") or "")
        if m and m.group(1) in names:
            continue
        out.append(r)
    return out


def chain_walk(rows: List[dict]) -> Tuple[List[dict], str]:
    """Follow NEXT from the single head: (sections in chain order, health). health = 'single' when one head leads
    through every main section; 'no_next' (no NEXT at all), 'multi_head', 'fan_out' (a section with two NEXT),
    'external' (a NEXT pointing outside the document) or 'partial' (one head, but sections off the chain)."""
    main = main_sections(rows)
    by_eid: Dict[str, dict] = {}
    out_degree: Counter = Counter()
    for r in main:
        by_eid[r["eid"]] = r
        if r.get("next"):
            out_degree[r["eid"]] += 1
    targets = {r["next"] for r in main if r.get("next")}
    if not targets:
        return [], "no_next"
    if any(n > 1 for n in out_degree.values()):
        return [], "fan_out"
    if any(t not in by_eid for t in targets):
        return [], "external"
    heads = [r for r in by_eid.values() if r["eid"] not in targets]
    if len(heads) != 1:
        return [], "multi_head"
    ordered, seen = [], set()
    cur: Optional[dict] = heads[0]
    while cur and cur["eid"] not in seen:
        seen.add(cur["eid"])
        ordered.append(cur)
        cur = by_eid.get(cur.get("next") or "")
    return ordered, ("single" if len(ordered) == len(by_eid) else "partial")


def chain_order(rows: List[dict]) -> List[dict]:
    """The document's main sections in reading order: along NEXT when one chain links them all, else in natural
    name order (the app's rule in article_text._reading_order)."""
    ordered, health = chain_walk(rows)
    if health == "single":
        return ordered
    unique = {r["eid"]: r for r in main_sections(rows)}          # a section with two NEXT came as two rows
    return sorted(unique.values(), key=lambda r: name_key(r.get("name") or ""))


def chain_health(rows: List[dict]) -> str:
    return chain_walk(rows)[1]


def rebuild_description(rows: List[dict]) -> str:
    return trim_words(" ".join((r.get("abstract") or "") for r in chain_order(rows)), 500)


COUNT_SECTIONS = (
    "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE s.abstract CONTAINS $needle "
    "RETURN count(s) AS sections, count(DISTINCT d) AS documents, "
    "count(DISTINCT CASE WHEN d.document_type = 'interpretation' THEN d END) AS rulings"
)
COUNT_DOCS = (
    "MATCH (d:Document) WHERE d.description CONTAINS $needle "
    "RETURN count(d) AS documents, count(CASE WHEN d.document_type = 'interpretation' THEN 1 END) AS rulings"
)
COUNT_BACKUPS = (
    "MATCH (s:Section) WHERE s.abstract_before_fix IS NOT NULL "
    "RETURN count(s) AS n, count(CASE WHEN s.abstract CONTAINS $needle THEN 1 END) AS still"
)
PICK_SECTIONS = (
    "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE s.abstract CONTAINS $needle "
    "RETURN elementId(s) AS eid, elementId(d) AS did, s.abstract AS abstract, s.plain_text AS text, "
    "s.titolo_articolo AS titolo LIMIT $n"
)
WRITE_SECTIONS = (
    "UNWIND $rows AS row MATCH (s:Section) WHERE elementId(s) = row.eid "
    "SET s.abstract_before_fix = coalesce(s.abstract_before_fix, s.abstract), s.abstract = row.abstract"
)
PARK_VECTORS = (
    "UNWIND $eids AS eid MATCH (s:Section) WHERE elementId(s) = eid AND s.embedding_v2 IS NOT NULL "
    "SET s.embedding_v2_before_fix = s.embedding_v2 REMOVE s.embedding_v2 RETURN count(s) AS n"
)
PICK_DOCS = "MATCH (d:Document) WHERE d.description CONTAINS $needle RETURN elementId(d) AS eid LIMIT $n"
DOCS_WITH_BACKUP = "MATCH (d:Document) WHERE d.description_before_fix IS NOT NULL RETURN elementId(d) AS eid LIMIT $n"
DOC_SECTIONS = (
    "MATCH (d:Document) WHERE elementId(d) = $did MATCH (d)-[:CONTAINS]->(s:Section) "
    "OPTIONAL MATCH (s)-[:NEXT]->(nx:Section) "
    "RETURN elementId(s) AS eid, s.name AS name, s.abstract AS abstract, elementId(nx) AS next"
)
WRITE_DOCS = (
    "UNWIND $rows AS row MATCH (d:Document) WHERE elementId(d) = row.eid "
    "SET d.description_before_fix = coalesce(d.description_before_fix, d.description), d.description = row.description"
)
UNDO_SECTIONS = (
    "MATCH (s:Section) WHERE s.abstract_before_fix IS NOT NULL WITH s LIMIT $n "
    "SET s.abstract = s.abstract_before_fix REMOVE s.abstract_before_fix RETURN count(s) AS n"
)
UNDO_VECTORS = (
    "MATCH (s:Section) WHERE s.embedding_v2_before_fix IS NOT NULL WITH s LIMIT $n "
    "SET s.embedding_v2 = s.embedding_v2_before_fix REMOVE s.embedding_v2_before_fix RETURN count(s) AS n"
)
UNDO_DOCS = (
    "MATCH (d:Document) WHERE d.description_before_fix IS NOT NULL WITH d LIMIT $n "
    "SET d.description = d.description_before_fix REMOVE d.description_before_fix RETURN count(d) AS n"
)
PURGE = {
    "abstract_before_fix": "MATCH (s:Section) WHERE s.abstract_before_fix IS NOT NULL WITH s LIMIT $n "
                           "REMOVE s.abstract_before_fix RETURN count(s) AS n",
    "embedding_v2_before_fix": "MATCH (s:Section) WHERE s.embedding_v2_before_fix IS NOT NULL WITH s LIMIT $n "
                               "REMOVE s.embedding_v2_before_fix RETURN count(s) AS n",
    "description_before_fix": "MATCH (d:Document) WHERE d.description_before_fix IS NOT NULL WITH d LIMIT $n "
                              "REMOVE d.description_before_fix RETURN count(d) AS n",
}


def run_batches(session, query: str, batch: int, pause: float) -> int:
    total = 0
    while True:
        n = session.run(query, n=batch).single()["n"]
        total += n
        if n < batch:
            return total
        time.sleep(pause)


def counts(session) -> None:
    for label, needle in (("error string", ERROR), ("content-filter string", FILTER)):
        r = session.run(COUNT_SECTIONS, needle=needle).single()
        print(f"section abstracts with the {label}: {r['sections']:,} sections in {r['documents']:,} documents "
              f"({r['rulings']:,} rulings)")
    r = session.run(COUNT_DOCS, needle=ERROR).single()
    print(f"document descriptions with the error string: {r['documents']:,} documents ({r['rulings']:,} rulings)")
    r = session.run(COUNT_BACKUPS, needle=ERROR).single()
    if r["n"]:
        print(f"sections already holding a backup (abstract_before_fix): {r['n']:,}, of which {r['still']:,} still "
              f"carry the error string")


def shapes(session, sample: int) -> None:
    """Read-only: what the error abstracts look like (masked) and what the fix would make of them."""
    rows = session.run(PICK_SECTIONS, needle=ERROR, n=sample).data()
    n = len(rows)
    print(f"sample: {n:,} sections with the error string")
    if not n:
        return
    c = Counter()
    before, after, unresolved = Counter(), Counter(), Counter()
    for r in rows:
        a, titolo = r["abstract"] or "", (r["titolo"] or "").strip()
        c["title empty"] += not titolo
        c["abstract starts with the node's title"] += bool(titolo) and a.startswith(titolo)
        c["abstract starts with 'title - '"] += bool(titolo) and a.startswith(f"{titolo} - ")
        c["error string at the end"] += a.rstrip().endswith(ERROR)
        c["text after the last error string"] += not a.rstrip().endswith(ERROR)
        c["'; ' joiner present"] += "; " in a
        c["several error strings"] += a.count(ERROR) > 1
        new, pure = rebuild_abstract(a, r["text"] or "", titolo)
        c["fix: nothing left but the title -> first 200 words"] += pure
        c["fix: good fragments kept"] += not pure
        c["fix: still carries the error (would not be written)"] += bool(ERR_RE.search(new))
        before[shape(a)] += 1
        after[shape(new)] += 1
        if ERR_RE.search(new):
            unresolved[shape(a)] += 1
    for k, v in c.items():
        print(f"  {v:>6,}  {k}")
    print("most common shapes before (W<n> = n words, § = error string):")
    for s, v in before.most_common(12):
        print(f"  {v:>6,}  {s}")
    print("most common shapes after the fix:")
    for s, v in after.most_common(8):
        print(f"  {v:>6,}  {s}")
    if unresolved:
        print("shapes the fix would NOT resolve:")
        for s, v in unresolved.most_common(8):
            print(f"  {v:>6,}  {s}")
    docs = [r["eid"] for r in session.run(DOCS_WITH_BACKUP, n=300)]
    if not docs:
        docs = sorted({r["did"] for r in rows})[:300]
    if docs:
        health, heads_hist, sizes = Counter(), Counter(), Counter()
        for did in docs:
            drows = session.run(DOC_SECTIONS, did=did).data()
            health[chain_health(drows)] += 1
            main = main_sections(drows)
            targets = {r["next"] for r in main if r.get("next")}
            heads = sum(1 for r in main if r["eid"] not in targets)
            heads_hist["1" if heads == 1 else "2" if heads == 2 else "3-5" if heads <= 5 else ">5"] += 1
            sizes["sections"] += len(drows)
            sizes["main sections"] += len(main)
            sizes["NEXT edges"] += sum(1 for r in main if r.get("next"))
        print(f"NEXT chain of {len(docs):,} documents (ones rebuilt by the first write when present): "
              + ", ".join(f"{k} {v:,}" for k, v in health.most_common()))
        print("  heads per document: " + ", ".join(f"{k}: {v:,}" for k, v in sorted(heads_hist.items())))
        print("  totals: " + ", ".join(f"{k} {v:,}" for k, v in sizes.items()))
        print("  description order: the chain when one head links every main section, else natural name order")


def fix_sections(session, args) -> Tuple[int, int, Set[str]]:
    done = pure_n = parked = 0
    unresolved = Counter()
    seen: Set[str] = set()
    dids: Set[str] = set()
    t0 = time.time()
    while True:
        rows = session.run(PICK_SECTIONS, needle=ERROR, n=args.batch).data()
        fresh = [r for r in rows if r["eid"] not in seen]
        if not fresh:                                           # nothing left, or only sections this run could not fix
            break
        seen.update(r["eid"] for r in fresh)
        payload = []
        for r in fresh:
            new, pure = rebuild_abstract(r["abstract"] or "", r["text"] or "", r["titolo"] or "")
            if ERR_RE.search(new):                              # must not happen: never write the string back
                unresolved[shape(r["abstract"] or "")] += 1
                continue
            pure_n += pure
            payload.append({"eid": r["eid"], "abstract": new})
            dids.add(r["did"])
        if payload:
            session.run(WRITE_SECTIONS, rows=payload).consume()
            if args.reembed:
                parked += session.run(PARK_VECTORS, eids=[p["eid"] for p in payload]).single()["n"]
            done += len(payload)
        if done % 5000 < args.batch:
            print(f"  {done:,} sections fixed ({time.time() - t0:.0f}s)")
        time.sleep(args.pause)
    print(f"fixed {done:,} section abstracts ({pure_n:,} had nothing left of the analysis, "
          f"{done - pure_n:,} kept their good fragments) in {time.time() - t0:.0f}s"
          + (f"; {parked:,} vectors parked for the nightly re-embedding" if args.reembed else ""))
    if unresolved:
        print(f"NOT fixed: {sum(unresolved.values()):,} sections whose rebuilt abstract would still carry the string; "
              "their masked shapes:")
        for s, v in unresolved.most_common(8):
            print(f"  {v:>6,}  {s}")
    return done, pure_n, dids


def fix_descriptions(session, args, dids: Iterable[str]) -> int:
    """Rebuild the description of every document with a fixed section, and of every document whose description
    still carries the string (the first-500-words window may have moved past a fixed section)."""
    todo: Set[str] = set(dids)
    seen: Set[str] = set()
    while True:
        more = {r["eid"] for r in session.run(PICK_DOCS, needle=ERROR, n=args.batch)} - seen
        if not more:
            break
        todo |= more
        seen |= more
    todo_list = sorted(todo)
    done = 0
    t0 = time.time()
    for start in range(0, len(todo_list), args.batch):
        chunk = todo_list[start:start + args.batch]
        payload = [{"eid": did, "description": rebuild_description(session.run(DOC_SECTIONS, did=did).data())}
                   for did in chunk]
        session.run(WRITE_DOCS, rows=payload).consume()
        done += len(payload)
        time.sleep(args.pause)
    print(f"rebuilt {done:,} document descriptions in {time.time() - t0:.0f}s")
    return done


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="/opt/v2tools/health.env", help="NEO4J_URI / USER / PASSWORD / DATABASE")
    p.add_argument("--shapes", action="store_true", help="read-only: masked shapes of a sample and the fix's effect")
    p.add_argument("--sample", type=int, default=3000, help="sections in the --shapes sample")
    p.add_argument("--write", action="store_true", help="fix the abstracts and descriptions (default: count only)")
    p.add_argument("--reembed", action="store_true",
                   help="with --write: park embedding_v2 of each fixed section so the nightly backfill re-embeds it")
    p.add_argument("--undo", action="store_true", help="put back abstracts, descriptions and parked vectors")
    p.add_argument("--purge-backups", action="store_true", help="remove the *_before_fix properties")
    p.add_argument("--batch", type=int, default=500, help="nodes per write")
    p.add_argument("--pause", type=float, default=0.2, help="seconds between writes")
    args = p.parse_args()

    env = load_env(args.env)
    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(env["NEO4J_URI"], auth=(env["NEO4J_USER"], env["NEO4J_PASSWORD"]))
    db = env.get("NEO4J_DATABASE", "neo4j")
    with driver.session(database=db) as s:
        if args.undo:
            print(f"restored {run_batches(s, UNDO_SECTIONS, args.batch, args.pause):,} section abstracts")
            print(f"restored {run_batches(s, UNDO_VECTORS, args.batch, args.pause):,} parked vectors")
            print(f"restored {run_batches(s, UNDO_DOCS, args.batch, args.pause):,} document descriptions")
            return 0
        if args.purge_backups:
            for prop, query in PURGE.items():
                print(f"removed {prop} from {run_batches(s, query, args.batch, args.pause):,} nodes")
            return 0
        counts(s)
        if args.shapes:
            shapes(s, args.sample)
            return 0
        if not args.write:
            print("count only; --shapes shows the sample, --write rebuilds (old values kept as *_before_fix; --undo restores)")
            return 0
        _, _, dids = fix_sections(s, args)
        fix_descriptions(s, args, dids)
        print("after the fix:")
        counts(s)
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
