"""Sections whose abstract carries the ingest's error string instead of a summary: count them, and with --write
rebuild the abstract the way the ingest (flatten.py) builds it when a chunk has no analysis.

Background (7 Nov 2026): the analysis module's last resort for a chunk whose model call fails twice (structured,
then plain text: a 404 on the model name, a chunk timeout, any error) is a result whose "requirement" is the
literal "[Errore analisi comma]". flatten.py joins the chunk requirements of a section with "; " and makes the
abstract from them, so the string ends up in the abstract: alone ("<title> - [Errore analisi comma]") when every
chunk failed, mixed with real summaries ("<title> - <summary>; [Errore analisi comma] Testo significativo: ...")
when one chunk of several failed. The post-processing only replaces abstracts shorter than 20 characters. The
document's description is the first 500 words of its section abstracts, so it carries the string too. The chatbot
returns s.abstract for a section view, searches abstracts and descriptions for keywords, and the nightly embedding
job embeds abstract + plain_text.

Count on 8 Nov 2026: 37,510 sections in 5,993 rulings (the 2025 batch of 2-5 Nov without a served model, plus
rulings of other batches with single failed chunks).

The fix follows flatten.py's rule for a section: requirement = the good fragments joined with "; ", or the first
200 words of plain_text when none is left; then " Testo significativo: <core text>" when the abstract had one;
then the article title in front ("titolo - ") when it is not already in the text; trimmed to 200 words. The
description of every document whose description carries the string is rebuilt from its section abstracts in
reading order (NEXT), trimmed to 500 words. Nothing is printed but counts.

    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/fix_error_abstracts.py            # count only
    cd /opt/chatbot && sudo venv/bin/python -I /home/sunnit/fix_error_abstracts.py --write    # fix, in batches
    ... --write --reembed    # also park each fixed section's vector (embedding_v2_before_fix) and clear
                             # embedding_v2, so the 02:00 backfill re-embeds it from the new text. Until that job
                             # runs, those sections are absent from vector search: run the fix in the evening,
                             # or start the backfill by hand right after. Not needed for retrieval (the vectors
                             # differ from the new text by a few tokens); it is for text / vector consistency.
    ... --undo               # put back abstracts, descriptions and parked vectors
    ... --purge-backups      # after checking the result: remove the *_before_fix properties
The old values stay on the nodes as abstract_before_fix / description_before_fix / embedding_v2_before_fix until
--purge-backups, so --undo restores everything.
"""
import argparse
import sys
import time
from typing import Dict, List, Optional, Tuple

ERROR = "[Errore analisi comma]"
FILTER = "[Content filter"
CORE_MARK = " Testo significativo: "


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


def is_error(fragment: str) -> bool:
    return fragment == ERROR or fragment.startswith(FILTER)


def rebuild_abstract(abstract: str, plain_text: str, titolo: str) -> Tuple[str, bool]:
    """The abstract flatten.py would have built had the failed chunks produced nothing: (new abstract, pure).
    pure = True when nothing of the analysis survives (every fragment was an error string and there is no core
    text): the abstract then becomes the first 200 words of the text, as for a section without analysis."""
    titolo = (titolo or "").strip()
    body = abstract or ""
    if titolo and body.startswith(f"{titolo} - "):
        body = body[len(titolo) + 3:]
    core = ""
    if CORE_MARK in body:
        body, core = body.split(CORE_MARK, 1)
        core = core.strip()
    fragments = [f.strip() for f in body.split("; ")]
    good = [f for f in fragments if f and not is_error(f)]
    requirement = "; ".join(good)
    pure = not good and not core
    new = requirement if requirement else trim_words(plain_text, 200)
    if core:
        new = f"{new}{CORE_MARK}{core}".strip()
    if titolo and titolo not in new:
        new = f"{titolo} - {new}"
    new = trim_words(new, 200)
    if not new.strip():
        new = "(sezione senza testo)"           # never leave the error string, never write an empty abstract
    return new, pure


def chain_order(rows: List[dict]) -> List[dict]:
    """The document's sections in reading order: along NEXT from the head(s); sections off the chain (the split
    pieces name_i, which flatten.py keeps out of the description) are left out, unless the document has no chain
    at all (a single section), in which case every section is used in name order."""
    by_eid = {r["eid"]: r for r in rows}
    targets = {r["next"] for r in rows if r.get("next")}
    on_chain = [r for r in rows if r.get("next") or r["eid"] in targets]
    if not on_chain:
        return sorted(rows, key=lambda r: r.get("name") or "")
    heads = sorted((r for r in on_chain if r["eid"] not in targets), key=lambda r: r.get("name") or "")
    ordered, seen = [], set()
    for head in heads:
        cur: Optional[dict] = head
        while cur and cur["eid"] not in seen:
            seen.add(cur["eid"])
            ordered.append(cur)
            cur = by_eid.get(cur.get("next") or "")
    return ordered


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
PICK_SECTIONS = (
    "MATCH (s:Section) WHERE s.abstract CONTAINS $needle "
    "RETURN elementId(s) AS eid, s.abstract AS abstract, s.plain_text AS text, s.titolo_articolo AS titolo LIMIT $n"
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


def fix_sections(session, args) -> Tuple[int, int, int]:
    done = pure_n = parked = 0
    seen = set()
    t0 = time.time()
    while True:
        rows = session.run(PICK_SECTIONS, needle=ERROR, n=args.batch).data()
        if not rows or all(r["eid"] in seen for r in rows):      # nothing left, or a batch that did not change
            break
        seen.update(r["eid"] for r in rows)
        payload = []
        for r in rows:
            new, pure = rebuild_abstract(r["abstract"] or "", r["text"] or "", r["titolo"] or "")
            pure_n += pure
            payload.append({"eid": r["eid"], "abstract": new})
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
    return done, pure_n, parked


def fix_descriptions(session, args) -> int:
    done = 0
    seen = set()
    t0 = time.time()
    while True:
        eids = [r["eid"] for r in session.run(PICK_DOCS, needle=ERROR, n=args.batch)]
        if not eids or all(e in seen for e in eids):
            break
        seen.update(eids)
        payload = [{"eid": did, "description": rebuild_description(session.run(DOC_SECTIONS, did=did).data())}
                   for did in eids]
        session.run(WRITE_DOCS, rows=payload).consume()
        done += len(payload)
        time.sleep(args.pause)
    print(f"rebuilt {done:,} document descriptions in {time.time() - t0:.0f}s")
    return done


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--env", default="/opt/v2tools/health.env", help="NEO4J_URI / USER / PASSWORD / DATABASE")
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
        if not args.write:
            print("count only; --write rebuilds them (old values kept as *_before_fix; --undo restores)")
            return 0
        fix_sections(s, args)
        fix_descriptions(s, args)
        print("after the fix:")
        counts(s)
    driver.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
