"""Builds the heading-meaning index HEADING_INDEX_ENABLED uses (src/rag/
defensive_generation.py's _heading_index_match): a vector index over every
primary Section's titolo_articolo (its official rubrica), so an institution
that shares no words at all with its rubrica ("tentativo" for art. 56's
"Delitto tentato") can still resolve by meaning instead of by word/stem
overlap (see HEADING_INDEX_ENABLED's own docstring in defensive_generation.py
for why article_by_title structurally can't reach that case).

THIS WRITES TO THE DATABASE: a new s.heading_embedding property on every
primary Section with a titolo_articolo, and (at the end) a new vector
index. Default is a DRY RUN that only counts and estimates -- pass --write
to actually embed and write. Run against a staging/test Neo4j first if you
have one.

Run from the app folder so it uses the app's code, .env and venv:
    cd /opt/chatbot && venv/bin/python scripts/build_heading_index.py              # dry run: counts + estimate
    cd /opt/chatbot && venv/bin/python scripts/build_heading_index.py --write      # embed + write + create the index
    ... --batch-size 32                 # smaller batches if the embedding endpoint rate-limits
    ... --dimensions 1536 --similarity cosine   # skip auto-detecting these from --reference-index

Size: the index holds one float32 vector per primary Section with a
titolo_articolo -- this script prints the real count before writing
anything (expect a few thousand to a few tens of thousands, depending on
how many primary codes/laws are in the corpus); size in bytes is
count * dimensions * 4. Time: this script prints how many titles still
need an embedding and a labelled GUESS at wall-clock time (not a
measurement -- there was no live access to time an actual embedding call
when this script was written); the real number depends on the embedding
endpoint's own latency and rate limits, which this script does not
control. Added latency to a live request, once the index exists and
HEADING_INDEX_ENABLED is on: at most one extra embedding call per
institution that doesn't resolve by exact title match (see
_heading_index_match's docstring) -- bounded, not zero.
"""

import argparse
import logging
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")

_COUNT_QUERY = (
    "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.document_type = 'primary' "
    "AND s.titolo_articolo IS NOT NULL AND s.titolo_articolo <> '' "
    "RETURN count(s) AS total, count(CASE WHEN s.heading_embedding IS NULL THEN 1 END) AS missing"
)
_FETCH_MISSING_QUERY = (
    "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.document_type = 'primary' "
    "AND s.titolo_articolo IS NOT NULL AND s.titolo_articolo <> '' AND s.heading_embedding IS NULL "
    "RETURN elementId(s) AS eid, s.titolo_articolo AS titolo_articolo"
)
_WRITE_QUERY = (
    "UNWIND $rows AS row MATCH (s) WHERE elementId(s) = row.eid SET s.heading_embedding = row.embedding"
)


def _detect_index_config(driver, database, reference_index):
    """{"dimensions": int, "similarity": str} read from an existing vector
    index's own options, or None if that couldn't be determined -- in
    which case the caller needs --dimensions/--similarity passed
    explicitly rather than guessed."""
    try:
        with driver.session(database=database) as session:
            rows = session.run(
                "SHOW VECTOR INDEXES YIELD name, options WHERE name = $name RETURN options",
                name=reference_index,
            ).data()
    except Exception as exc:
        logging.warning("could not read vector index options via SHOW VECTOR INDEXES: %s", exc)
        return None
    if not rows:
        return None
    config = (rows[0].get("options") or {}).get("indexConfig") or {}
    dimensions = config.get("vector.dimensions")
    similarity = config.get("vector.similarity_function")
    if not dimensions or not similarity:
        return None
    return {"dimensions": dimensions, "similarity": similarity}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="actually embed and write (default: dry run)")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--reference-index", default=os.getenv("CASE_POOL_VECTOR_INDEX", "section_embeddings_v2"),
                        help="existing vector index to copy dimensions/similarity from")
    parser.add_argument("--dimensions", type=int, help="override auto-detected dimensions")
    parser.add_argument("--similarity", choices=["cosine", "euclidean"], help="override auto-detected similarity")
    args = parser.parse_args()

    import src.rag.defensive_generation as dg
    from src.rag.main import NEO4J_DATABASE, driver

    with driver.session(database=NEO4J_DATABASE) as session:
        counts = session.run(_COUNT_QUERY).single()
    total, missing = counts["total"], counts["missing"]
    print(f"{total} primary Sections have a titolo_articolo; {missing} still need an embedding "
         f"(index name: {dg._HEADING_VECTOR_INDEX}).")
    if missing:
        print(f"Rough, UNMEASURED guess: at roughly 60 titles/s batched, embedding would take about "
             f"{missing / 60 / 60:.1f} minute(s). This is a guess, not a timing -- the real number "
             f"depends entirely on the embedding endpoint's own latency and rate limits.")
        print(f"Index size once built: {total} vectors * dimensions * 4 bytes "
             f"(e.g. {total * 1536 * 4 / 1e6:.1f} MB at 1536 dimensions, {total * 384 * 4 / 1e6:.1f} MB at 384).")

    dimensions, similarity = args.dimensions, args.similarity
    if dimensions is None or similarity is None:
        detected = _detect_index_config(driver, NEO4J_DATABASE, args.reference_index)
        if detected is None:
            parser.error(f"could not read {args.reference_index!r}'s own dimensions/similarity "
                        "(SHOW VECTOR INDEXES) -- pass --dimensions and --similarity explicitly")
        dimensions = dimensions or detected["dimensions"]
        similarity = similarity or detected["similarity"]

    create_statement = (
        f"CREATE VECTOR INDEX {dg._HEADING_VECTOR_INDEX} IF NOT EXISTS "
        f"FOR (s:Section) ON (s.heading_embedding) "
        f"OPTIONS {{indexConfig: {{`vector.dimensions`: {dimensions}, "
        f"`vector.similarity_function`: '{similarity}'}}}}"
    )
    print(f"\nIndex statement (dimensions/similarity copied from {args.reference_index!r}):\n  {create_statement}")

    if not args.write:
        print("\nDry run only -- nothing was embedded or written. Pass --write to actually do it.")
        return

    if missing == 0:
        print("\nNothing to embed. Creating the index (if it doesn't exist yet)...")
    else:
        print(f"\nFetching the {missing} Sections that still need an embedding...")
        with driver.session(database=NEO4J_DATABASE) as session:
            todo = session.run(_FETCH_MISSING_QUERY).data()

        print(f"Embedding {len(todo)} headings in batches of {args.batch_size}...")
        written, t0 = 0, time.time()
        for i in range(0, len(todo), args.batch_size):
            batch = todo[i:i + args.batch_size]
            texts = [row["titolo_articolo"] for row in batch]
            try:
                vectors = dg.embedding_model.embed_documents(texts)
            except Exception as exc:
                print(f"embedding failed for batch starting at index {i}, skipping it: {exc}", file=sys.stderr)
                continue
            with driver.session(database=NEO4J_DATABASE) as session:
                session.run(_WRITE_QUERY, rows=[{"eid": row["eid"], "embedding": list(vec)}
                                               for row, vec in zip(batch, vectors)])
            written += len(batch)
            print(f"... {written}/{len(todo)} ({time.time() - t0:.0f}s elapsed)", flush=True)
        print(f"\nEmbedded and wrote {written}/{len(todo)}. Creating the index (if it doesn't exist yet)...")

    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(create_statement)
    print(f"Done. Set {dg.HEADING_INDEX_FLAG}=true to use it.")


if __name__ == "__main__":
    main()
