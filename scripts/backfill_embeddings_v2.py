"""
Backfill script: re-embed nodes with the plain-text client (no tiktoken token IDs).

Default (parallel plan): writes the new vector to `embedding_v2` and leaves `embedding` untouched.
With --property embedding (in-place plan): overwrites `embedding` and sets `embedding_version`.

Text per node: abstract + " " + plain_text, stripped (the same text BE's post-processing embeds).
A node is re-embedded when plain_text OR abstract is non-empty, so abstract-only Sections keep a
vector too. Labels: Section, CommentatoSection, FiscalitaSection (same fields).

Resumable: only nodes without the new vector are picked (embedding_v2 IS NULL, or for
--property embedding an embedding_version different from EMBEDDING_VERSION). It can be stopped
and run again at any time.

Long texts: the server cuts anything over TRUNCATE_TOKENS (vLLM truncate_prompt_tokens) instead of
rejecting the request. If a batch still fails, its texts are retried one by one; only texts that
fail on their own are skipped, and their element ids are written to --failed-file.

The ids to process are collected once per label; texts are then fetched and vectors written in
batches (one UNWIND per batch), so the graph is not re-scanned for every batch.

Usage (from /opt/sunnitai-be with the venv; .env is loaded automatically if present):
    venv/bin/python backfill_embeddings_v2.py --dry-run
    venv/bin/python backfill_embeddings_v2.py --labels Section --doc "Codice Civile" --limit 50
    venv/bin/python backfill_embeddings_v2.py --create-index

Options:
    --labels        comma-separated labels (default: Section,CommentatoSection,FiscalitaSection)
    --property      embedding_v2 (default, parallel plan) | embedding (in-place plan)
    --batch-size    texts per embedding request (default 64)
    --sleep         seconds to pause between batches, to go easy on the shared GPU and Neo4j (default 0)
    --limit         at most N nodes per label (for tests)
    --doc           only Sections of documents whose name or hash contains this text
    --create-index  create the <index>_v2 vector index for each label if missing (parallel plan only)
    --dry-run       only count what would be done; no writes
    --failed-file   file that receives the element ids that could not be embedded
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

DEFAULT_LABELS = ["Section", "CommentatoSection", "FiscalitaSection"]
V2_INDEX = {
    "Section": "section_embeddings_v2",
    "CommentatoSection": "commentato_section_embeddings_v2",
    "FiscalitaSection": "fiscalita_section_embeddings_v2",
}
EMBEDDING_VERSION = "qwen3-8b-text-v1"   # set on nodes by --property embedding
EXPECTED_DIM = 4096                      # qwen3-embedding-8b; every vector index is 4096
TRUNCATE_TOKENS = 8000                   # server limit is 8192 tokens
HAS_TEXT = "(coalesce(n.plain_text, '') <> '' OR coalesce(n.abstract, '') <> '')"


def _load_env() -> None:
    """Load .env from the script's folder or its parent; never overrides variables already set."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve().parent
    for candidate in (here / ".env", here.parent / ".env"):
        if candidate.exists():
            load_dotenv(candidate)
            return


def _get_driver():
    from neo4j import GraphDatabase
    uri = os.environ["NEO4J_URI"]
    user = os.environ.get("NEO4J_USER", "neo4j")
    password = os.environ["NEO4J_PASSWORD"]
    return GraphDatabase.driver(uri, auth=(user, password), notifications_min_severity="OFF")


def _get_embedder():
    """Plain-text embedding client — no tiktoken tokenization."""
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(
        model=os.environ["LLM_EMBEDDING_MODEL"],
        api_key=os.environ.get("LLM_EMBEDDING_API_KEY") or os.environ.get("LLM_API_KEY", "EMPTY"),
        base_url=os.environ.get("LLM_EMBEDDING_BASE_URL") or os.environ.get("LLM_BASE_URL"),
        check_embedding_ctx_length=False,  # send the text itself, not OpenAI token IDs
        # let the server cut texts over its limit instead of rejecting the whole request
        model_kwargs={"extra_body": {"truncate_prompt_tokens": TRUNCATE_TOKENS}},
        timeout=60.0,  # fail fast instead of hanging for minutes
        max_retries=2,
    )


def _pending_condition(prop: str) -> str:
    if prop == "embedding_v2":
        return f"n.embedding_v2 IS NULL AND {HAS_TEXT}"
    return f"coalesce(n.embedding_version, '') <> $version AND {HAS_TEXT}"


def _counts(session, label: str, prop: str) -> tuple[int, int]:
    done = ("n.embedding_v2 IS NOT NULL" if prop == "embedding_v2"
            else "n.embedding_version = $version")
    r = session.run(
        f"MATCH (n:`{label}`) RETURN count(n) AS total, "
        f"count(CASE WHEN {done} THEN 1 END) AS done",
        version=EMBEDDING_VERSION,
    ).single()
    return r["total"], r["done"]


def _pending_ids(session, label: str, prop: str, doc: str | None, limit: int | None) -> list[str]:
    if doc:
        match = ("MATCH (d:Document)-[:CONTAINS]->(n:Section) "
                 "WHERE (d.name CONTAINS $doc OR d.hash CONTAINS $doc) AND ")
    else:
        match = f"MATCH (n:`{label}`) WHERE "
    query = match + _pending_condition(prop) + " RETURN DISTINCT elementId(n) AS eid"
    if limit:
        query += " LIMIT $limit"
    return [r["eid"] for r in session.run(query, doc=doc, limit=limit, version=EMBEDDING_VERSION)]


def _ensure_index(session, label: str) -> None:
    name = V2_INDEX[label]
    session.run(
        f"CREATE VECTOR INDEX {name} IF NOT EXISTS FOR (n:`{label}`) ON (n.embedding_v2) "
        f"OPTIONS {{ indexConfig: {{ `vector.dimensions`: {EXPECTED_DIM}, "
        f"`vector.similarity_function`: 'cosine' }} }}"
    ).consume()
    logger.info("Vector index %s ready (created if it was missing)", name)


def _index_exists(session, name: str) -> bool:
    rows = session.run("SHOW INDEXES YIELD name WHERE name = $name RETURN name", name=name).data()
    return bool(rows)


def _embed(embedder, texts: list[str]) -> list:
    """Embed a batch; if the request fails, retry each text alone. Failed texts -> None."""
    try:
        return embedder.embed_documents(texts)
    except Exception as exc:
        if len(texts) == 1:
            logger.error("Text failed on its own (%d chars): %s", len(texts[0]), str(exc)[:200])
            return [None]
        logger.warning("Batch of %d failed (%s) — retrying one by one", len(texts), str(exc)[:200])
    vectors = []
    for text in texts:
        try:
            vectors.append(embedder.embed_documents([text])[0])
        except Exception as exc:
            logger.error("Text failed on its own (%d chars): %s", len(text), str(exc)[:200])
            vectors.append(None)
    return vectors


def _write(session, rows: list[dict], prop: str) -> None:
    if prop == "embedding_v2":
        query = "UNWIND $rows AS row MATCH (n) WHERE elementId(n) = row.eid SET n.embedding_v2 = row.emb"
    else:
        query = ("UNWIND $rows AS row MATCH (n) WHERE elementId(n) = row.eid "
                 "SET n.embedding = row.emb, n.embedding_version = $version")
    session.run(query, rows=rows, version=EMBEDDING_VERSION).consume()


def backfill_label(driver, embedder, database: str, label: str, args) -> tuple[int, list[str]]:
    with driver.session(database=database) as session:
        total, done = _counts(session, label, args.property)
        ids = _pending_ids(session, label, args.property, args.doc, args.limit)
    logger.info("%s: %d nodes, %d already done, %d to embed now", label, total, done, len(ids))
    if args.dry_run or not ids:
        return 0, []

    if args.property == "embedding_v2" and args.create_index:
        with driver.session(database=database) as session:
            _ensure_index(session, label)

    written, failed = 0, []
    t_start = time.perf_counter()
    for start in range(0, len(ids), args.batch_size):
        chunk = ids[start:start + args.batch_size]
        with driver.session(database=database) as session:
            rows = session.run(
                "UNWIND $eids AS eid MATCH (n) WHERE elementId(n) = eid "
                "RETURN eid, coalesce(n.abstract, '') AS abstract, coalesce(n.plain_text, '') AS plain_text",
                eids=chunk,
            ).data()
        texts = [(r["abstract"] + " " + r["plain_text"]).strip() for r in rows]
        vectors = _embed(embedder, texts)

        good, bad = [], []
        for r, v in zip(rows, vectors):
            if v and len(v) == EXPECTED_DIM:
                good.append({"eid": r["eid"], "emb": v})
            else:
                bad.append(r["eid"])

        write_failed = []
        if good:
            with driver.session(database=database) as session:
                try:
                    _write(session, good, args.property)
                except Exception as exc:
                    logger.warning("Batch write failed (%s) — writing one by one", str(exc)[:200])
                    for row in good:
                        try:
                            _write(session, [row], args.property)
                        except Exception as exc_one:
                            logger.error("Write failed for %s: %s", row["eid"], str(exc_one)[:200])
                            write_failed.append(row["eid"])
        written += len(good) - len(write_failed)
        failed.extend(bad + write_failed)

        batch_no = start // args.batch_size + 1
        if batch_no % 10 == 0 or start + args.batch_size >= len(ids):
            elapsed = time.perf_counter() - t_start
            rate = (written + len(failed)) / elapsed if elapsed > 0 else 0
            eta = (len(ids) - written - len(failed)) / rate if rate > 0 else 0
            logger.info("%s: %d/%d written (%.1f/s) — failed: %d — ETA: %.0f min",
                        label, written, len(ids), rate, len(failed), eta / 60)
        if args.sleep:
            time.sleep(args.sleep)

    logger.info("%s done — written: %d, failed: %d, %.1f min",
                label, written, len(failed), (time.perf_counter() - t_start) / 60)
    return written, failed


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-embed nodes with the plain-text client.")
    parser.add_argument("--labels", default=",".join(DEFAULT_LABELS))
    parser.add_argument("--property", choices=["embedding_v2", "embedding"], default="embedding_v2")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--doc", default=None)
    parser.add_argument("--create-index", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--failed-file", default="/tmp/backfill_failed.txt")
    args = parser.parse_args()

    _load_env()
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    unknown = [l for l in labels if l not in V2_INDEX]
    if unknown:
        parser.error(f"unsupported label(s): {', '.join(unknown)}")
    if args.doc and labels != ["Section"]:
        parser.error("--doc only works with --labels Section")
    if args.create_index and args.property != "embedding_v2":
        parser.error("--create-index is only for the parallel plan (--property embedding_v2)")

    driver = _get_driver()
    embedder = None if args.dry_run else _get_embedder()
    database = os.environ.get("NEO4J_DATABASE", "neo4j")
    logger.info("Starting backfill — property=%s labels=%s dry_run=%s batch_size=%d",
                args.property, labels, args.dry_run, args.batch_size)

    all_failed: list[tuple[str, str]] = []
    total_written = 0
    try:
        for label in labels:
            written, failed = backfill_label(driver, embedder, database, label, args)
            total_written += written
            all_failed.extend((label, eid) for eid in failed)

        if args.property == "embedding_v2":
            with driver.session(database=database) as session:
                for label in labels:
                    if not _index_exists(session, V2_INDEX[label]):
                        logger.warning("Index %s does not exist yet: V2 can't search %s.embedding_v2 "
                                       "until it is created (run again with --create-index)",
                                       V2_INDEX[label], label)
    finally:
        driver.close()

    if all_failed:
        try:
            Path(args.failed_file).write_text("".join(f"{l}\t{e}\n" for l, e in all_failed), encoding="utf-8")
            logger.info("%d nodes could not be embedded — listed in %s", len(all_failed), args.failed_file)
        except OSError as exc:
            logger.warning("Could not write %s (%s); first failed ids: %s",
                           args.failed_file, exc, [e for _, e in all_failed[:20]])
    logger.info("Backfill finished — written: %d, failed: %d", total_written, len(all_failed))
    return 0


if __name__ == "__main__":
    sys.exit(main())
