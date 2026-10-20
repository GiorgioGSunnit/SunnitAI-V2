"""Builds the situation index (offline document expansion): for every article
of the main codes and laws, 3-5 short plain-Italian descriptions of concrete
situations it applies to, so a lay account of the facts ("ha preso il
portafoglio dalla borsa della collega") can reach an article whose official
wording shares none of its words ("impossessamento della cosa mobile altrui").

TIMELINE (scripts/exp_distill.py, round 10) showed the model decomposes the
facts fine but retrieval fails on the wording gap; this index is the other
side of that gap. exp_distill.py's SITUATION channel reads it.

Per article (base article name: "640", "640-ter", "112"; one per article,
fragments and commi are not indexed separately):
  - one LLM call, temperature 0, given the article's official heading
    (titolo_articolo) and text (plain_text, else abstract; first
    _TEXT_CHARS characters), returning 3-5 situations: no legal jargon, no
    article numbers, no code names;
  - each situation becomes a (:Situation {id, text, embedding, article,
    document, document_id, position, prompt_version, model}) node, linked
    (main Section)-[:HAS_SITUATION]->(:Situation);
  - embedded with the v2 embedding model (qwen3-embedding-8b, plain text,
    no instruction -- documents get none, the same way BE's
    backfill_embeddings_v2.py embeds Sections), into its own vector index
    situation_embeddings_v2, whose indexConfig is copied verbatim from
    section_embeddings_v2 (dimensions, similarity, SCALAR quantisation,
    HNSW settings: whatever SHOW VECTOR INDEXES reports for it).

The article's "main Section" is the Section named exactly as the base
article ("640"), which holds the whole article; when there is none, the
fragment with the longest text (_main_sections).

DRY RUN BY DEFAULT: reads the graph, counts what is left to do, then runs
the LLM and the embedding model on a sample (--sample, default 50 articles)
WITHOUT writing anything, prints a few of the generated situations and the
estimated total time measured on that sample. Only --write writes: the
Situation nodes, their HAS_SITUATION links, a uniqueness constraint on
Situation.id and the vector index.

Resumable: an article whose main Section already has situations of the
current PROMPT_VERSION is skipped; re-running after an interruption picks up
where it stopped. Changing PROMPT_VERSION regenerates everything (the old
situations of a regenerated Section are deleted in the same transaction).

Run from the app folder so it uses the app's code, .env and venv:
    cd /opt/chatbot && venv/bin/python scripts/build_situation_index.py                  # dry run: counts + 50-article sample + estimate
    cd /opt/chatbot && venv/bin/python scripts/build_situation_index.py --sample 0       # dry run: counts only, no LLM calls
    cd /opt/chatbot && venv/bin/python scripts/build_situation_index.py --write --limit 50 --doc "Codice Penale"   # a first real batch
    cd /opt/chatbot && venv/bin/python scripts/build_situation_index.py --write          # everything (resumable)
    ... --workers 4          # parallel LLM calls (default 4)
    ... --batch-size 32      # articles per write transaction / embedding request (default 32)
    ... --doc TEXT           # only primary documents whose name contains TEXT (any primary document, not just the default list)
    ... --limit N            # at most N articles in this run
"""

import argparse
import hashlib
import logging
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _path in (ROOT, os.getcwd()):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from dotenv import load_dotenv  # noqa: E402

load_dotenv(os.path.join(os.getcwd(), ".env"))

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

DEFAULT_DOCUMENTS = [
    "Codice Penale 2026",
    "Codice Civile 2026",
    "Codice di procedura civile Edizione 2026",
    "Codice di procedura penale 2026",
    "Codice del processo amministrativo 2025",
    "Codice del Consumo 2026",
    "L. 241/1990",
]
INDEX_NAME = "situation_embeddings_v2"
REFERENCE_INDEX = "section_embeddings_v2"
LABEL = "Situation"
REL = "HAS_SITUATION"
PROMPT_VERSION = "situations-v1"
_TEXT_CHARS = 4000
_TRUNCATE_TOKENS = 8000   # the embedding server's limit is 8192 tokens

_SYSTEM = (
    "Sei un avvocato italiano che spiega la legge a persone senza formazione giuridica. Ti do la rubrica e il "
    "testo di un articolo di legge. Scrivi da 3 a 5 descrizioni brevi (una frase ciascuna, al massimo 25 "
    "parole) di situazioni concrete della vita reale a cui questo articolo si applica, come le racconterebbe "
    "una persona comune: chi fa cosa, a chi, con quale conseguenza. Situazioni diverse tra loro. Niente "
    "linguaggio giuridico, niente numeri di articolo, niente nomi di codici o di leggi. Restituisci SOLO un "
    "array JSON di stringhe."
)

_EXTENSIONS = (r"bis|ter|quater|quinquies|sexies|septies|octies|novies|decies|undecies|duodecies|terdecies|"
               r"quaterdecies|quattuordecies|quinquiesdecies|quindecies|sexiesdecies|sexdecies|septiesdecies|"
               r"septendecies|octiesdecies|octodecies|noviesdecies|novendecies|vicies")
# "640", "640-ter", "640-ter.1.0", "640_2" -> base "640" / "640-ter". Names
# with an annex/decree prefix ("all2-1", "decreto-1") are not main articles.
_BASE_ARTICLE_RE = re.compile(r"^(\d+(?:-(?:" + _EXTENSIONS + r"))?)(?=$|[._])", re.IGNORECASE)
# A situation that slipped into citing the law instead of describing a situation.
_LEGAL_REF_RE = re.compile(
    r"\bart(?:t|icol[oi])?\.?\s*\d|\bcomma\s+\d|\bc\.\s*[cp]\.|\bcodice\b|\bd\.\s*lgs|\blegge\s+(?:n\.\s*)?\d",
    re.IGNORECASE)
_TAG_PREFIX_RE = re.compile(r"^\s*\[[^\]]*\]\s*")


# --- Pure helpers (unit-tested in tests/test_build_situation_index.py) --------

def _base_article(section_name):
    m = _BASE_ARTICLE_RE.match((section_name or "").strip())
    return m.group(1).lower() if m else None


def _section_text(row):
    return (row.get("text") or row.get("abstract") or "").strip()


def _main_sections(rows):
    """{base article: its main Section row} from one document's Section rows.
    The Section named exactly as the base article wins (it holds the whole
    article); otherwise the fragment with the longest text. Rows with no
    text, or whose name is not an article, are ignored."""
    by_base = {}
    for row in rows:
        base = _base_article(row.get("name"))
        if base and _section_text(row):
            by_base.setdefault(base, []).append(row)
    main = {}
    for base, candidates in by_base.items():
        exact = [r for r in candidates if (r.get("name") or "").strip().lower() == base]
        main[base] = exact[0] if exact else max(candidates, key=lambda r: len(_section_text(r)))
    return main


def _normalise_name(name):
    return " ".join(_TAG_PREFIX_RE.sub("", name or "").lower().split())


def _resolve_documents(targets, documents):
    """{target: [matching documents]} -- a document matches when its name,
    without a leading [TAG] and ignoring case and spacing, equals the target
    or (failing any exact match) contains it. The caller decides what to do
    with zero or several matches."""
    resolved = {}
    for target in targets:
        t = _normalise_name(target)
        exact = [d for d in documents if _normalise_name(d["name"]) == t]
        resolved[target] = exact or [d for d in documents if t in _normalise_name(d["name"])]
    return resolved


def _parse_situations(parsed):
    """The model's reply -> 3-5 distinct situations (at most 5; fewer than 3
    are kept as they are -- better than none; the caller logs them). Drops
    anything that isn't a non-empty string or that cites a law/article."""
    if not isinstance(parsed, list):
        return []
    seen, out = set(), []
    for item in parsed:
        if not isinstance(item, str):
            continue
        text = " ".join(item.split())
        key = text.lower()
        if not text or key in seen or _LEGAL_REF_RE.search(text):
            continue
        seen.add(key)
        out.append(text)
    return out[:5]


def _situation_id(document_id, article, position):
    return hashlib.sha1(f"{document_id}|{article}|{position}|{PROMPT_VERSION}".encode("utf-8")).hexdigest()


def _cypher_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("\\", "\\\\").replace("'", "\\'") + "'"


def _index_statement(index_config):
    """CREATE VECTOR INDEX for Situation.embedding with exactly this
    indexConfig (a dict as SHOW VECTOR INDEXES returns it)."""
    entries = ", ".join(f"`{key}`: {_cypher_value(value)}" for key, value in sorted(index_config.items()))
    return (f"CREATE VECTOR INDEX {INDEX_NAME} IF NOT EXISTS FOR (x:{LABEL}) ON (x.embedding) "
            f"OPTIONS {{indexConfig: {{{entries}}}}}")


# --- The v2 embedding model, plain text ---------------------------------------

def plain_text_embedder():
    """qwen3-embedding-8b through its OpenAI-compatible endpoint, sending
    the text itself -- never tiktoken token IDs, whatever EMBEDDING_QUERY_MODE
    the app runs with (with LangChain's defaults the server would read
    cl100k_base token IDs with its own vocabulary). Same settings as BE's
    backfill_embeddings_v2.py, which built section_embeddings_v2."""
    from langchain_openai import OpenAIEmbeddings
    model = os.getenv("EMBEDDING_MODEL") or os.getenv("LLM_EMBEDDING_MODEL")
    if not model:
        sys.exit("No embedding model: set EMBEDDING_MODEL")
    return OpenAIEmbeddings(
        model=model,
        api_key=os.getenv("EMBEDDING_API_KEY") or os.getenv("LLM_API_KEY") or "EMPTY",
        base_url=os.getenv("EMBEDDING_BASE_URL") or os.getenv("LLM_EMBEDDING_BASE_URL"),
        check_embedding_ctx_length=False,
        model_kwargs={"extra_body": {"truncate_prompt_tokens": _TRUNCATE_TOKENS}},
        timeout=60.0,
        max_retries=2,
    )


# --- Graph reads/writes ---------------------------------------------------------

_DOCUMENTS_QUERY = (
    "MATCH (d:Document) WHERE d.document_type = 'primary' "
    "RETURN d.id AS id, d.name AS name ORDER BY d.name"
)
_SECTIONS_QUERY = (
    "MATCH (d:Document)-[:CONTAINS]->(s:Section) WHERE d.id = $doc_id "
    "RETURN elementId(s) AS eid, s.name AS name, s.titolo_articolo AS titolo, "
    "s.plain_text AS text, s.abstract AS abstract, "
    "EXISTS { MATCH (s)-[:" + REL + "]->(x:" + LABEL + ") WHERE x.prompt_version = $version } AS done"
)
_DELETE_OLD_QUERY = (
    "UNWIND $eids AS eid MATCH (s)-[:" + REL + "]->(old:" + LABEL + ") WHERE elementId(s) = eid "
    "DETACH DELETE old"
)
_WRITE_QUERY = (
    "UNWIND $rows AS row MATCH (s) WHERE elementId(s) = row.eid "
    "UNWIND row.items AS item "
    "MERGE (x:" + LABEL + " {id: item.id}) "
    "SET x.text = item.text, x.embedding = item.embedding, x.position = item.position, "
    "x.article = row.article, x.document = row.document, x.document_id = row.document_id, "
    "x.prompt_version = $version, x.model = $model "
    "MERGE (s)-[:" + REL + "]->(x)"
)
_CONSTRAINT_STATEMENT = f"CREATE CONSTRAINT situation_id IF NOT EXISTS FOR (x:{LABEL}) REQUIRE x.id IS UNIQUE"


def _reference_index_config(driver, database):
    with driver.session(database=database) as session:
        rows = session.run("SHOW VECTOR INDEXES YIELD name, options WHERE name = $name RETURN options",
                           name=REFERENCE_INDEX).data()
    if not rows:
        return None
    return (rows[0].get("options") or {}).get("indexConfig") or None


def _collect_todo(driver, database, documents, limit):
    """[(document, article, Section row)] still without current situations, and per-document counts."""
    todo, counts = [], []
    with driver.session(database=database) as session:
        for doc in documents:
            rows = session.run(_SECTIONS_QUERY, doc_id=doc["id"], version=PROMPT_VERSION).data()
            main = _main_sections(rows)
            pending = [(doc, base, row) for base, row in main.items() if not row.get("done")]
            counts.append((doc["name"], len(rows), len(main), len(pending)))
            todo.extend(pending)
    return (todo[:limit] if limit else todo), counts


# --- Generation -----------------------------------------------------------------

def _generate(article_row):
    """3-5 situations for one article -- one LLM call, temperature 0."""
    import src.rag.defensive_generation as dg
    from langchain_core.messages import HumanMessage, SystemMessage
    titolo = (article_row.get("titolo") or "").strip()
    human = f"RUBRICA: {titolo or '(nessuna)'}\n\nTESTO:\n{_section_text(article_row)[:_TEXT_CHARS]}"
    parsed = dg._chat_json([SystemMessage(content=_SYSTEM), HumanMessage(content=human)],
                           max_tokens=600, temperature=0)
    return _parse_situations(parsed)


def _process_batch(batch, embedder, workers):
    """[(document, article, row, situations, vectors)] for one batch: LLM calls
    in parallel, then one embedding request for every situation in the batch.
    An article whose LLM call fails is left out (it stays pending)."""
    def generate(item):
        doc, base, row = item
        try:
            return item, _generate(row)
        except Exception as exc:
            logging.warning("%s art. %s: generation failed, left pending: %s", doc["name"], base, exc)
            return item, []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        generated = list(pool.map(generate, batch))
    for (doc, base, _), situations in generated:
        if 0 < len(situations) < 3:
            logging.warning("%s art. %s: only %d situation(s) usable", doc["name"], base, len(situations))
    texts = [text for _, situations in generated for text in situations]
    vectors = embedder.embed_documents(texts) if texts else []
    out, i = [], 0
    for (doc, base, row), situations in generated:
        if situations:
            out.append((doc, base, row, situations, vectors[i:i + len(situations)]))
        i += len(situations)
    return out


def _write_batch(driver, database, results, model):
    rows = [{
        "eid": row["eid"], "article": base, "document": doc["name"], "document_id": doc["id"],
        "items": [{"id": _situation_id(doc["id"], base, pos), "text": text, "embedding": list(vec), "position": pos}
                  for pos, (text, vec) in enumerate(zip(situations, vectors), start=1)],
    } for doc, base, row, situations, vectors in results]

    def tx(t):
        t.run(_DELETE_OLD_QUERY, eids=[r["eid"] for r in rows])
        t.run(_WRITE_QUERY, rows=rows, version=PROMPT_VERSION, model=model)

    with driver.session(database=database) as session:
        session.execute_write(tx)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--write", action="store_true", help="actually write (default: dry run)")
    parser.add_argument("--doc", help="only primary documents whose name contains this text")
    parser.add_argument("--limit", type=int, help="at most this many articles in this run")
    parser.add_argument("--sample", type=int, default=50, help="dry run: articles to time without writing (0: none)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4, help="parallel LLM calls")
    args = parser.parse_args()

    from src.rag import embedding_config
    from src.rag.main import NEO4J_DATABASE, driver

    # This script always embeds plain text; the mode matters for how the app
    # (and exp_distill.py's other vector channels) query the _v2 indexes.
    print(f"Live EMBEDDING_QUERY_MODE (this .env): {embedding_config.QUERY_MODE}")
    if not embedding_config.text_mode():
        print("WARNING: not 'text' -- the app's vector queries send token IDs against plain-text _v2 vectors.")

    with driver.session(database=NEO4J_DATABASE) as session:
        primary = session.run(_DOCUMENTS_QUERY).data()
    if args.doc:
        needle = _normalise_name(args.doc)
        documents = [d for d in primary if needle in _normalise_name(d["name"])]
        if not documents:
            parser.error(f"no primary document's name contains {args.doc!r}")
    else:
        documents = []
        for target, matches in _resolve_documents(DEFAULT_DOCUMENTS, primary).items():
            if len(matches) == 1:
                documents.append(matches[0])
            else:
                print(f"SKIPPED {target!r}: {len(matches)} primary documents match "
                      f"{[m['name'] for m in matches] or ''} -- pass --doc with the exact name to include it")
    documents = list({d["id"]: d for d in documents}.values())

    todo, counts = _collect_todo(driver, NEO4J_DATABASE, documents, args.limit)
    print(f"{'document':<55s}{'sections':>10s}{'articles':>10s}{'pending':>10s}")
    for name, sections, articles, pending in counts:
        print(f"{name[:54]:<55s}{sections:>10d}{articles:>10d}{pending:>10d}")
    print(f"\n{len(todo)} article(s) to process in this run (prompt {PROMPT_VERSION}).")

    reference = _reference_index_config(driver, NEO4J_DATABASE)
    if reference is None:
        sys.exit(f"Could not read {REFERENCE_INDEX!r}'s indexConfig (SHOW VECTOR INDEXES) -- not guessing it.")
    statement = _index_statement(reference)
    print(f"\nIndex statement (indexConfig copied from {REFERENCE_INDEX!r}):\n  {statement}")

    embedder = plain_text_embedder()
    model = embedder.model
    print(f"Embedding model: {model} at {embedder.openai_api_base or '(default endpoint)'}, plain text.")
    probe = embedder.embed_documents(["prova"])[0]
    dimensions = reference.get("vector.dimensions")
    if dimensions and len(probe) != dimensions:
        sys.exit(f"The embedding model returns {len(probe)} dimensions, {REFERENCE_INDEX} has {dimensions}: "
                 "not the v2 model -- check EMBEDDING_MODEL/EMBEDDING_BASE_URL.")

    if not args.write:
        sample = todo[:args.sample]
        if not sample:
            print("\nDry run: nothing sampled. Pass --write to actually build the index.")
            return
        print(f"\nDry run: timing {len(sample)} article(s) with {args.workers} worker(s) -- LLM + embedding, "
              "nothing is written...")
        t0, done = time.time(), []
        for i in range(0, len(sample), args.batch_size):
            done += _process_batch(sample[i:i + args.batch_size], embedder, args.workers)
        elapsed = time.time() - t0
        per_article = elapsed / len(sample)
        n_situations = sum(len(r[3]) for r in done)
        print(f"{len(done)}/{len(sample)} article(s) produced situations ({n_situations} in all, "
              f"{n_situations / max(len(done), 1):.1f} per article) in {elapsed:.0f}s = "
              f"{per_article:.2f}s per article at {args.workers} worker(s).")
        print(f"ESTIMATE for the {len(todo)} pending article(s): {len(todo) * per_article / 60:.0f} minute(s) "
              f"({len(todo) * per_article / 3600:.1f} h), measured on this sample; Neo4j writes not included "
              "(one transaction per batch, small next to the LLM calls).")
        print("\nExamples:")
        for doc, base, row, situations, _ in done[:8]:
            print(f"  {doc['name']} art. {base} -- {(row.get('titolo') or '').strip()}")
            for text in situations:
                print(f"     - {text}")
        print("\nDry run only -- nothing was written. Pass --write to build the index.")
        return

    with driver.session(database=NEO4J_DATABASE) as session:
        session.run(_CONSTRAINT_STATEMENT)
        session.run(statement)
    written, t0 = 0, time.time()
    for i in range(0, len(todo), args.batch_size):
        batch = todo[i:i + args.batch_size]
        try:
            results = _process_batch(batch, embedder, args.workers)
        except Exception as exc:
            print(f"batch starting at {i} failed, left pending: {exc}", file=sys.stderr)
            continue
        if results:
            _write_batch(driver, NEO4J_DATABASE, results, model)
        written += len(results)
        elapsed = time.time() - t0
        eta = elapsed / (i + len(batch)) * (len(todo) - i - len(batch))
        print(f"... {i + len(batch)}/{len(todo)} processed, {written} written "
              f"({elapsed:.0f}s elapsed, ~{eta / 60:.0f} min left)", flush=True)
    print(f"\nWrote situations for {written}/{len(todo)} article(s); the rest stay pending for the next run. "
          f"Index {INDEX_NAME} exists (it fills as nodes are written).")


if __name__ == "__main__":
    main()
