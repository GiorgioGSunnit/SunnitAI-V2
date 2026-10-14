"""Generate and write vector embeddings for Neo4j nodes missing them.

Targets the PascalCase labels used by the RAG pipeline, except the section
labels (BE_OWNED_LABELS): their vectors are written by BE.
Run after write_kg_from_extracted has populated nodes.
"""

import argparse
import logging
import os
from typing import Dict, List

from dotenv import load_dotenv

load_dotenv()

try:  # reads the env, so after load_dotenv
    from ..rag import embedding_config  # noqa: E402
except ImportError:  # run as a file rather than as src.preprocessing.generate_embeddings
    from src.rag import embedding_config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

BATCH_SIZE = 64
# Every vector index in production is 4096 (qwen3-embedding-8b).
VECTOR_DIMENSIONS = int(os.getenv("EMBEDDING_DIMENSIONS", "4096"))

# Labels whose vectors BE owns: written on upload, gaps filled by BE's backfill.
# This script would embed a different text and overwrite `embedding`, the old
# vectors a rollback depends on, so it never touches them.
BE_OWNED_LABELS = ("Section", "CommentatoSection", "FiscalitaSection")
BE_OWNED_MESSAGE = (
    "Section, CommentatoSection and FiscalitaSection vectors are written by BE on upload "
    "(post_process step 3); fill any gaps with BE's scripts/backfill_embeddings_v2.py, not this script."
)


def _add_doc_prefix(text: str) -> str:
    """Add instruction prefix for qwen embedding models (token_ids mode only:
    in text mode documents get no instruction, as Qwen recommends)."""
    if embedding_config.text_mode():
        return text
    provider = os.getenv("EMBEDDING_PROVIDER", "local")
    model = os.getenv("EMBEDDING_MODEL", "")
    if provider == "openai" and "qwen" in model.lower():
        return f"Instruct: Represent this Italian legal document section for retrieval\n{text}"
    return text


def _text_expr(*props: str) -> str:
    """Cypher expression that concatenates non-null properties into a single string."""
    parts = [f"toString(COALESCE(n.{p}, ''))" for p in props]
    return " + ' ' + ".join(parts)


LABEL_CONFIGS: List[Dict] = [
    {
        "label": "Document",
        "index_name": "document_embeddings",
        "fetch_cypher": (
            "MATCH (n:Document) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("document_title", "document_type", "document_date", "name", "description")
            + " AS text"
        ),
    },
    {
        "label": "LegalAct",
        "index_name": "legalact_embeddings",
        "fetch_cypher": (
            "MATCH (n:LegalAct) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("act_type", "act_number", "act_year", "title", "subject")
            + " AS text"
        ),
    },
    {
        "label": "Article",
        "index_name": "article_embeddings",
        "fetch_cypher": (
            "MATCH (n:Article) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("heading", "text_en")
            + " AS text"
        ),
    },
    {
        "label": "Clause",
        "index_name": "clause_embeddings",
        "fetch_cypher": (
            "MATCH (n:Clause) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("text_en")
            + " AS text"
        ),
    },
    # No Section entry: see BE_OWNED_LABELS.
    {
        "label": "Institution",
        "index_name": "institution_embeddings",
        "fetch_cypher": (
            "MATCH (n:Institution) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("name", "type")
            + " AS text"
        ),
    },
    {
        "label": "Person",
        "index_name": "person_embeddings",
        "fetch_cypher": (
            "MATCH (n:Person) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("name", "role")
            + " AS text"
        ),
    },
    {
        "label": "Company",
        "index_name": "company_embeddings",
        "fetch_cypher": (
            "MATCH (n:Company) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("name", "legal_form")
            + " AS text"
        ),
    },
    {
        "label": "Court",
        "index_name": "court_embeddings",
        "fetch_cypher": (
            "MATCH (n:Court) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("name")
            + " AS text"
        ),
    },
    {
        "label": "CourtCase",
        "index_name": "courtcase_embeddings",
        "fetch_cypher": (
            "MATCH (n:CourtCase) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("case_number", "title")
            + " AS text"
        ),
    },
    {
        "label": "LegalParty",
        "index_name": "legalparty_embeddings",
        "fetch_cypher": (
            "MATCH (n:LegalParty) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("display_name")
            + " AS text"
        ),
    },
    {
        "label": "Tender",
        "index_name": "tender_embeddings",
        "fetch_cypher": (
            "MATCH (n:Tender) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("title", "subject", "issuer")
            + " AS text"
        ),
    },
    {
        "label": "Award",
        "index_name": "award_embeddings",
        "fetch_cypher": (
            "MATCH (n:Award) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("title", "recommendation_text")
            + " AS text"
        ),
    },
    {
        "label": "Contract",
        "index_name": "contract_embeddings",
        "fetch_cypher": (
            "MATCH (n:Contract) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("contract_id", "title")
            + " AS text"
        ),
    },
    {
        "label": "ChangeOrder",
        "index_name": "changeorder_embeddings",
        "fetch_cypher": (
            "MATCH (n:ChangeOrder) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("reason")
            + " AS text"
        ),
    },
    {
        "label": "Auction",
        "index_name": "auction_embeddings",
        "fetch_cypher": (
            "MATCH (n:Auction) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("title", "terms", "conditions")
            + " AS text"
        ),
    },
    {
        "label": "Penalty",
        "index_name": "penalty_embeddings",
        "fetch_cypher": (
            "MATCH (n:Penalty) "
            "RETURN elementId(n) AS eid, "
            + _text_expr("type", "reason")
            + " AS text"
        ),
    },
]

_WRITE_EMBEDDING_QUERY = (
    "UNWIND $batch AS item "
    "MATCH (n) WHERE elementId(n) = item.eid "
    "SET n.embedding = item.embedding"
)


def _init_embedding_model():
    provider = os.getenv("EMBEDDING_PROVIDER", "auto").lower()
    base_url = os.getenv("EMBEDDING_BASE_URL")
    api_key = os.getenv("EMBEDDING_API_KEY", os.getenv("LLM_API_KEY", os.getenv("OPENAI_API_KEY")))

    def _openai():
        from langchain_openai import OpenAIEmbeddings
        kwargs: Dict = {
            "model": os.getenv("EMBEDDING_MODEL", os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")),
            "api_key": api_key,
            **embedding_config.openai_embedding_kwargs(),
        }
        if base_url:
            kwargs["base_url"] = base_url
        logger.info("Using OpenAI embeddings: model=%s", kwargs["model"])
        return OpenAIEmbeddings(**kwargs)

    def _local():
        from langchain_huggingface import HuggingFaceEmbeddings
        model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-large-en-v1.5")
        logger.info("Using local HuggingFace embeddings: model=%s", model_name)
        return HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    if provider == "local":
        return _local()
    if provider == "openai":
        return _openai()
    if base_url or os.getenv("OPENAI_API_KEY"):
        return _openai()
    return _local()


def _create_vector_indexes(driver, database: str, configs: List[Dict]) -> None:
    with driver.session(database=database) as session:
        for cfg in configs:
            cypher = (
                f"CREATE VECTOR INDEX {cfg['index_name']} IF NOT EXISTS "
                f"FOR (n:{cfg['label']}) ON (n.embedding) "
                f"OPTIONS {{indexConfig: {{"
                f" `vector.dimensions`: {VECTOR_DIMENSIONS},"
                f" `vector.similarity_function`: 'cosine'"
                f"}}}}"
            )
            session.run(cypher)
            logger.info("Ensured vector index: %s", cfg["index_name"])


def _process_label(driver, database: str, cfg: Dict, embedding_model, batch_size: int) -> int:
    label = cfg["label"]

    with driver.session(database=database) as session:
        records = list(session.run(cfg["fetch_cypher"]))

    if not records:
        logger.info("%s: no nodes need embeddings", label)
        return 0

    eids = [r["eid"] for r in records]
    texts = [_add_doc_prefix(r["text"].strip() or label) for r in records]
    logger.info("%s: embedding %d nodes", label, len(records))

    total_written = 0
    for i in range(0, len(texts), batch_size):
        batch_eids = eids[i : i + batch_size]
        batch_texts = texts[i : i + batch_size]

        embeddings = embedding_model.embed_documents(batch_texts)
        batch = [{"eid": eid, "embedding": emb} for eid, emb in zip(batch_eids, embeddings)]

        with driver.session(database=database) as session:
            session.run(_WRITE_EMBEDDING_QUERY, batch=batch)

        total_written += len(batch)
        logger.info("%s: wrote batch %d–%d", label, i, i + len(batch) - 1)

    return total_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate embeddings for Neo4j PascalCase nodes.")
    parser.add_argument("--database", default="neo4j", help="Neo4j database name (overrides NEO4J_DATABASE env)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE, help="Embedding batch size")
    parser.add_argument("--label", help="Process only this label (default: all except the section labels)")
    parser.add_argument("--create-indexes", action="store_true", help="Create vector indexes before embedding")
    args = parser.parse_args()

    if args.label in BE_OWNED_LABELS:                 # refused before any connection
        parser.error(BE_OWNED_MESSAGE)
    if not args.label:
        logger.info("Skipping %s. %s", ", ".join(BE_OWNED_LABELS), BE_OWNED_MESSAGE)

    database = args.database if args.database != "neo4j" else os.getenv("NEO4J_DATABASE", "neo4j")

    neo4j_user = os.getenv("NEO4J_USER", "neo4j")
    print(f"[debug] NEO4J_USER={neo4j_user!r}")

    from neo4j import GraphDatabase
    driver = GraphDatabase.driver(
        os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        auth=(neo4j_user, os.getenv("NEO4J_PASSWORD", "")),
    )

    embedding_model = _init_embedding_model()

    try:
        configs = (
            [cfg for cfg in LABEL_CONFIGS if cfg["label"] == args.label]
            if args.label
            else LABEL_CONFIGS
        )

        if args.label and not configs:
            logger.error(
                "Unknown label: %s. Available: %s",
                args.label,
                [c["label"] for c in LABEL_CONFIGS],
            )
            return

        total_written = 0
        for cfg in configs:
            total_written += _process_label(driver, database, cfg, embedding_model, args.batch_size)

        if args.create_indexes:
            _create_vector_indexes(driver, database, configs)

        logger.info("Done. Total embeddings written: %d", total_written)
    finally:
        driver.close()


if __name__ == "__main__":
    main()
