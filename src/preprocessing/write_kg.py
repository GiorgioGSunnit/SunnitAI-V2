import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

from ..utils.db_client import DBClient
from .schema.schema import SCHEMA_CONSTRAINTS_AND_INDEXES, KnowledgeGraphExtraction

logger = logging.getLogger(__name__)


def save_to_neo4j(extraction: KnowledgeGraphExtraction) -> None:
    """
    Salva nodi e relazioni estratti dal LLM in Neo4j.

    Args:
        extraction: Oggetto KnowledgeGraphExtraction contenente nodes e relationships
    """
    with DBClient() as db:
        # 1. Salva nodi
        for node in extraction.nodes:
            cypher = f"""
            MERGE (n:{node.label} {{id: $id}})
            SET n.embedding_text = $embedding_text,
                n.page = $page,
                n += $properties
            """
            db.query(
                cypher,
                {
                    "id": node.id,
                    "embedding_text": node.embedding_text,
                    "page": node.page,
                    "properties": node.properties,
                },
            )

        # 2. Salva relazioni
        for rel in extraction.relationships:
            cypher = f"""
            MATCH (source {{id: $source_id}})
            MATCH (target {{id: $target_id}})
            MERGE (source)-[r:{rel.type}]->(target)
            SET r += $properties
            """
            db.query(
                cypher,
                {
                    "source_id": rel.source_id,
                    "target_id": rel.target_id,
                    "properties": rel.properties,
                },
            )


def write_indexing():
    """
    Crea constraints e indexes in Neo4j come definito in SCHEMA_CONSTRAINTS_AND_INDEXES.
    """
    with DBClient() as db:
        # Splitta le query per linee e esegui quelle non vuote
        queries = [
            q.strip()
            for q in SCHEMA_CONSTRAINTS_AND_INDEXES.split(";")
            if q.strip() and not q.strip().startswith("//")
        ]

        for query in queries:
            # Rimuovi i commenti inline
            query_lines = [
                line
                for line in query.split("\n")
                if line.strip() and not line.strip().startswith("//")
            ]
            clean_query = "\n".join(query_lines)

            if clean_query:
                db.query(clean_query)
        print("done")


LABEL_MIGRATIONS: Dict[str, str] = {
    "DOCUMENT_SECTION": "Section",
    "LEGAL_DOC": "Document",
    "LEGAL_SOURCE": "LegalAct",
    "LEGAL_CONCEPT": "Topic",
    "LEGAL_ACTION": "Penalty",
    "ORGANIZATION": "Institution",
    "PERSON": "Person",
    "LOCATION": "Section",
    "ROLE": "Topic",
    "DATE": "Section",
    "EVENT": "Meeting",
}


def _to_pascal(label: str) -> str:
    return "".join(word.capitalize() for word in label.split("_"))


def relabel_legacy_nodes(driver, database: str = "neo4j") -> Dict[str, Tuple[str, int]]:
    """Migrate fully-uppercase node labels to PascalCase equivalents.

    Detects all labels in the database where label = toUpper(label), maps each
    to its PascalCase equivalent via LABEL_MIGRATIONS (with auto-conversion fallback),
    then re-labels nodes in place using SET/REMOVE.

    Args:
        driver:   A neo4j.Driver instance.
        database: Neo4j database name.

    Returns:
        Dict of {old_label: (new_label, migrated_count)} for every label
        where at least one node was migrated.
    """
    detect_query = (
        "MATCH (n) UNWIND labels(n) AS label "
        "WITH label WHERE label = toUpper(label) "
        "RETURN DISTINCT label"
    )
    with driver.session(database=database) as session:
        uppercase_labels = [r["label"] for r in session.run(detect_query)]

    migrated: Dict[str, Tuple[str, int]] = {}
    for old_label in uppercase_labels:
        new_label = LABEL_MIGRATIONS.get(old_label) or _to_pascal(old_label)
        if new_label == old_label:
            continue
        migrate_query = (
            f"MATCH (n:`{old_label}`) SET n:`{new_label}` REMOVE n:`{old_label}` "
            "RETURN count(n) AS migrated"
        )
        with driver.session(database=database) as session:
            record = session.run(migrate_query).single()
            count = record["migrated"] if record else 0
        if count > 0:
            logger.info("Relabelled %d nodes: %s → %s", count, old_label, new_label)
            migrated[old_label] = (new_label, count)
    return migrated


def write_kg_from_extracted(path, database: str = "neo4j") -> Tuple[int, int]:
    """Read a JSONL file produced by _write_extracted_jsonl and MERGE its records into Neo4j.

    Each line is either a node record {"label", "key", "properties"} or a
    relationship record {"type", "from": {"label", "key"}, "to": {"label", "key"}, "properties"}.

    Returns:
        (n_written, r_written): count of node and relationship records processed.
    """
    path = Path(path)
    node_records: List[dict] = []
    rel_records: List[dict] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "type" in rec and "from" in rec:
                rel_records.append(rec)
            else:
                node_records.append(rec)

    n_written = 0
    r_written = 0

    with DBClient() as db:
        for rec in node_records:
            label = rec["label"]
            key: dict = rec["key"]
            props = rec.get("properties", {})
            key_clause = ", ".join(f"{k}: ${k}" for k in key)
            cypher = f"MERGE (n:{label} {{{key_clause}}}) SET n += $props"
            db.query(cypher, {**key, "props": props})
            n_written += 1

        for rec in rel_records:
            rel_type = rec["type"]
            from_rec = rec["from"]
            to_rec = rec["to"]
            props = rec.get("properties", {})
            from_key: dict = from_rec["key"]
            to_key: dict = to_rec["key"]
            from_clause = ", ".join(f"{k}: $from_{k}" for k in from_key)
            to_clause = ", ".join(f"{k}: $to_{k}" for k in to_key)
            cypher = (
                f"MATCH (src:{from_rec['label']} {{{from_clause}}}) "
                f"MATCH (tgt:{to_rec['label']} {{{to_clause}}}) "
                f"MERGE (src)-[r:{rel_type}]->(tgt) "
                "SET r += $props"
            )
            db.query(cypher, {
                **{f"from_{k}": v for k, v in from_key.items()},
                **{f"to_{k}": v for k, v in to_key.items()},
                "props": props,
            })
            r_written += 1

    return n_written, r_written


if __name__ == "__main__":
    write_indexing()
