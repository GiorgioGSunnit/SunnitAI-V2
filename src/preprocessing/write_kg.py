import json
from pathlib import Path
from typing import List, Tuple

from ..utils.db_client import DBClient
from .schema.schema import SCHEMA_CONSTRAINTS_AND_INDEXES, KnowledgeGraphExtraction


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
