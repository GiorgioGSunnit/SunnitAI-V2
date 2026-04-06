import json
from typing import List

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


def write_kg_from_extracted(path, database: str = "neo4j") -> tuple:
    """Read a JSONL file of extracted node/rel records and write them to Neo4j.

    Each line must be either:
      - a node record:  {"label": str, "key": {…}, "properties": {…}}
      - a rel record:   {"type": str, "from": {"label", "key"}, "to": {"label", "key"}, "properties": {…}}

    The ``database`` parameter is accepted for interface compatibility but is not
    forwarded to DBClient, which reads the target database from environment variables.

    Returns (n_written, r_written).
    """
    from pathlib import Path

    path = Path(path)
    n_written = 0
    r_written = 0

    with DBClient() as db:
        with path.open("r", encoding="utf-8") as f:
            for raw in f:
                raw = raw.strip()
                if not raw:
                    continue
                rec = json.loads(raw)
                if "from" in rec:
                    # Relationship record
                    from_label = rec["from"]["label"]
                    from_key: dict = rec["from"]["key"]
                    to_label = rec["to"]["label"]
                    to_key: dict = rec["to"]["key"]
                    rel_type = rec["type"]
                    props = rec.get("properties") or {}

                    from_where = " AND ".join(f"src.{k} = $f_{k}" for k in from_key)
                    to_where = " AND ".join(f"tgt.{k} = $t_{k}" for k in to_key)
                    params: dict = {f"f_{k}": v for k, v in from_key.items()}
                    params.update({f"t_{k}": v for k, v in to_key.items()})
                    params["props"] = props

                    cypher = (
                        f"MATCH (src:{from_label}) WHERE {from_where} "
                        f"MATCH (tgt:{to_label}) WHERE {to_where} "
                        f"MERGE (src)-[r:{rel_type}]->(tgt) SET r += $props"
                    )
                    db.query(cypher, params)
                    r_written += 1
                else:
                    # Node record
                    label = rec["label"]
                    key: dict = rec["key"]
                    props = rec.get("properties") or {}

                    merge_props = ", ".join(f"{k}: ${k}" for k in key)
                    params = {**key, "props": props}

                    cypher = f"MERGE (n:{label} {{{merge_props}}}) SET n += $props"
                    db.query(cypher, params)
                    n_written += 1

    return n_written, r_written


if __name__ == "__main__":
    write_indexing()
